import json
from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tidyllm.library import FunctionLibrary
from tidyllm.llm import (
    AssistantMessage,
    LLMClient,
    LLMMessage,
    LLMResponse,
    PrintWriter,
    Role,
    StreamWriter,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def create_request(system_prompt: str, user_prompt: str | None = None) -> list[LLMMessage]:
    """Create initial messages for LLMAgent.

    Args:
        system_prompt: System instruction
        user_prompt: Optional user message

    Returns:
        List of LLMMessage objects
    """
    messages: list[LLMMessage] = [SystemMessage(content=system_prompt)]
    if user_prompt:
        messages.append(UserMessage(content=user_prompt))
    return messages


class TaskStatusType(str, Enum):
    SUCCESS = "SUCCESS"
    INCOMPLETE = "INCOMPLETE"
    GIVE_UP = "GIVE_UP"


class TaskStatus(BaseModel):
    status: TaskStatusType = TaskStatusType.SUCCESS
    errors: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    def merge(
        self, other: "TaskStatus", status: TaskStatusType = TaskStatusType.INCOMPLETE
    ) -> "TaskStatus":
        """Merge another TaskStatus into this one."""
        merged_status = copy(self)
        merged_status.status = status
        merged_status.errors.extend(other.errors)
        merged_status.diagnostics.extend(other.diagnostics)
        return merged_status

    def diagnostic(self, message: str) -> None:
        """Add a diagnostic message to the task status."""
        self.diagnostics.append(message)

    def error(self, message: str) -> None:
        """Add an error message to the task status."""
        self.errors.append(message)
        if self.status == TaskStatusType.SUCCESS:
            self.status = TaskStatusType.INCOMPLETE

    def is_done(self) -> bool:
        """Check if the task is completed successfully or failed."""
        return self.status in [TaskStatusType.SUCCESS, TaskStatusType.GIVE_UP]

    def as_message(self) -> str:
        """Format the status response as a message suitable for the LLM agent."""
        if self.status == TaskStatusType.SUCCESS:
            return "Task completed successfully."

        return f"Task status: {self.status}. Errors: {'\n *'.join(self.errors)}.\nDiagnostics: {'\n *'.join(self.diagnostics)}"


def default_completion_callback(messages: list[LLMMessage], *, ctx: Any = None) -> TaskStatus:
    """Default completion callback that listens for agent responses like 'SUCCESS' or 'I GIVE UP'."""
    if not messages:
        return TaskStatus(status=TaskStatusType.INCOMPLETE)

    # Get the last assistant message
    last_message = None
    for msg in reversed(messages):
        if msg.role == Role.ASSISTANT:
            last_message = msg
            break

    if not last_message or not last_message.content:
        return TaskStatus(status=TaskStatusType.INCOMPLETE)

    content = last_message.content.upper()

    # Check for completion signals
    success_signals = ["SUCCESS", "COMPLETE", "FINISHED"]
    failure_signals = ["I GIVE UP", "FAILED", "ERROR", "CANNOT"]

    for signal in success_signals:
        if signal in content:
            return TaskStatus(status=TaskStatusType.SUCCESS)

    for signal in failure_signals:
        if signal in content:
            return TaskStatus(status=TaskStatusType.INCOMPLETE)

    return TaskStatus(status=TaskStatusType.INCOMPLETE)


@dataclass
class LogSnapshot:
    timestamp: str
    logger: "LLMLogger"
    writer: StreamWriter | None = None

    def log(self, response: LLMResponse):
        """Log a snapshot of the LLM response."""
        log_file = self.logger.log_dir / f"{self.timestamp}.json"
        print(f"[DEBUG] Logging snapshot to {log_file}")
        log_file.write_text(response.model_dump_json(indent=2))
        
        # Update the writer with current log file if it's a RichStreamWriter
        if isinstance(self.writer, RichStreamWriter):
            self.writer.update_log_file(log_file)


class LLMLogger:
    log_dir: Path

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # add __enter__ and __exit__ methods to use LLMLogger as a context manager, with exception handling  
    def __enter__(self):
        return LogSnapshot(
            timestamp=datetime.now().strftime("%Y-%m-%dT%H_%M_%S.%f")[:-3],
            logger=self,
            writer=getattr(self, '_current_writer', None),
        )
        
    def set_writer(self, writer: StreamWriter | None):
        """Set the current writer for log file updates."""
        self._current_writer = writer

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # Log the exception details
            import traceback as tb

            error_log = {
                "error": str(exc_value),
                "type": str(exc_type),
                "traceback": tb.format_tb(traceback) if traceback else [],
            }
            error_file = self.log_dir / f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            error_file.write_text(json.dumps(error_log, indent=2))
        return False


LLM_LOGGER = LLMLogger(Path("logs/litellm"))


class RichStreamWriter:
    """Rich console writer for enhanced LLM streaming output."""

    def __init__(
        self,
        live_display: Live,
        max_rounds: int,
        messages: list[LLMMessage],
    ):
        from rich.layout import Layout

        self.live_display = live_display
        self.round_num = 1  # Will be updated in the loop
        self.max_rounds = max_rounds
        self.messages = messages  # This will be updated via update_messages
        self.current_content = Text()
        self.status_text = Text()
        self.lines = [""]
        self.current_log_file = None

        # Create persistent panels and layout
        self.progress_panel = Panel(
            Text(f"Round 1/{max_rounds}", style="bold blue"),
            title="Conversation Progress",
            border_style="blue",
        )
        self.chat_panel = Panel(
            self._create_chat_table(), title="Chat State", border_style="green"
        )
        self.tool_panel = Panel(
            self._create_tool_table(), title="Recent Tool Calls", border_style="yellow"
        )
        self.log_panel = Panel(
            Text("No log file yet", style="dim"), title="Current Log File", border_style="magenta"
        )
        self.stream_panel = Panel("Ready...", title="LLM Response", border_style="cyan")

        # Create layout once
        top_row = Columns(
            [self.progress_panel, self.chat_panel, self.tool_panel, self.log_panel], equal=True
        )
        self.layout = Layout()
        self.layout.split_column(Layout(top_row, size=8), Layout(self.stream_panel))

    def write(self, content: str) -> None:
        self.lines[-1] += content
        if "\n" in self.lines[-1]:
            new_lines = self.lines[-1].split("\n")
            self.lines[-1] = new_lines[0]
            self.lines.extend(new_lines[1:])  # Add remaining lines as new lines
        self._update_display()

    def update_status(self, status: str) -> None:
        self.status_text = Text(status, style="blue")
        self._update_display()

    def update_messages(self, messages: list[LLMMessage]) -> None:
        self.messages = messages
        self._update_display()

    def update_log_file(self, log_file: Path | None) -> None:
        self.current_log_file = log_file
        self._update_display()

    def _create_chat_table(self) -> Table:
        """Create chat state table."""
        assistant_msgs = [m for m in self.messages if m.role == Role.ASSISTANT]
        user_msgs = [m for m in self.messages if m.role == Role.USER]
        tool_msgs = [m for m in self.messages if m.role == Role.TOOL]

        chat_table = Table(show_header=False, box=None)
        chat_table.add_row("Assistant messages:", f"{len(assistant_msgs)}")
        chat_table.add_row("User messages:", f"{len(user_msgs)}")
        chat_table.add_row("Tool responses:", f"{len(tool_msgs)}")
        return chat_table

    def _create_tool_table(self) -> Table:
        """Create tool calls table."""
        tool_table = Table(show_header=True, box=None)
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Result Summary", style="white")

        for message in self.messages:
            if isinstance(message, AssistantMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_table.add_row(tool_call.tool_name)

        return tool_table

    def _create_log_info(self) -> Text:
        """Create log file information display.""" 
        if not self.current_log_file:
            return Text("No log file yet", style="dim")
        
        log_text = Text()
        log_text.append(f"File: {self.current_log_file.name}\n", style="cyan")
        log_text.append(f"Path: {self.current_log_file.parent}\n", style="dim")
        
        if self.current_log_file.exists():
            size = self.current_log_file.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            log_text.append(f"Size: {size_str}", style="green")
        else:
            log_text.append("Not created yet", style="yellow")
            
        return log_text

    def _update_display(self) -> None:
        stream_content = Text()
        for line in self.lines[-50:]:
            stream_content.append(line + "\n")

        self.progress_panel.renderable = Text(
            f"Round {self.round_num}/{self.max_rounds}", style="bold blue"
        )
        self.chat_panel.renderable = self._create_chat_table()
        self.tool_panel.renderable = self._create_tool_table()
        self.log_panel.renderable = self._create_log_info()
        self.stream_panel.renderable = stream_content
        self.live_display.update(self.layout)


@dataclass
class LLMAgent:
    """An LLM agent processes multiple rounds with a model and performs automatic tool calling and logging."""

    function_library: FunctionLibrary
    llm_client: LLMClient
    model: str

    def __post_init__(self):
        """Copy the function library post init to allow registering new tools without modifying the original."""
        self.function_library = copy(self.function_library)

    def _execute_tool_calls(
        self, tool_calls: list[ToolCall], writer: StreamWriter
    ) -> list[LLMMessage]:
        """Execute tool calls and return tool response messages."""
        if not tool_calls:
            return []

        print(f"\n[DEBUG] Executing {len(tool_calls)} tool calls")
        tool_messages = []
        for i, tool_call in enumerate(tool_calls, 1):
            writer.update_status(
                f"Calling tool {tool_call.tool_name} ({i}/{len(tool_calls)})"
            )
            writer.write(
                f"[DEBUG] Executing {tool_call.tool_name} with args: {str(tool_call.tool_args)[:100]}..."
            )
            # Execute and get result
            result_str = self.function_library.call_with_json_response(
                tool_call.tool_name, tool_call.tool_args, tool_call.id
            )

            print(f"[DEBUG] Tool {tool_call.tool_name} finished.")
            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.tool_name,
                    content=result_str,
                )
            )

        return tool_messages

    def ask(
        self,
        prompt: list[LLMMessage],
        **llm_kwargs,
    ) -> LLMResponse:
        """Ask LLM to perform a task using available tools.

        Args:
            prompt: List of LLMMessage objects
            **llm_kwargs: Additional arguments passed to LLM client

        Returns:
            LLMResponse with tool call and execution details
        """
        tools = self.function_library.get_schemas()

        messages = prompt

        with LLM_LOGGER as logger:
            response = self.llm_client.completion(
                model=self.model, messages=messages, tools=tools, **llm_kwargs
            )

            writer = PrintWriter()

            # Execute any tool calls that were returned
            if response.messages:
                last_message = response.messages[-1]
                if (
                    isinstance(last_message, AssistantMessage)
                    and last_message.tool_calls
                ):
                    self._execute_tool_calls(last_message.tool_calls, writer=writer)

            logger.log(response)

        return response

    def ask_and_validate(
        self,
        prompt: list[LLMMessage],
        expected_tool: str,
        validation_fn: Callable | None = None,
        **llm_kwargs,
    ) -> LLMResponse:
        """Ask LLM and validate the response.

        Args:
            prompt: List of LLMMessage objects
            expected_tool: Expected tool name to be called
            validation_fn: Optional function to validate tool result
            **llm_kwargs: Additional LLM arguments

        Returns:
            LLMResponse with validation status
        """
        response = self.ask(prompt, **llm_kwargs)

        if validation_fn and not validation_fn(response):
            raise ValueError("Tool result validation failed")

        return response

    def ask_with_conversation(
        self,
        prompt: list[LLMMessage],
        max_rounds: int = 25,
        completion_callback: Callable[[list[LLMMessage]], TaskStatus] | None = None,
        console: Console | None = None,
        **llm_kwargs,
    ) -> LLMResponse:
        """Ask LLM with conversational flow allowing multiple tool calls.

        Args:
            prompt: List of LLMMessage objects
            max_rounds: Maximum conversation rounds (default 25)
            completion_callback: Optional callback to check if task is complete
            console: Optional rich Console for enhanced UI display
            **llm_kwargs: Additional arguments passed to LLM client

        Returns:
            LLMResponse with all tool calls and conversation history
        """
        messages = prompt

        # Set up display context - Live or nullcontext
        if console:
            display_context = Live(
                Panel("Initializing...", title="PortKit Agent", border_style="blue"),
                console=console,
                refresh_per_second=1,
            )
        else:
            display_context = nullcontext(None)

        with display_context as live:
            if console:
                writer = RichStreamWriter(live, max_rounds, messages)
                live.update(writer.layout)
            else:
                writer = PrintWriter()

            # Set the writer on the logger so it can update the display
            LLM_LOGGER.set_writer(writer)

            with LLM_LOGGER as logger:
                # Run conversation rounds
                for round_num in range(1, max_rounds + 1):
                    writer.update_status(
                        f"Starting LLM round {round_num} of {max_rounds}..."
                    )
                    if isinstance(writer, RichStreamWriter):
                        writer.round_num = round_num

                    # Get LLM response
                    response = self.llm_client.completion(
                        model=self.model,
                        messages=messages,
                        tools=self.function_library.get_schemas(),
                        writer=writer,
                        **llm_kwargs,
                    )

                    messages = response.messages
                    writer.update_messages(messages)

                    # Execute tool calls and add responses to conversation
                    last_message = response.messages[-1]
                    tool_calls = (
                        last_message.tool_calls
                        if isinstance(last_message, AssistantMessage)
                        else []
                    )
                    writer.update_status(f"Executing {len(tool_calls)} tool calls...")
                    tool_messages = self._execute_tool_calls(tool_calls, writer)
                    messages.extend(tool_messages)
                    writer.update_messages(messages)

                    # Check if task is finished using completion callback
                    if completion_callback:
                        task_status = completion_callback(messages)

                        if task_status.is_done():
                            print("[DEBUG] Task completed with status:", task_status.status)
                            break

                        feedback_msg = UserMessage(content=task_status.as_message())
                        messages.append(feedback_msg)
                        writer.update_messages(messages)

                    # Create updated response with full conversation for logging
                    response = LLMResponse(
                        messages=messages,
                        model=self.model,
                        tools=self.function_library.get_schemas(),
                    )

                    logger.log(response)

                return LLMResponse(
                    messages=messages,
                    model=self.model,
                    tools=self.function_library.get_schemas(),
                )
