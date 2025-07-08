"""LLM integration helper for TidyAgent tools."""

import datetime
import json
from dataclasses import field
from enum import Enum
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from tidyllm.function_schema import JSONSchema


class StreamWriter(Protocol):
    """Protocol for handling streaming output from LLM clients."""

    def write(self, content: str) -> None:
        """Write content to the stream."""
        ...

    def update_status(self, status: str) -> None:
        """Update status display."""
        ...

    def update_messages(self, messages: list[Any]) -> None:
        """Update messages for display."""
        ...


class PrintWriter:
    """Default console writer for LLM streaming output."""

    def write(self, content: str) -> None:
        print(content, end="", flush=True)

    def update_status(self, status: str) -> None:
        print(f"[STATUS] {status}")  # Print status with newline and prefix

    def update_messages(self, messages: list[Any]) -> None:
        pass


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    tool_name: str
    tool_args: dict[str, Any]
    id: str | None = None


class UserMessage(BaseModel):
    role: Role = Role.USER
    content: str

    def llm_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
        }


class SystemMessage(BaseModel):
    role: Role = Role.SYSTEM
    content: str

    def llm_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
        }


class ToolMessage(BaseModel):
    role: Role = Role.TOOL
    content: str
    tool_call_id: str
    name: str

    def llm_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }


class AssistantMessage(BaseModel):
    role: Role = Role.ASSISTANT
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage_data: dict[str, int] | None = None

    def llm_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.tool_args),
                    },
                }
                for tc in self.tool_calls
            ],
        }


LLMMessage = UserMessage | SystemMessage | ToolMessage | AssistantMessage


class LLMResponse(BaseModel):
    """Response from LLM with tool calling details."""

    model: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    messages: list[LLMMessage]
    tools: list[dict] = field(default_factory=list)


class LiteLLMClient:
    def completion(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[JSONSchema],
        temperature: float = 0.1,
        timeout_seconds: int = 30,
        writer: StreamWriter | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Get completion using LiteLLM with streaming."""
        if writer is None:
            writer = PrintWriter()

        writer.update_status("Calling LLM...")
        writer.write(f"[DEBUG] Model: {model}\n")
        writer.write(f"[DEBUG] Tools available: {len(tools)}\n")
        writer.write(f"[DEBUG] Messages count: {len(messages)}\n")

        # Convert LLMMessage objects to dict format for LiteLLM
        message_dicts = []
        for msg in messages:
            msg_dict = msg.llm_dict()
            message_dicts.append(msg_dict)

        # Use streaming mode like tinyagent
        import litellm
        from litellm.types.utils import ModelResponseStream

        response = litellm.completion(
            model=model,
            messages=message_dicts,
            tools=tools,
            temperature=temperature,
            timeout=timeout_seconds,
            stream=True,
            stream_options={"include_usage": True},
            tool_choice="auto",
            **kwargs,
        )

        content_parts = []
        tool_calls_by_index = {}
        usage_data = None

        for chunk in cast(litellm.CustomStreamWrapper, response):
            chunk = cast(ModelResponseStream, chunk)
            choice = chunk.choices[0]
            if choice.delta.role == "user":
                continue

            if choice.delta.content is not None:
                content = choice.delta.content
                content_parts.append(content)
                writer.write(content)

            # Handle tool calls
            if choice.delta.tool_calls:
                writer.write("[DEBUG] Tool call delta received\n")
                for tool_call_delta in choice.delta.tool_calls:
                    index = tool_call_delta.index
                    if index not in tool_calls_by_index:
                        writer.write(f"[DEBUG] New tool call at index {index}\n")
                        tool_calls_by_index[index] = {
                            "id": tool_call_delta.id,
                            "type": tool_call_delta.type,
                            "function": {"name": None, "arguments": ""},
                        }

                    tool_call = tool_calls_by_index[index]

                    # Accumulate function arguments
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                    # Update name if provided
                    if tool_call_delta.function and tool_call_delta.function.name:
                        tool_call["function"]["name"] = tool_call_delta.function.name
                        writer.write(
                            f"[DEBUG] Tool call name: {tool_call_delta.function.name}\n"
                        )

            # Handle usage data
            if hasattr(chunk, "usage") and chunk.usage is not None:  # type: ignore
                # Convert Usage object to dict for Pydantic validation
                usage_obj = chunk.usage  # type: ignore
                usage_data = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    "total_tokens": getattr(usage_obj, "total_tokens", 0),
                }

        tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())]
        assistant_msg = AssistantMessage(
            content="".join(content_parts), usage_data=usage_data
        )

        # Convert tool calls to ToolCall objects
        processed_tool_calls = []
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])
                processed_tool_calls.append(
                    ToolCall(tool_name=tool_name, tool_args=tool_args, id=tc.get("id"))
                )
            assistant_msg.tool_calls = processed_tool_calls

        response_messages = messages + [assistant_msg]

        response = LLMResponse(
            model=model,
            messages=response_messages,
            tools=tools,
        )

        # Don't log here - logging will happen in LLMHelper after tool execution
        return response


T = TypeVar("T", bound=BaseModel)


def completion_with_schema(
    model: str,
    messages: list[dict[str, str]],
    response_schema: type[T],
    **kwargs,
) -> T:
    """Wrapper for litellm.completion with Pydantic schema response format.
    
    Args:
        model: Model name to use
        messages: List of messages in OpenAI format
        response_schema: Pydantic model class to use for response format
        **kwargs: Additional arguments passed to litellm.completion
        
    Returns:
        Parsed response as instance of response_schema
    """
    import litellm

    response = litellm.completion(
        model=model,
        messages=messages,
        response_format=response_schema,
        **kwargs,
    )

    message_content = cast(litellm.Choices, response.choices)[0].message.content  # type: ignore
    assert message_content is not None, "Response content is None"
    return response_schema.model_validate_json(message_content)
