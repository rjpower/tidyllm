Of course. This is an excellent task, as it involves architectural thinking and merging the best of both worlds. Here is a detailed specification document designed for a junior engineer to implement these features in App B (TidyLLM).

---

## **Specification Document: Integrating Persistent Chat and an Interactive REPL into TidyLLM**

### **1. Overview**

This document outlines the plan to integrate two key features from the `TidyApp` framework (App A) into the `TidyLLM` library (App B):

1.  **Persistent, Database-backed Chat Sessions:** A `ChatManager` to replace the current file-based logging with a robust, queryable history of conversations stored in the application's SQLite database.
2.  **Integrated Interactive REPL:** A new, rich command-line interface (`TidyREPL`) for developers and power users to interact with the TidyLLM agent, execute code, and manage chat sessions.

This will elevate `TidyLLM` from a simple tool library into a more powerful, stateful, and developer-friendly environment.

### **2. Goals**

*   To provide users with a persistent, reviewable history of their interactions with the LLM agent.
*   To enable multi-turn conversations where the agent can recall previous messages and tool outputs.
*   To create a new, powerful interactive entry point for TidyLLM for debugging, development, and advanced scripting.
*   To implement these features using TidyLLM's existing architectural patterns (SQLite, `contextvars`, Pydantic models).

#### **Non-Goals**

*   This spec does not cover building a web interface or WebSocket support.
*   We will use App B's existing SQLite database; we are not migrating to DuckDB.
*   The existing `cli.py` for individual tool execution will remain; the REPL is a new, separate interface.

### **3. Elegant Restatement of the Architecture**

Currently, TidyLLM's `LLMAgent` is a stateless processor: it takes a list of messages, gets a one-shot response, and logs it. The proposed architecture introduces statefulness through the `ChatManager`, which becomes the central orchestrator for conversations. The `TidyREPL` acts as a rich, interactive client to this new system.

**New Architectural Flow:**

```
+----------------+      +----------------+      +---------------------+      +-------------+
|      User      |----->|   TidyREPL     |----->|     ChatManager     |----->|   LLMAgent  |
+----------------+      | (Interactive   |      | (State & History)   |      | (Execution) |
                        |   Shell)       |      +----------+----------+      +-------+-----+
                        +-------+--------+                 |                         |
                                |                          |                         |
       Python Code Execution <--+                          |        LLM/Tools <------+
                                |                          v
                       Command Handling (`/chat`)     +-----------------+
                                                     | SQLite Database |
                                                     | (chat_sessions, |
                                                     |  chat_messages) |
                                                     +-----------------+
```

**Key Principles of the New Architecture:**

*   **Stateful Core:** The `ChatManager` is the new stateful core of the application. It owns the conversation history.
*   **Decoupled Agent:** The `LLMAgent` becomes a stateless execution engine. It receives a full message history from the `ChatManager` for each turn and returns the new messages, but it no longer manages the conversation log itself.
*   **Rich Client:** The `TidyREPL` is the primary interface for this stateful system, providing users with commands to control and observe the chat state.
*   **Unified Persistence:** All chat data resides in the existing SQLite database, co-located with other application data like vocabulary.

---

### **4. Implementation Plan**

This will be broken down into two main features.

#### **Feature 1: Persistent Chat History (`ChatManager`)**

This involves creating a new module, `chat.py`, and modifying `agent.py`.

##### **A. New Files/Modules to Add**

Create a new file: `src/tidyllm/chat.py`. This file will contain the data models and the `ChatManager` class.

##### **B. Data Models (in `src/tidyllm/chat.py`)**

Define Pydantic models to represent sessions and stored messages.

```python
# In src/tidyllm/chat.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from tidyllm.llm import LLMMessage  # Import from existing llm.py
from tidyllm.agent import LLMAgent
from tidyllm.database import Database

class ChatSession(BaseModel):
    """Represents a single, persistent conversation session."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    model: str = "gemini/gemini-2.5-flash"
    system_prompt: str = "You are a helpful assistant."
    created_at: datetime = Field(default_factory=datetime.now)

class StoredMessage(BaseModel):
    """A wrapper for LLMMessage to be stored in the database."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    message: LLMMessage  # This will hold UserMessage, AssistantMessage, etc.
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("message", mode="before")
    @classmethod
    def parse_message_json(cls, v: Any) -> Any:
        """Handles parsing the message from a JSON string from the DB."""
        if isinstance(v, str):
            return json.loads(v)
        return v
```

##### **C. Database Schema Updates (in `src/tidyllm/database.py`)**

Modify the `init_schema` method in `Database` to create the new tables.

```python
# In src/tidyllm/database.py, inside the Database class's init_schema method:

def init_schema(self) -> None:
    # ... (existing table creation for vocab, etc.)

    # Add these new tables for chat history
    self.mutate('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    self.mutate('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL, -- Storing the message as a JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
    ''')

    self.mutate('''
        CREATE INDEX IF NOT EXISTS idx_messages_session_created
        ON chat_messages(session_id, created_at)
    ''')
```

##### **D. `ChatManager` Class (in `src/tidyllm/chat.py`)**

This class will manage all interactions with chat sessions.

```python
# In src/tidyllm/chat.py
from tidyllm.context import get_tool_context

class ChatManager:
    """Manages chat sessions and interactions with the LLM agent."""

    def __init__(self, agent: LLMAgent):
        self.agent = agent
        # The database is now retrieved from the context when needed.
        # This aligns with TidyLLM's context-based architecture.

    def _get_db(self) -> Database:
        """Retrieves the database connection from the current tool context."""
        return get_tool_context().db

    def new_session(self, system_prompt: str | None = None) -> ChatSession:
        """Creates and persists a new chat session."""
        db = self._get_db()
        session = ChatSession(system_prompt=(system_prompt or "You are a helpful assistant."))
        # Persist to DB
        db.mutate(
            "INSERT INTO chat_sessions (id, model, system_prompt) VALUES (?, ?, ?)",
            (session.id, session.model, session.system_prompt)
        )
        return session

    def list_sessions(self) -> list[ChatSession]:
        """Returns a list of all chat sessions from the database."""
        db = self._get_db()
        rows = db.query("SELECT * FROM chat_sessions ORDER BY created_at DESC").all()
        return [ChatSession(**row) for row in rows]

    def _load_messages(self, session_id: str) -> list[LLMMessage]:
        """Loads all messages for a given session from the database."""
        db = self._get_db()
        rows = db.query(
            "SELECT message FROM chat_messages WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).all()
        # The StoredMessage model's validator will handle the JSON string parsing
        return [StoredMessage(session_id=session_id, message=row['message']).message for row in rows]

    def _store_message(self, session_id: str, message: LLMMessage):
        """Stores a single message into the database."""
        db = self._get_db()
        # The message model needs to be converted to a JSON string for storage
        stored_msg = StoredMessage(session_id=session_id, message=message)
        message_json_str = stored_msg.message.model_dump_json()
        db.mutate(
            "INSERT INTO chat_messages (id, session_id, message) VALUES (?, ?, ?)",
            (stored_msg.id, session_id, message_json_str)
        )

    async def send_message(self, session_id: str, content: str) -> LLMMessage:
        """
        The main interaction method. It loads history, calls the agent,
        stores the new messages, and returns the agent's final response.
        """
        history = self._load_messages(session_id)
        user_message = UserMessage(content=content)
        self._store_message(session_id, user_message)
        history.append(user_message)

        # Call the agent with the full history
        response = self.agent.ask(history) # Assuming ask is now async

        # The `ask` method returns an LLMResponse object. We need the last message.
        assistant_message = response.messages[-1]
        self._store_message(session_id, assistant_message)

        return assistant_message
```

##### **E. Integration into `LLMAgent`**

The `LLMAgent` in `src/tidyllm/agent.py` will be simplified. It will no longer manage conversation history itself. Its `ask` and `ask_with_conversation` methods will be replaced or simplified.

*   **`LLMAgent.ask`:** This method should now accept a full message history and simply execute one round of LLM completion and tool calling. The `ChatManager` will be responsible for building this history.
*   **Deprecate `LLMLogger`:** The `LLMLogger`'s role is superseded by the `ChatManager`. It can be removed or kept for low-level debugging only.

#### **Feature 2: Integrated Interactive REPL**

This involves creating a new `repl.py` module and a new entry point.

##### **A. New Files/Modules to Add**

Create a new file: `src/tidyllm/repl.py`.

##### **B. New Dependencies**

Add `rich` to `pyproject.toml` for styled console output.

```toml
# In pyproject.toml, under [project.dependencies]
dependencies = [
    # ... existing dependencies
    "rich>=13.0.0",
]
```
Run `uv sync` to install it.

##### **C. `TidyREPL` Class (in `src/tidyllm/repl.py`)**

This class will manage the interactive loop, command parsing, and chat state.

```python
# In src/tidyllm/repl.py
import asyncio
from rich.console import Console
from rich.markdown import Markdown

from tidyllm.chat import ChatManager
from tidyllm.agent import LLMAgent
from tidyllm.context import ToolContext, set_tool_context
from tidyllm.tools.config import Config

class TidyREPL:
    """An interactive REPL for TidyLLM."""

    def __init__(self):
        self.console = Console()
        # Initialize context, agent, and chat manager
        self.context = ToolContext(config=Config())
        self.agent = LLMAgent(function_library=None, llm_client=None, model='default') # Configure appropriately
        self.chat_manager = ChatManager(self.agent)
        self.chat_mode = False
        self.current_session_id = None

    async def run(self):
        """The main loop for the REPL."""
        self.console.print("[bold blue]TidyLLM Interactive REPL[/bold blue]")
        self.console.print("Type [green]/help[/green] for commands.")
        self.current_session_id = self.chat_manager.new_session().id
        self.console.print(f"New session started: [cyan]{self.current_session_id[:8]}[/cyan]")

        while True:
            prompt = "[yellow]chat>[/yellow] " if self.chat_mode else ">>> "
            user_input = self.console.input(prompt)

            if user_input.startswith('/'):
                await self._handle_command(user_input)
                continue

            if self.chat_mode:
                if user_input.lower() == 'exit':
                    self.chat_mode = False
                    self.console.print("[green]Exited chat mode.[/green]")
                    continue
                await self._handle_chat(user_input)
            else:
                if user_input.lower() == 'exit':
                    break
                self._handle_python(user_input)

    async def _handle_command(self, command: str):
        """Handles special slash commands."""
        cmd, *args = command.strip().split()
        if cmd == '/help':
            # Print help text for /chat, /exit, /history, /sessions, etc.
            self.console.print("Available commands: /chat, /exit, /history, /sessions, /new, /help")
        elif cmd == '/chat':
            self.chat_mode = True
            self.console.print("[green]Entered chat mode. Type 'exit' to return to Python REPL.[/green]")
        # ... implement other commands like /history, /sessions ...

    async def _handle_chat(self, message: str):
        """Sends a message to the ChatManager and prints the response."""
        if not self.current_session_id:
            self.console.print("[red]No active session.[/red]")
            return
        
        with self.console.status("Agent is thinking..."):
            response = await self.chat_manager.send_message(self.current_session_id, message)
        
        self.console.print(f"\n[bold]Assistant:[/bold]")
        self.console.print(Markdown(response.content))

    def _handle_python(self, code: str):
        """Executes Python code in the REPL's context."""
        try:
            # Note: A real implementation needs a safe execution context.
            # For this spec, we'll just show the concept.
            result = eval(code, {"__name__": "__console__"})
            self.console.print(result)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

def run_repl():
    """Entry point function to start the REPL."""
    # The REPL needs to be run within the tool context
    config = Config()
    context = ToolContext(config=config)
    with set_tool_context(context):
         repl = TidyREPL()
         asyncio.run(repl.run())
```

##### **D. New Entry Point**

Add a new script entry point in `pyproject.toml`.

```toml
# In pyproject.toml
[project.scripts]
# ... existing scripts
tidyllm-repl = "tidyllm.repl:run_repl"
```
After this change, the REPL can be started with `uv run tidyllm-repl`.

### **5. Summary of Changes to App B**

*   **Files to Add:**
    *   `src/tidyllm/chat.py`
    *   `src/tidyllm/repl.py`
*   **Files to Modify:**
    *   `src/tidyllm/database.py`: Add `CREATE TABLE` statements for `chat_sessions` and `chat_messages` to `init_schema`.
    *   `src/tidyllm/agent.py`: Simplify `LLMAgent` to be stateless. The `ask` method will now expect a full conversation history and not manage it internally. Remove `LLMLogger` or reduce its role.
    *   `pyproject.toml`: Add `rich` as a dependency and `tidyllm-repl` as a script entry point.
*   **Features to Extend:**
    *   The `Database` class is extended with new tables.
    *   The `LLMAgent` is extended to work with the `ChatManager`.
*   **Features to Replace:**
    *   The file-based `LLMLogger` is effectively replaced by the database-backed `ChatManager` for conversation history.

### **6. Testing Strategy**

1.  **Unit Tests for `ChatManager`:** Create `tests/test_chat.py`.
    *   Test `new_session()` creates a record in the database.
    *   Test `list_sessions()` retrieves all sessions.
    *   Test `_store_message()` and `_load_messages()` correctly serialize and deserialize `LLMMessage` objects to/from JSON.
    *   Mock the `LLMAgent` and test `send_message()` to ensure it correctly loads history, calls the agent, and stores the new messages.
2.  **Manual Testing for `TidyREPL`:**
    *   Run `uv run tidyllm-repl`.
    *   Verify the welcome message and prompt.
    *   Test `/help` command.
    *   Switch to chat mode with `/chat`.
    *   Have a multi-turn conversation.
    *   Exit chat mode with `exit`.
    *   Execute a simple Python command (e.g., `1+1`).
    *   Verify that after restarting the REPL, the previous session's history can be loaded (once the `/history` and `/sessions` commands are implemented).

---
This specification provides a clear, step-by-step path for a junior engineer to implement these powerful new features, bringing the best of App A's stateful interaction model into the clean, tool-focused architecture of App B.