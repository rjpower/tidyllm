# TidyLLM Tools and Utilities Reference

TidyLLM is a clean tool management system for LLMs that provides various tools and utilities for common tasks. This document enumerates all available tools and utilities in the project.

## Core Architecture

TidyLLM uses a registry-based system where tools are automatically registered and can be discovered, executed, and integrated into various interfaces (CLI, FastAPI, FastMCP).

### Key Components

- **Registry System** (`src/tidyllm/registry.py`): Global registry for tools with execution capabilities
- **Function Schemas** (`src/tidyllm/schema.py`): Automatic schema generation from function signatures
- **Context Management** (`src/tidyllm/context.py`): Dependency injection for tool context
- **Database Layer** (`src/tidyllm/database.py`): SQLite operations with schema inspection
- **CLI Generation** (`src/tidyllm/cli.py`): Automatic CLI generation from function signatures
- **LLM Integration** (`src/tidyllm/llm.py`): LLM client abstraction with tool calling support
- **Caching System** (`src/tidyllm/cache.py`): Function result caching with database storage

## Tools (`src/tidyllm/tools/`)

### 1. Anki Flashcard Management (`anki.py`)

Comprehensive Anki flashcard management tool for language learning and memorization.

**Functions:**
- `anki_query(args)` - Search notes in Anki database by query text
- `anki_create(args)` - Create flashcard decks using genanki
- `anki_list()` - List all available Anki decks with card counts

**Key Features:**
- Auto-discovery of Anki database on macOS
- Custom vocabulary card templates with CSS styling
- Support for audio files and example sentences
- SQLite database queries with custom collations
- Export to .apkg format

**Models:**
- `AnkiCard` - Vocabulary card with source word, translation, examples, audio
- `AnkiQueryArgs/Result` - Search functionality
- `AnkiCreateArgs/Result` - Deck creation

### 2. Calculator (`calculator/`)

Basic mathematical operations tool with structured input/output.

**Functions:**
- `calculator(args)` - Perform add, subtract, multiply, divide operations

**Features:**
- Type-safe operations with Pydantic models
- Division by zero protection
- Human-readable expression output
- Modular architecture with separate lib file

**Models:**
- `CalculatorArgs` - Operation type and operands
- `CalculatorResult` - Result with expression format

### 3. Configuration Management (`config.py`)

System configuration for TidyLLM tools with auto-discovery capabilities.

**Features:**
- Environment variable support with TIDYLLM_ prefix
- Auto-discovery of Anki database path
- Configurable notes directory and user database
- Model selection (fast/slow) configuration

**Settings:**
- `notes_dir` - Documents/Notes directory
- `user_db` - SQLite database path
- `anki_path` - Auto-discovered Anki database
- `fast_model/slow_model` - Gemini model configuration

### 4. Database Management (`manage_db.py`)

Safe database operations for user data management.

**Functions:**
- `db_query(args)` - Execute SELECT queries safely
- `db_execute(args)` - Execute INSERT/UPDATE/DELETE statements
- `db_list_tables(args)` - List all database tables
- `db_schema(args)` - Get schema information

**Security Features:**
- Restricted to safe operations only
- Prevents dangerous operations (DROP, TRUNCATE, ALTER)
- Parameterized queries for SQL injection protection
- Schema inspection capabilities

### 5. Notes Management (`notes.py`)

Comprehensive note-taking system with markdown and frontmatter support.

**Functions:**
- `note_add(args)` - Add new notes with frontmatter
- `note_search(args)` - Search notes by content and filename
- `note_list(args)` - List notes with tag filtering
- `note_open(args)` - Open and display notes
- `note_recent(args)` - List recently modified notes
- `note_tags(args)` - List all unique tags

**Features:**
- YAML frontmatter parsing for metadata
- Tag-based organization
- Full-text search using ripgrep
- Filename sanitization and collision handling
- Content preview generation

**Models:**
- `Note` - Complete note with metadata
- Various Args/Result models for each operation

### 6. Audio Transcription (`transcribe.py`)

Audio transcription using Gemini models via litellm.

**Functions:**
- `transcribe(args)` - Transcribe audio with translation support

**Features:**
- Multi-format audio support (MP3, WAV, M4A, OGG, etc.)
- Language detection and translation
- Structured JSON output with word-level breakdown
- Base64 encoding for API transmission

**Models:**
- `TranscribedWord` - Word with translation and timing
- `TranscribeArgs` - Audio file, language, translation target
- `TranscriptionResult` - Full transcription with metadata

### 7. Vocabulary Management (`vocab_table.py`)

Database-backed vocabulary learning system.

**Functions:**
- `vocab_add(args)` - Add vocabulary words with examples and tags
- `vocab_search(args)` - Search vocabulary by word, translation, or tag
- `vocab_update(args)` - Update existing vocabulary entries
- `vocab_delete(args)` - Delete vocabulary words

**Features:**
- JSON storage for examples and tags
- Flexible search with partial matching
- Tag-based categorization
- Timestamped entries with update tracking

**Models:**
- `VocabItem` - Complete vocabulary entry
- CRUD operation models for each function

### 8. Tool Context (`context.py`)

Shared context system for all tools.

**Features:**
- Dependency injection for configuration and database
- Automatic initialization with default Config
- Database connection management

## Core Utilities (`src/tidyllm/`)

### Agent System (`agent.py`)

LLM agent with conversation management and tool calling capabilities.

**Features:**
- Multi-round conversations with tool execution
- Rich console display with live updates
- Task completion callbacks
- Comprehensive logging with JSON snapshots
- Tool call execution with error handling

**Classes:**
- `LLMAgent` - Main agent with conversation flow
- `TaskStatus` - Task completion tracking
- `RichStreamWriter` - Enhanced UI for agent interactions

### Cache System (`cache.py`)

Function result caching with database persistence.

**Decorators:**
- `@cached_function` - Cache synchronous function results
- `@async_cached_function` - Cache asynchronous function results

**Features:**
- SHA256 hashing of function arguments
- Automatic cache table creation
- Pydantic model serialization support
- Context-aware caching (no-op without database)

### CLI Generation (`cli.py`)

Automatic CLI generation from function signatures.

**Functions:**
- `generate_cli(func)` - Create Click CLI for a function
- `cli_main(func)` - Single function CLI execution
- `multi_cli_main(functions)` - Multi-command CLI with subcommands

**Features:**
- Automatic option generation from Pydantic models
- JSON input support for complex arguments
- Type conversion and validation
- Error handling with JSON output

### Context Management (`context.py`)

Dependency injection system using contextvars.

**Functions:**
- `get_tool_context()` - Retrieve current tool context
- `set_tool_context(context)` - Context manager for setting context

**Features:**
- Thread-safe context management
- Generic context manager implementation
- Runtime error handling for missing context

### Database Layer (`database.py`)

SQLite database abstraction with schema inspection.

**Features:**
- Connection management with lazy initialization
- Schema inspection with Pydantic models
- Row factory for dictionary-like access
- Query and mutation methods with parameter binding

### Discovery System (`discover.py`)

Auto-discovery of tools in directories and packages.

**Functions:**
- `discover_tools_in_directory(directory)` - Find tools in filesystem
- `discover_tools_in_package(package)` - Import and discover from Python packages

**Features:**
- Pattern-based file filtering
- Recursive directory scanning
- Exclude pattern support
- Dynamic import and registration

### LLM Integration (`llm.py`)

LLM client abstraction with tool calling support.

**Features:**
- Role-based message system
- Tool call execution framework
- Streaming output support
- Response formatting and parsing

### Registry System (`registry.py`)

Global tool registry with execution capabilities.

**Features:**
- Function registration with schema generation
- Tool execution with JSON serialization
- Error handling and validation
- Schema export for LLM integration

## Database Schema

TidyLLM automatically creates and manages database tables for:

- **Vocabulary** (`vocab`) - Words, translations, examples, tags
- **Cache Tables** (dynamic) - Function result caching
- **User Data** - Extensible schema for tool-specific data

## Integration Points

### FastAPI Adapter (`adapters/fastapi_adapter.py`)
- REST API endpoints for all registered tools
- Automatic OpenAPI schema generation
- Request/response validation

### FastMCP Adapter (`adapters/fastmcp_adapter.py`)
- Model Context Protocol (MCP) server implementation
- Tool discovery and execution via MCP

### CLI Interface
- Automatic command generation
- JSON input/output support
- Multi-tool dispatch

## Configuration

TidyLLM uses environment variables with the `TIDYLLM_` prefix:

- `TIDYLLM_NOTES_DIR` - Notes directory path
- `TIDYLLM_USER_DB` - User database path
- `TIDYLLM_ANKI_PATH` - Anki database path
- `TIDYLLM_FAST_MODEL` - Fast model for simple tasks
- `TIDYLLM_SLOW_MODEL` - Slow model for complex tasks

## Usage Examples

### Tool Registration
```python
from tidyllm import register

@register()
def my_tool(args: MyArgs) -> MyResult:
    """My custom tool."""
    return MyResult(data="processed")
```

### CLI Usage
```bash
# Single tool
python -m my_tool --word "hello" --translation "hola"

# Multi-tool
python -m my_tools vocab_add --word "hello" --translation "hola"
```

### Agent Usage
```python
from tidyllm.agent import LLMAgent
from tidyllm import REGISTRY

agent = LLMAgent(REGISTRY, llm_client, "gemini/gemini-2.5-flash")
response = agent.ask_with_conversation(messages)
```

This reference provides a comprehensive overview of TidyLLM's tools and utilities. Each component is designed to work together in a cohesive system for LLM tool management and execution.