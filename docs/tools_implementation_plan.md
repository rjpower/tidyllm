# TidyLLM Tools Implementation Plan

## Overview
Implement 6 tools as single-file modules following the calculator pattern, each with:
- Pydantic models for args/results
- Tool registration using `@register` decorator
- CLI integration via `cli_main`

## Architecture Analysis

### Key Components
1. **Tool Registration**: `@register` decorator from `tidyllm.registry`
2. **LLM Integration**: `litellm` client in `tidyllm.llm.py`
3. **CLI Generation**: `cli_main` function from `tidyllm.cli`
4. **Schema**: Pydantic models for validation, auto-generates JSON schemas

### Pattern
Each tool file should:
```python
from pydantic import BaseModel
from tidyllm.registry import register
from tidyllm.cli import cli_main

class ToolArgs(BaseModel):
    # Input fields

class ToolResult(BaseModel):
    # Output fields

@register
def tool_name(args: ToolArgs) -> ToolResult:
    # Implementation

if __name__ == "__main__":
    cli_main(tool_name)
```

## Tools to Implement

### 1. Anki Tool (`anki.py`)
**Purpose**: Read vocab items from Anki database, create cards
**Models**:
- `AnkiReadArgs`: deck_name, query filters
- `AnkiCreateArgs`: source_word, translated_word, examples, audio paths
- `AnkiResult`: success status, card_id/vocab_items

Details:

* Read from the standard Anki database location, e.g. 

$HOME/Library/Application Support/Anki2/*/collection.anki2

- YOu'll use your understanding of the Anki DB format to analyze the DB and extract data, writing helpers to do so or pulling packages from PypI as needed.

* When creating cards, use https://pypi.org/project/genanki/ to generate decks, using a standard template that you define.
* Read this URL for an example: https://raw.githubusercontent.com/rjpower/multivox/refs/heads/main/server/multivox/flashcards/generate_anki.py

### 2. Vocab Table Tool (`vocab_table.py`)
**Purpose**: Manage vocabulary table in user database
**Models**:
- `VocabArgs`: operations (add/update/delete/query)
- `VocabItem`: word, translation, examples, created_at
- `VocabResult`: items list, operation status

**Details**

Create a db.py which manages access to the user DB.
User DB by default lives in ~/.config/user.db as sqlite
Write a reasonable schema
Provide reasonable search filters

### 3. User DB Tool (`manage_db.py`)
**Purpose**: General database operations
**Models**:
- `DBArgs`: operation, table_name, query/data
- `DBResult`: rows, affected_count, tables_list

### 4. Transcription Tool (`transcribe.py`)
**Purpose**: Transcribe audio using Gemini Flash via litellm
**Models**:
- `TranscribeArgs`: audio_file_path, language
- `TranscriptionResult`: sentence, words list with translations

**Integration**: Register with LLM system for tool calls

**Questions**:
- Audio format requirements (mp3, wav, anything supported by gemini)
- Language detection vs specification (specified, autodetect if not)
- Translation service integration (have gemini transcribe & translate as part of the structured output)

### 5. User Notes Tool (`notes.py`)
**Purpose**: Add and search user notes
**Models**:
- `NotesArgs`: operation (add/search), content, tags
- `NotesResult`: notes list, search results

**Questions**:
- Note storage format: markdown in $NOTES_DIR, default to ~/Documents/Notes
- Full-text search requirements (yes, just use ilike or whatever)
- Tag system design (assume frontmatter contains tags)

## Implementation Order
1. User DB (foundation)
2. Vocab Table (uses DB)
3. Notes (uses DB)
4. Anki (external integration)
5. Transcription (LLM integration)

## Technical Decisions

Define a shared context & config environment for all the tools:

config.py:

class Config(PydanticSettings):
  notes_dir: Path = "$HOME/..."
  user_db: Path = "$HOME/..."
  anki_path: Path|None # autodiscover if needed
  fast_model: str = "gemini/gemini-2.5-flash"
  slow_model: str = "gemini/gemini-2.5-pro"

context.py:

## shared context 

passed to all of the tools, loaded by cli.py
A dummy context can be used with the FunctionLibrary during tests to point to temp files or a test DB.

class ToolContext(BaseModel):
  config: Config = Config()
  user_db: sqlite3.Connection

Add helper methods to the config/context as needed, all tools should indirect through the context 
instead of relying on constants or env vars.

e.g. 

@register
def tool(... *, ctx: ToolContext):
  ctx.important_path.read_text()

If you need to, you can have the tool accept a protocol to avoid circular dependencies, e.g.

```
# tool.py
# tool that only needs the config, not anything else
class MyToolContext(Protocol):
  config: "Config"
```
  
1. **Database**: SQLite for user data (portable, simple) 
2. **Anki Integration**: Use `anki` Python package if available
3. **Audio Handling**: Take paths to binary data
4. **Error Handling**: Consistent error model across tools, just use exceptions, let @llm.py handle it
5. **Testing**: Use a real sqlite db, inject into the context.

## Dependencies
- `sqlite3` (built-in)
- `anki` or manual SQLite for Anki
- `litellm` (already present)
- Audio processing library (if needed)

## Next Steps
1. Confirm database design decisions
2. Implement user_db.py as foundation
3. Build other tools on top of DB infrastructure
4. Add comprehensive error handling
5. Write tests for each tool

## Detailed Implementation Section

### File Structure
```
src/tidyllm/tools/
├── __init__.py
├── config.py           # Configuration management
├── context.py          # Shared context for all tools
├── db.py              # Database utilities and connection management
├── anki.py            # Anki tool implementation
├── vocab_table.py     # Vocabulary table tool
├── manage_db.py       # General database operations tool
├── transcribe.py      # Audio transcription tool
└── notes.py           # User notes tool
```

### Implementation Details

#### 1. **config.py**
```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    notes_dir: Path = Path.home() / "Documents" / "Notes"
    user_db: Path = Path.home() / ".config" / "user.db"
    anki_path: Path | None = None  # Will autodiscover
    fast_model: str = "gemini/gemini-2.0-flash-exp"
    slow_model: str = "gemini/gemini-2.0-flash-thinking-exp-1219"
    
    model_config = {
        "env_prefix": "TIDYLLM_",
        "env_file": ".env"
    }
```

#### 2. **context.py**
```python
import sqlite3
from typing import Protocol
from pydantic import BaseModel, Field
from .config import Config

class ToolContext(BaseModel):
    config: Config = Field(default_factory=Config)
    
    class Config:
        arbitrary_types_allowed = True
        
    def get_db_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        # Ensure parent directory exists
        self.config.user_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.config.user_db))
        conn.row_factory = sqlite3.Row
        return conn
        
    def ensure_notes_dir(self) -> Path:
        """Ensure notes directory exists and return it."""
        self.config.notes_dir.mkdir(parents=True, exist_ok=True)
        return self.config.notes_dir
```

#### 3. **db.py**
```python
# Database utilities
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol

class DBContext(Protocol):
    config: Config
    
def init_database(ctx: DBContext) -> None:
    """Initialize database with required tables."""
    db_path = ctx.config.user_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Vocab table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL UNIQUE,
            translation TEXT NOT NULL,
            examples TEXT,  -- JSON array
            tags TEXT,      -- JSON array
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Notes metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            title TEXT,
            tags TEXT,      -- JSON array
            content_preview TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
```

#### 4. **anki.py**
```python
# Two separate tools for read/create operations
class AnkiReadArgs(BaseModel):
    deck_name: str
    limit: int = 100
    tags: list[str] | None = None

class AnkiReadResult(BaseModel):
    cards: list[dict]
    deck_name: str
    count: int

class AnkiCreateArgs(BaseModel):
    deck_name: str
    cards: list[AnkiCard]
    
class AnkiCard(BaseModel):
    source_word: str
    translated_word: str
    examples: list[str]
    audio_path: Path | None = None

class AnkiCreateResult(BaseModel):
    success: bool
    deck_path: Path
    cards_created: int

@register
def anki_read(args: AnkiReadArgs, *, ctx: ToolContext) -> AnkiReadResult:
    # Implementation

@register  
def anki_create(args: AnkiCreateArgs, *, ctx: ToolContext) -> AnkiCreateResult:
    # Implementation using genanki
```

#### 5. **vocab_table.py**
```python
class VocabArgs(BaseModel):
    operation: Literal["add", "update", "delete", "query"]
    word: str | None = None
    translation: str | None = None
    examples: list[str] | None = None
    tags: list[str] | None = None
    filters: dict[str, Any] | None = None

class VocabResult(BaseModel):
    success: bool
    items: list[VocabItem] | None = None
    message: str | None = None
    
class VocabItem(BaseModel):
    id: int
    word: str
    translation: str
    examples: list[str]
    tags: list[str]
    created_at: datetime
    updated_at: datetime

@register
def vocab_table(args: VocabArgs, *, ctx: ToolContext) -> VocabResult:
    # Implementation using ctx.get_db_connection()
```

#### 6. **manage_db.py**
```python
class DBArgs(BaseModel):
    operation: Literal["query", "execute", "list_tables", "schema"]
    sql: str | None = None
    params: dict[str, Any] | None = None

class DBResult(BaseModel):
    success: bool
    rows: list[dict] | None = None
    affected_count: int | None = None
    tables: list[str] | None = None
    error: str | None = None

@register
def manage_db(args: DBArgs, *, ctx: ToolContext) -> DBResult:
    # Safe database operations
```

#### 7. **transcribe.py**
```python
class TranscribeArgs(BaseModel):
    audio_file_path: Path
    language: str | None = None  # Auto-detect if not provided
    translate_to: str = "en"

class TranscriptionResult(BaseModel):
    transcription: str
    language: str
    words: list[TranscribedWord]
    
class TranscribedWord(BaseModel):
    word: str
    translation: str | None = None
    start_time: float | None = None
    end_time: float | None = None

@register
def transcribe(args: TranscribeArgs, *, ctx: ToolContext) -> TranscriptionResult:
    # Use litellm with Gemini for transcription
    # Return structured output with translations
```

#### 8. **notes.py**
```python
class NotesArgs(BaseModel):
    operation: Literal["add", "search", "list"]
    content: str | None = None
    tags: list[str] | None = None
    title: str | None = None
    query: str | None = None

class NotesResult(BaseModel):
    success: bool
    notes: list[Note] | None = None
    message: str | None = None
    
class Note(BaseModel):
    file_path: Path
    title: str
    tags: list[str]
    content_preview: str
    created_at: datetime
    updated_at: datetime

@register
def notes(args: NotesArgs, *, ctx: ToolContext) -> NotesResult:
    # Markdown files with frontmatter
    # Store metadata in user_db for fast search
```

### Updated Architecture Notes

Based on analysis of `library.py` and `cli.py`:

1. **Context Injection**: The `@register` decorator already supports context injection. Tools that need context should declare a `ctx` parameter as keyword-only (after `*`). The FunctionLibrary will automatically inject the context when calling the tool.

2. **CLI Context Creation**: The CLI already handles context creation for Pydantic models. For tools with context, CLI options are prefixed with `--ctx-`. The CLI will create the context instance and pass it to the tool.

3. **Multiple Functions per File**: For tools like Anki that have multiple operations (read/create), we register multiple functions. Each gets its own CLI command when using `cli_main()`.

4. **Error Handling**: Tools should raise exceptions. The FunctionLibrary catches exceptions and converts them to ToolError instances.

5. **Database Schema Decisions**:
   - Vocab table: id, word (unique), translation, examples (JSON), tags (JSON), created_at, updated_at
   - Notes table: Store metadata in DB for fast search, actual content in markdown files
   - Use JSON columns for lists (examples, tags) for flexibility

6. **Protocol Types**: Tools can declare minimal Protocol types for their context requirements to avoid circular dependencies while maintaining type safety.

### CLI Multi-Command Support

The current `cli_main` accepts a single function. For tools with multiple operations:
```python
# In anki.py
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        cli_main(anki_create)
    else:
        cli_main(anki_read)
```

Or we can enhance cli_main to accept multiple functions and generate subcommands.