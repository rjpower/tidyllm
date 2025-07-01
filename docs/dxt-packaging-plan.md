# TidyLLM DXT Packaging Plan

## Overview

This document outlines the requirements and implementation plan for packaging TidyLLM as a Desktop Extension (DXT) file. TidyLLM is a Python-based MCP (Model Context Protocol) server that provides AI tools for notes management, Anki flashcard creation, vocabulary management, transcription, and database operations.

## Current Codebase Analysis

### Project Structure
- **Type**: Python package with MCP server capabilities
- **Main Package**: `src/tidyllm/` 
- **Entry Point**: `tidyllm-mcp` script → `tidyllm.adapters.fastmcp_adapter:run_tidyllm_mcp_server`
- **Dependencies**: FastMCP, LiteLLM, Pydantic, Click, Rich, and domain-specific tools
- **Tools**: 5 main tool modules in `src/tidyllm/tools/`

### Available Tools
1. **Anki** (`anki.py`) - Flashcard creation and management
2. **Notes** (`notes.py`) - Markdown notes with frontmatter, search, and management
3. **Vocabulary Table** (`vocab_table.py`) - Vocabulary tracking and export
4. **Transcribe** (`transcribe.py`) - Audio transcription capabilities
5. **Database Management** (`manage_db.py`) - SQLite operations for user data

### Current MCP Implementation  
- Uses FastMCP server framework
- Tools auto-registered via `REGISTRY` system
- Context management via `ToolContext` and contextvars
- Configuration via `Config` class with environment variable support
- Entry point: `fastmcp_adapter.py:run_tidyllm_mcp_server()`

## DXT Requirements Analysis

### Required Files

1. **manifest.json** - Extension metadata and configuration (checked into git)
2. **scripts/dxt_main.py** - DXT entry point script
3. **src/tidyllm/** - Existing source code structure
4. **requirements.txt** - Generated during packaging
5. Optional: icon.png, README.md

### Manifest.json Structure

Based on DXT spec v0.1, using our existing repo structure:

```json
{
  "dxt_version": "0.1",
  "name": "tidyllm",
  "display_name": "TidyLLM Tools",
  "version": "0.1.0",
  "description": "AI-powered tools for notes, vocabulary, Anki cards, and transcription",
  "long_description": "TidyLLM provides a comprehensive suite of AI-enhanced productivity tools including intelligent note management with search, Anki flashcard generation, vocabulary tracking, audio transcription, and database operations. Perfect for language learners, researchers, and knowledge workers.",
  "author": {
    "name": "Russell Power",
    "email": "russell.power@gmail.com", 
    "url": "https://github.com/rjpower/tidyllm"
  },
  "server": {
    "type": "python",
    "entry_point": "scripts/dxt_main.py",
    "mcp_config": {
      "command": "python",
      "args": [
        "${__dirname}/scripts/dxt_main.py"
      ],
      "env": {
        "PYTHONPATH": "${__dirname}/src",
        "TIDYLLM_NOTES_DIR": "${user_config.notes_directory}",
        "TIDYLLM_USER_DB": "${user_config.database_path}",
        "TIDYLLM_ANKI_PATH": "${user_config.anki_path}",
        "TIDYLLM_FAST_MODEL": "${user_config.fast_model}",
        "TIDYLLM_SLOW_MODEL": "${user_config.slow_model}"
      }
    }
  },
  "tools": [
    {
      "name": "add_note",
      "description": "Add a new markdown note with optional frontmatter"
    },
    {
      "name": "search_notes",
      "description": "Search notes by content, title, or tags"
    },
    {
      "name": "create_anki_deck",
      "description": "Generate Anki flashcard deck from vocabulary data"  
    },
    {
      "name": "add_vocab_entry",
      "description": "Add vocabulary entry with translations and examples"
    },
    {
      "name": "transcribe_audio",
      "description": "Transcribe audio files to text"
    },
    {
      "name": "execute_sql",
      "description": "Execute SQL queries on user database"
    }
  ],
  "keywords": ["notes", "anki", "vocabulary", "transcription", "ai", "productivity", "learning"],
  "license": "MIT",
  "repository": "https://github.com/rjpower/tidyllm",
  "user_config": {
    "notes_directory": {
      "type": "directory",
      "title": "Notes Directory",
      "description": "Directory where markdown notes are stored",
      "default": "${HOME}/Documents/Notes",
      "required": false
    },
    "database_path": {
      "type": "file",
      "title": "User Database",
      "description": "SQLite database for user data",
      "default": "${HOME}/.config/tidyllm/user.db",
      "required": false
    },
    "anki_path": {
      "type": "file",
      "title": "Anki Database",
      "description": "Path to Anki collection.anki2 file (auto-discovered if empty)",
      "default": "",
      "required": false
    },
    "fast_model": {
      "type": "string",
      "title": "Fast AI Model",
      "description": "Model for quick operations",
      "default": "gemini/gemini-2.5-flash",
      "required": false
    },
    "slow_model": {
      "type": "string", 
      "title": "Slow AI Model",
      "description": "Model for complex operations",
      "default": "gemini/gemini-2.0-pro",
      "required": false
    }
  },
  "compatibility": {
    "claude_desktop": ">=0.10.0",
    "platforms": ["darwin", "win32", "linux"],
    "runtimes": {
      "python": ">=3.11.0 <4"
    }
  }
}
```

## Implementation Requirements

### 1. Scripts/package-dxt.py

Create a Python script that:

1. **Dependency Bundling**:
   - Run `uv export --format requirements-txt > requirements.txt`  
   - Create `server/lib/` directory
   - Install dependencies to `server/lib/` using `pip install -t server/lib/ -r requirements.txt`

2. **Server Entry Point**:
   - Create `server/main.py` that imports and runs the MCP server
   - Handle argument parsing for user configuration
   - Set up proper Python path for bundled dependencies

3. **Manifest Generation**:
   - Generate `manifest.json` with correct metadata from `pyproject.toml`
   - Include all tool definitions from registry
   - Set up user configuration options

4. **Archive Creation**:
   - Create zip file with `.dxt` extension
   - Include all required files in proper structure

### 2. scripts/dxt_main.py Structure

```python
#!/usr/bin/env python3
"""DXT entry point for TidyLLM MCP Server."""

from tidyllm.adapters.fastmcp_adapter import run_tidyllm_mcp_server

if __name__ == "__main__":
    # Config class automatically reads TIDYLLM_* environment variables
    # set by the DXT manifest user_config
    run_tidyllm_mcp_server()
```

### 3. Package Structure

```
tidyllm.dxt (ZIP file)
├── manifest.json           # Extension metadata (checked into git)
├── scripts/               # Entry point
│   └── dxt_main.py        # DXT entry point script
├── src/                   # Source code (existing structure)
│   └── tidyllm/           # Main package
├── requirements.txt       # Generated during packaging
└── README.md             # Optional documentation
```

### 4. Packaging Script Implementation

Key functions needed in `scripts/package-dxt.py`:

1. `extract_metadata()` - Read version/author from pyproject.toml
2. `generate_requirements()` - Run `uv export --format requirements-txt`
3. `copy_files_to_tempdir()` - Copy manifest.json, scripts/, src/, requirements.txt to temp directory
4. `create_dxt_archive()` - Zip everything into .dxt file
5. `discover_tools()` - Auto-detect available tools from registry (for manifest generation)

### 5. Environment Variables

The Config class already supports these environment variables via `TIDYLLM_` prefix:

- `TIDYLLM_NOTES_DIR` - Notes directory path
- `TIDYLLM_USER_DB` - User database path  
- `TIDYLLM_ANKI_PATH` - Anki database path
- `TIDYLLM_FAST_MODEL` - Fast AI model identifier
- `TIDYLLM_SLOW_MODEL` - Slow AI model identifier

## Testing Strategy

1. **Local Testing**: Build DXT and test with `dxt` CLI tools
2. **Integration Testing**: Install in Claude Desktop and verify tool functionality
3. **Cross-Platform**: Test on macOS, Windows, Linux
4. **Configuration**: Verify user config options work correctly
5. **Dependencies**: Ensure all bundled dependencies load properly

## File Dependencies

The following key files need to be available for the implementation:

- `pyproject.toml` - Project metadata and dependencies
- `src/tidyllm/` - Main package code  
- `src/tidyllm/adapters/fastmcp_adapter.py` - MCP server implementation
- `src/tidyllm/tools/` - All tool modules
- `src/tidyllm/registry.py` - Tool registration system

## Next Steps

1. Generate and check in `manifest.json` to git
2. Create `scripts/dxt_main.py` - simple entry point
3. Create `scripts/package-dxt.py` with the above functionality
4. Build and test the DXT package locally
5. Document installation and usage instructions

This simplified approach leverages:
- Existing repo structure (no server/lib bundling)
- Built-in Config class environment variable support (TIDYLLM_ prefix)
- Simple packaging script that copies files to tempdir and zips
- Manifest.json checked into git for version control