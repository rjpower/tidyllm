# Claude Desktop Integration for TidyLLM

This guide shows how to integrate TidyLLM tools with Claude Desktop using the FastMCP adapter.

## Prerequisites

1. Install TidyLLM with FastMCP support:
   ```bash
   uv add tidyllm[fastmcp]
   # or
   pip install tidyllm
   ```

2. Make sure Claude Desktop is installed on your system.

## Configuration

### Step 1: Locate Claude Desktop Configuration

Find your Claude Desktop configuration file:

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Step 2: Add TidyLLM MCP Server

Add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tidyllm": {
      "command": "uv",
      "args": ["run", "tidyllm-mcp"],
      "cwd": "/path/to/tidyllm/source/directory",
      "env": {
        "TIDYLLM_NOTES_DIR": "/path/to/your/notes",
        "TIDYLLM_USER_DB": "/path/to/your/tidyllm.db"
      }
    }
  }
}
```

**Important**: Set `cwd` to the directory where TidyLLM is installed/cloned so the server can find all dependencies properly.

#### Alternative Configuration (using pip)

If you installed TidyLLM with pip instead of uv:

```json
{
  "mcpServers": {
    "tidyllm": {
      "command": "python",
      "args": ["-m", "tidyllm.adapters.fastmcp_adapter"],
      "cwd": "/path/to/tidyllm/source/directory",
      "env": {
        "TIDYLLM_NOTES_DIR": "/path/to/your/notes",
        "TIDYLLM_USER_DB": "/path/to/your/tidyllm.db"
      }
    }
  }
}
```

#### Alternative: Using fastmcp run

For development or testing, you can also use `fastmcp run` directly:

```json
{
  "mcpServers": {
    "tidyllm": {
      "command": "uv",
      "args": ["run", "fastmcp", "run", "src/tidyllm/adapters/fastmcp_adapter.py"],
      "cwd": "/path/to/tidyllm/source/directory",
      "env": {
        "TIDYLLM_NOTES_DIR": "/path/to/your/notes",
        "TIDYLLM_USER_DB": "/path/to/your/tidyllm.db"
      }
    }
  }
}
```

### Step 3: Customize Environment Variables (Optional)

You can customize TidyLLM behavior using environment variables:

- `TIDYLLM_NOTES_DIR`: Directory for storing notes (default: `~/Documents/Notes`)
- `TIDYLLM_USER_DB`: Path to user database (default: `~/.config/tidyllm/user.db`)
- `TIDYLLM_ANKI_PATH`: Path to Anki database (auto-discovered if not set)
- `TIDYLLM_FAST_MODEL`: Fast model name (default: `gemini/gemini-2.0-flash-exp`)
- `TIDYLLM_SLOW_MODEL`: Slow model name (default: `gemini/gemini-2.0-flash-thinking-exp-1219`)

Example with custom settings:

```json
{
  "mcpServers": {
    "tidyllm": {
      "command": "uv",
      "args": ["run", "tidyllm-mcp"],
      "cwd": "/Users/yourname/code/tidyllm",
      "env": {
        "TIDYLLM_NOTES_DIR": "/Users/yourname/MyNotes",
        "TIDYLLM_USER_DB": "/Users/yourname/.config/tidyllm/my_vocab.db",
        "TIDYLLM_FAST_MODEL": "gpt-4o-mini",
        "TIDYLLM_SLOW_MODEL": "gpt-4o"
      }
    }
  }
}
```

### Step 4: Restart Claude Desktop

After adding the configuration:

1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The TidyLLM tools should now be available

## Available Tools

Once configured, you'll have access to these TidyLLM tools in Claude Desktop:

### Vocabulary Management
- **vocab_add**: Add new vocabulary words with translations and examples
- **vocab_search**: Search through your vocabulary database
- **vocab_update**: Update existing vocabulary entries
- **vocab_delete**: Remove vocabulary words

### Note Management
- **note_add**: Create new notes with markdown and frontmatter
- **note_search**: Search through your notes
- **note_list**: List notes with optional tag filtering

### Database Management
- **db_query**: Execute SELECT queries on your database
- **db_execute**: Execute INSERT/UPDATE/DELETE statements
- **db_list_tables**: List all database tables
- **db_schema**: Get database schema information

### Anki Integration
- **anki_read**: Read cards from Anki decks
- **anki_create**: Create new Anki decks with flashcards

### Audio Transcription
- **transcribe**: Transcribe audio files using AI models

## Usage Examples

### Adding Vocabulary
```
Please add the Spanish word "biblioteca" meaning "library" to my vocabulary with some example sentences.
```

### Taking Notes
```
Create a note titled "Meeting Notes" with the tag "work" containing today's discussion points.
```

### Searching Notes
```
Find all my notes related to "Python programming".
```

### Creating Anki Cards
```
Create an Anki deck called "Spanish Basics" with cards for the words I've been learning.
```

## Troubleshooting

### Server Not Starting
1. Check that TidyLLM is properly installed: `uv run tidyllm-mcp --help`
2. Verify the paths in your configuration are correct
3. Check Claude Desktop logs for error messages

### Permission Issues
Make sure the directories specified in `TIDYLLM_NOTES_DIR` and `TIDYLLM_USER_DB` are writable by the user running Claude Desktop.

### Tool Not Responding
1. Restart Claude Desktop
2. Check that the MCP server is configured correctly
3. Verify environment variables are set properly

## Development Mode

For development, you can run the server directly to test:

```bash
# Run the MCP server
uv run tidyllm-mcp

# Or with custom config
TIDYLLM_NOTES_DIR=/tmp/notes uv run tidyllm-mcp
```

The server will start and wait for MCP client connections via STDIO.