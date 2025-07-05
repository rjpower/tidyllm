# File and Object Manipulation Tools Specification

**Version**: 1.0  
**Author**: Claude Code  

## Executive Summary

This specification defines tools for Unix-style piping between tidyllm tools, enabling workflows like `uv run tts.py generate_speech --format=pickle | object.getattr '.audio_bytes' > out.wav`.

## Requirements

### Object Tool (object.py)
- **getattr**: Extract fields using dot notation (`.field.subfield`) or bracket notation (`[0]`, `['key']`)
- **getitem**: Array/dict access with bracket notation
- Handle missing fields with `--default` or `--strict` flags

### File Tool (file.py) 
- **read**: Read file contents, output to stdout
- **write**: Write binary data from stdin to file (always binary mode)

### CLI Format Support
- `--format=json|pickle|raw` flag for all tools
- `json`: Current JSON output (default)
- `pickle`: Binary pickle data for preserving Python objects
- `raw`: String representation via `str(result)`

## Data Models & API

```python
# object.py
@register()
def getattr(path: str, default: Any = None, strict: bool = True) -> Any:
    """Extract field from stdin data using dot/bracket notation."""

@register()  
def getitem(key: str | int) -> Any:
    """Extract item by key/index from stdin data."""

# file.py
@register()
def read(path: str) -> bytes:
    """Read file contents as binary."""

@register()
def write(path: str) -> str:
    """Write binary data from stdin to file."""

# CLI adapter changes
@click.option("--format", type=click.Choice(["json", "pickle", "raw"]), default="json")
def cli_wrapper(format: str, **kwargs):
    result = execute_function(**kwargs)
    if format == "json":
        sys.stdout.write(json.dumps(result))
    elif format == "pickle":
        sys.stdout.buffer.write(pickle.dumps(result))
    elif format == "raw":
        sys.stdout.write(str(result))
```

## Usage Examples

```bash
# Extract audio bytes and save to file
uv run tts.py generate_speech --content="Hello" --format=pickle | object.getattr '.audio_bytes' > audio.mp3

# Extract nested field
echo '{"user": {"name": "John"}}' | object.getattr '.user.name'

# Read file and extract config
file.read 'config.json' | object.getattr '.database.host'

# Array access
echo '{"items": [1,2,3]}' | object.getitem '[1]'  # Returns 2
```

## Implementation Plan

### Phase 1: Core Tools
1. Create `src/tidyllm/tools/object.py` with getattr/getitem functions
2. Create `src/tidyllm/tools/file.py` with read/write functions  
3. Add path parsing logic for dot/bracket notation

### Phase 2: CLI Integration
1. Update `src/tidyllm/adapters/cli.py` to support `--format` flag
2. Add stdin reading capability
3. Implement format detection and output serialization

### Phase 3: Testing
1. Unit tests for path parsing and field extraction
2. Integration tests for piping workflows
3. CLI tests for format handling

## Technical Details

### Path Parsing
- Support `.field.subfield` for nested object access
- Support `[0]` for array index, `['key']` for dict key
- Support mixed notation: `.items[0].name`

### Error Handling
- Missing paths: return default value or raise error based on `--strict` flag
- Invalid paths: always raise clear error message
- File operations: handle permissions and path issues gracefully

### Performance
- Stream processing for large files
- Lazy evaluation for object traversal
- Memory-efficient binary data handling

This simplified specification focuses on the core functionality needed for effective data piping between tidyllm tools.