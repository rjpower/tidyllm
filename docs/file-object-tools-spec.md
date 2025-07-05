# File and Object Manipulation Tools Specification

**Version**: 1.0  
**Last Updated**: 2025-07-05  
**Author**: Claude Code  

## Executive Summary

This specification defines a comprehensive system for file and object manipulation tools that enables Unix-style piping between tidyllm tools. The system allows users to extract data from JSON objects, transform it, and write it to files through a composable command-line interface. This enables workflows like `uv run tts.py generate_speech | object.getattr '.audio_bytes' | file.write 'out.wav'`.

## Background & Context

### Current State
- tidyllm has a robust tool registration system using `@register()` decorator
- Tools return structured data as Pydantic models (e.g., `SpeechResult` with `audio_bytes`, `content`, etc.)
- CLI adapter automatically generates command-line interfaces from function signatures
- Output is currently JSON-formatted by default, making it difficult to extract specific fields for further processing

### Pain Points
- No mechanism to extract specific fields from JSON output
- No way to chain tools together via Unix pipes
- Binary data (like audio) must be manually decoded from base64 in JSON
- File operations require separate tools rather than composable utilities

### Dependencies
- Existing tool registry system (`src/tidyllm/registry.py`)
- CLI adapter system (`src/tidyllm/adapters/cli.py`)
- Pydantic models for structured data
- Click for CLI generation

## Functional Requirements

### FR-1: Object Attribute Access Tool
**Given**: A JSON object or Pydantic model output from stdin or file  
**When**: User executes `object.getattr '.field_name'` or `object.getitem '[0]'`  
**Then**: The tool extracts the specified field/index and outputs it to stdout

- Support dot notation for nested fields (`.field.subfield`)
- Support bracket notation for array/dict access (`[0]`, `['key']`)
- Support chained access (`.field[0].subfield`)
- Handle missing fields gracefully with configurable behavior (error, null, default)

### FR-2: File Write Tool
**Given**: Data from stdin or command-line arguments  
**When**: User executes `file.write 'path/to/file'`  
**Then**: The tool writes the data to the specified file path

- Support text and binary data
- Create parent directories if they don't exist
- Support append mode with `--append` flag
- Handle encoding for text files (default UTF-8)

### FR-3: File Read Tool
**Given**: A file path  
**When**: User executes `file.read 'path/to/file'`  
**Then**: The tool reads the file and outputs its contents to stdout

- Support text and binary files
- Handle encoding detection for text files
- Support partial reads with `--offset` and `--limit`
- Support base64 encoding for binary output

### FR-4: CLI Pickle Output Support
**Given**: Any tool with `--pickle` flag  
**When**: User executes `uv run tool.py func --pickle`  
**Then**: The tool outputs binary pickle data instead of JSON

- Preserve all Python object types including binary data
- Compatible with stdin/stdout piping
- Automatic detection of pickle vs JSON input

### FR-5: Data Type Conversion Tools
**Given**: Data of various types from stdin  
**When**: User executes conversion commands  
**Then**: Data is converted between formats (JSON, base64, binary, text)

- `data.to_base64`: Convert binary to base64
- `data.from_base64`: Convert base64 to binary  
- `data.to_json`: Convert Python object to JSON
- `data.from_json`: Convert JSON to Python object

## API Contracts and Code Examples

### Object Manipulation Tool

```python
@register()
def object_getattr(
    path: str,
    input_data: str | None = None,
    default: Any = None,
    strict: bool = True
) -> Any:
    """Extract attribute/item from JSON object using dot/bracket notation.
    
    Args:
        path: Dot notation path like '.field.subfield' or '[0].key'
        input_data: JSON string input (stdin if None)
        default: Default value if path not found
        strict: Raise error if path not found (vs return default)
        
    Returns:
        Extracted value
    """
```

**Usage Examples**:
```bash
# Extract audio bytes from TTS result
echo '{"audio_bytes": "base64data", "content": "hello"}' | object.getattr '.audio_bytes'

# Extract from nested object
echo '{"result": {"items": [{"name": "test"}]}}' | object.getattr '.result.items[0].name'

# With default value
echo '{}' | object.getattr '.missing_field' --default "not found"
```

### File Operations Tool

```python
@register()
def file_write(
    path: str,
    data: str | bytes | None = None,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> str:
    """Write data to file from stdin or argument.
    
    Args:
        path: File path to write to
        data: Data to write (stdin if None)
        create_dirs: Create parent directories if needed
        
    Returns:
        Success message with file path
    """

@register()
def file_read(
    path: str,
    encoding: str = "utf-8",
    binary: bool = False,
    base64_encode: bool = False
) -> str | bytes:
    """Read file contents.
    
    Args:
        path: File path to read
        encoding: Text encoding
        binary: Read as binary
        base64_encode: Encode binary as base64 string
        
    Returns:
        File contents
    """
```

**Usage Examples**:
```bash
# Write text to file
echo "Hello world" | file.write "output.txt"

# Write binary data
some_binary_command | file.write "output.bin" --mode wb

# Read file
file.read "input.txt"

# Read binary as base64
file.read "audio.mp3" --binary --base64-encode
```

### CLI Adapter Augmentations

```python
def _generate_cli_from_description(
    func_desc: FunctionDescription, 
    context_cls: type[BaseModel] | None = None
) -> click.Command:
    """Enhanced CLI generation with pickle support."""
    
    @click.command(name=func_desc.name)
    @click.option("--json", "json_input", help="JSON input for all arguments")
    @click.option("--pickle", "pickle_output", is_flag=True, help="Output as pickle instead of JSON")
    @click.option("--stdin", "use_stdin", is_flag=True, help="Read input from stdin")
    def cli(json_input: str | None, pickle_output: bool, use_stdin: bool, **kwargs):
        # Handle stdin input
        if use_stdin:
            stdin_data = sys.stdin.read()
            try:
                # Try pickle first, then JSON
                if stdin_data.startswith(b'\x80'):  # Pickle magic bytes
                    args_dict = pickle.loads(stdin_data.encode('latin1'))
                else:
                    args_dict = json.loads(stdin_data)
            except:
                # Treat as raw string
                args_dict = {"input_data": stdin_data}
        elif json_input:
            args_dict = json.loads(json_input)
        else:
            args_dict = parse_cli_arguments(kwargs, func_options)
        
        # Execute function
        result = execute_with_context(func_desc, args_dict, context_cls)
        
        # Output handling
        if pickle_output:
            output = pickle.dumps(result)
            sys.stdout.buffer.write(output)
        else:
            output = json.dumps(result, indent=2)
            click.echo(output)
```

## Data Models

### Core Data Types

```python
class ObjectPath(BaseModel):
    """Represents a path for object traversal."""
    path: str
    segments: list[str | int] = Field(default_factory=list)
    
    def __init__(self, path: str, **kwargs):
        super().__init__(path=path, **kwargs)
        self.segments = self._parse_path(path)
    
    def _parse_path(self, path: str) -> list[str | int]:
        """Parse dot/bracket notation into segments."""
        # Implementation for parsing '.field[0].subfield' -> ['field', 0, 'subfield']

class FileOperation(BaseModel):
    """File operation configuration."""
    path: Path
    mode: str = "r"
    encoding: str = "utf-8"
    binary: bool = False
    create_dirs: bool = True

class DataTransformation(BaseModel):
    """Data transformation pipeline step."""
    operation: str  # 'getattr', 'getitem', 'convert', etc.
    parameters: dict[str, Any]
    input_type: type
    output_type: type
```

## State Transitions

```
INPUT DATA (JSON/Pickle/Raw)
  ├─ object.getattr → EXTRACTED VALUE
  │    ├─ data.to_base64 → BASE64 STRING
  │    ├─ data.from_base64 → BINARY DATA
  │    └─ file.write → FILE WRITTEN
  ├─ file.read → FILE CONTENTS
  │    ├─ object.getattr → EXTRACTED VALUE
  │    └─ data.convert → CONVERTED DATA
  └─ data.convert → CONVERTED DATA
       └─ file.write → FILE WRITTEN
```

## Non-Functional Requirements

### Performance
- Object traversal: < 1ms for paths up to 10 levels deep
- File operations: Support files up to 1GB efficiently
- Memory usage: Stream large files rather than loading entirely into memory

### Security
- Path traversal prevention for file operations
- Sandboxed execution environment
- Input validation for all path expressions

### Compatibility
- Backward compatible with existing tool system
- Works with all existing Pydantic models
- Supports both JSON and pickle serialization
- Cross-platform file path handling

### Error Handling
- Graceful handling of missing object paths
- Clear error messages for invalid syntax
- Automatic recovery with default values
- Proper cleanup of temporary files

## Implementation Architecture

### Tool Registration
```python
# New tools register with existing system
@register(tags=["utility", "object"])
def object_getattr(...): ...

@register(tags=["utility", "file"])
def file_write(...): ...

@register(tags=["utility", "file"])
def file_read(...): ...

@register(tags=["utility", "data"])
def data_convert(...): ...
```

### CLI Module Structure
```
src/tidyllm/tools/
├── object_tools.py      # Object manipulation tools
├── file_tools.py        # File I/O tools
├── data_tools.py        # Data transformation tools
└── pipeline.py          # Pipeline composition utilities
```

### Enhanced CLI Adapter
```python
# src/tidyllm/adapters/cli.py augmentations
def detect_input_format(data: bytes) -> str:
    """Detect if input is JSON, pickle, or raw data."""
    
def serialize_output(data: Any, format: str) -> bytes:
    """Serialize output in specified format."""
```

## Testing Scenarios

### Object Manipulation Tests
```python
def test_object_getattr_simple():
    """Test basic attribute access."""
    data = {"name": "test", "value": 42}
    result = object_getattr(".name", json.dumps(data))
    assert result == "test"

def test_object_getattr_nested():
    """Test nested attribute access."""
    data = {"user": {"profile": {"name": "John"}}}
    result = object_getattr(".user.profile.name", json.dumps(data))
    assert result == "John"

def test_object_getattr_array():
    """Test array access."""
    data = {"items": [{"id": 1}, {"id": 2}]}
    result = object_getattr(".items[0].id", json.dumps(data))
    assert result == 1

def test_object_getattr_missing_strict():
    """Test missing field with strict mode."""
    data = {"name": "test"}
    with pytest.raises(KeyError):
        object_getattr(".missing", json.dumps(data), strict=True)

def test_object_getattr_missing_default():
    """Test missing field with default."""
    data = {"name": "test"}
    result = object_getattr(".missing", json.dumps(data), default="N/A", strict=False)
    assert result == "N/A"
```

### File Operation Tests
```python
def test_file_write_text(tmp_path):
    """Test writing text to file."""
    file_path = tmp_path / "test.txt"
    result = file_write(str(file_path), "Hello world")
    assert file_path.read_text() == "Hello world"

def test_file_write_binary(tmp_path):
    """Test writing binary data."""
    file_path = tmp_path / "test.bin"
    data = b"\x89PNG\r\n\x1a\n"
    result = file_write(str(file_path), data, mode="wb")
    assert file_path.read_bytes() == data

def test_file_read_text(tmp_path):
    """Test reading text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello world")
    result = file_read(str(file_path))
    assert result == "Hello world"

def test_file_read_binary_base64(tmp_path):
    """Test reading binary as base64."""
    file_path = tmp_path / "test.bin"
    data = b"\x89PNG\r\n\x1a\n"
    file_path.write_bytes(data)
    result = file_read(str(file_path), binary=True, base64_encode=True)
    assert base64.b64decode(result) == data
```

### Integration Tests
```python
def test_tts_to_file_pipeline():
    """Test complete TTS to file pipeline."""
    # Simulate: uv run tts.py generate_speech | object.getattr '.audio_bytes' | file.write 'out.wav'
    
    # Generate speech
    tts_result = generate_speech("Hello world")
    
    # Extract audio bytes
    audio_b64 = object_getattr(".audio_bytes", tts_result.model_dump_json())
    
    # Convert from base64 to binary
    audio_binary = data_from_base64(audio_b64)
    
    # Write to file
    output_path = file_write("out.wav", audio_binary, mode="wb")
    
    # Verify file exists and has content
    assert Path("out.wav").exists()
    assert Path("out.wav").stat().st_size > 0

def test_pickle_roundtrip():
    """Test pickle serialization roundtrip."""
    original_data = {"audio_bytes": b"\x89PNG", "metadata": {"duration": 1.5}}
    
    # Serialize to pickle
    pickled = pickle.dumps(original_data)
    
    # Deserialize and extract
    result = object_getattr(".audio_bytes", pickle.loads(pickled))
    assert result == b"\x89PNG"
```

### CLI Integration Tests
```bash
# Test scripts for shell integration
#!/bin/bash

# Test basic piping
echo '{"name": "test"}' | uv run object_tools.py getattr '.name'

# Test TTS pipeline
uv run tts.py generate_speech --content="Hello" | uv run object_tools.py getattr '.audio_bytes' | uv run file_tools.py write 'output.wav' --mode=wb --from-base64

# Test pickle mode
uv run tts.py generate_speech --content="Hello" --pickle | uv run object_tools.py getattr '.audio_bytes' | uv run file_tools.py write 'output.wav' --mode=wb

# Test file operations
uv run file_tools.py read 'input.json' | uv run object_tools.py getattr '.config.model' | uv run file_tools.py write 'model_name.txt'
```

## Migration & Rollback Plan

### Phase 1: Core Tool Implementation
1. Implement object manipulation tools in `src/tidyllm/tools/object_tools.py`
2. Implement file operation tools in `src/tidyllm/tools/file_tools.py`
3. Add comprehensive test coverage
4. Update CLI adapter to support pickle mode

### Phase 2: Enhanced CLI Integration
1. Add stdin/stdout support to CLI adapter
2. Implement automatic format detection
3. Add pipeline composition utilities
4. Update existing tools to support new modes

### Phase 3: Documentation and Examples
1. Create usage examples and tutorials
2. Update existing tool documentation
3. Add integration examples with common workflows

### Rollback Strategy
- All changes are additive and backward compatible
- Existing tools continue to work without modification
- New flags are optional and default to current behavior
- Feature flags can disable new functionality if needed

## Open Questions & Decisions

### Resolved Decisions
- [x] **Use existing tool registration system**: Leverage `@register()` decorator for consistency
- [x] **Pickle support**: Add `--pickle` flag to CLI adapter for binary data handling
- [x] **Dot notation parsing**: Implement comprehensive path parsing for nested access
- [x] **Backward compatibility**: Ensure all changes are additive

### Open Questions
- [ ] **Error handling strategy**: Should missing paths return None, raise exceptions, or use configurable behavior?
- [ ] **Performance optimization**: Should we implement caching for repeated object access?
- [ ] **Security considerations**: What sandboxing is needed for file operations?
- [ ] **Stream processing**: Should we support streaming for large files?

## Implementation Checklist

### Core Tools
- [ ] Implement `object_getattr` with dot/bracket notation parsing
- [ ] Implement `object_getitem` for array/dict access
- [ ] Implement `file_write` with binary/text modes
- [ ] Implement `file_read` with encoding detection
- [ ] Implement `data_convert` tools for format conversion

### CLI Enhancements
- [ ] Add `--pickle` flag to CLI adapter
- [ ] Add `--stdin` flag for input handling
- [ ] Implement format detection for input data
- [ ] Add pipeline composition utilities

### Testing
- [ ] Unit tests for all new tools
- [ ] Integration tests for piping workflows
- [ ] Performance tests for large data handling
- [ ] Security tests for path traversal prevention

### Documentation
- [ ] API documentation for new tools
- [ ] Usage examples and tutorials
- [ ] Migration guide for existing users
- [ ] Troubleshooting guide

## Expected Benefits

1. **Composability**: Enable Unix-style piping between tidyllm tools
2. **Flexibility**: Extract specific data fields without manual JSON parsing
3. **Performance**: Efficient binary data handling without JSON overhead
4. **Usability**: Intuitive command-line interface for data manipulation
5. **Extensibility**: Framework for adding more data manipulation tools

This specification provides a comprehensive foundation for implementing file and object manipulation tools that seamlessly integrate with the existing tidyllm architecture while enabling powerful data processing pipelines.