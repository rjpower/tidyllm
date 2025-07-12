# Part Unification Specification

## Executive Summary

This document proposes unifying tidyllm's dual type system (`Source` and `Part`) into a single, comprehensive `Part`-based architecture. The current separation creates confusion, duplication, and inconsistent patterns across the codebase. This unification draws inspiration from Google's genai-processors architecture and provides a path toward a more elegant, extensible design.

## Current State Analysis

### Source Type (`src/tidyllm/types/source.py`)

**Purpose**: Handle file and byte data with automatic conversion
- Converts paths, bytes, data URIs, and file-like objects to `BytesIO`
- Provides utility functions like `read_bytes()` and `read_text()`
- Used primarily in audio processing and PDF handling

**Limitations**:
- No MIME type awareness
- Limited to byte data only
- No extensibility for different content types
- Awkward serialization as data URIs or file paths

### Part Type (`src/tidyllm/types/part.py`)

**Purpose**: Unified content handling with registry system
- MIME type awareness with `mime_type` and `data` fields
- Extensible registry system for custom URL schemes (`file://`, `gdrive://`, etc.)
- Proper JSON schema support
- Type guards for common content types (`ImagePart`, `AudioPart`, etc.)

**Current Usage**:
- Tool parameters that need rich media support
- Custom URL scheme handling (Google Drive integration)
- CLI/API/MCP adapters for content serialization

### Problems with Current Dual System

1. **Conceptual Overlap**: Both handle "content from somewhere" but with different APIs
2. **Inconsistent Usage**: `audio.py` uses `Source` but should handle URLs, streaming, etc.
3. **Missing Functionality**: `audio_from_source` function referenced but doesn't exist
4. **Type Confusion**: Tools must choose between `Source` and `Part` arbitrarily
5. **Limited Extensibility**: `Source` can't be extended for new protocols or content types

## Proposed Unified Architecture

### PartSource Protocol

Define a streaming-first `PartSource` protocol that returns `Enumerable[Part]`:

```python
from typing import Protocol
from tidyllm.types.linq import Enumerable

class PartSource(Protocol):
    """Protocol for sources that can stream Parts."""
    
    def __call__(self, url: Url) -> Enumerable[Part]:
        """Generate a stream of Parts from the given URL."""
        ...

class PartSourceRegistry(dict[str, PartSource]):
    """Registry mapping URL schemes to PartSource implementations."""
    
    def register_part_source(self, scheme: str, source: PartSource):
        """Register a PartSource for a URL scheme."""
        self[scheme] = source
    
    def create_stream(self, url: Url) -> Enumerable[Part]:
        """Create a Part stream from a URL."""
        if url.scheme not in self:
            raise KeyError(f"Unregistered URL scheme: {url.scheme}")
        return self[url.scheme](url)

PART_SOURCE_REGISTRY = PartSourceRegistry()
```

### Enhanced Part Class

Keep the existing `Part` class simple, but add streaming support:

```python
class Part(BaseModel):
    mime_type: str
    data: Base64Bytes = b""
    
    # Existing properties remain unchanged
    @property
    def text(self) -> str:
        return self.data.decode('utf-8')
    
    @property
    def bytes(self) -> bytes:
        return self.data
    
    @classmethod
    def from_url(cls, url: str | Url) -> Enumerable["Part"]:
        """Create Part stream from URL using registry."""
        if isinstance(url, str):
            url = Url(url)
        return PART_SOURCE_REGISTRY.create_stream(url)
    
    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str = "application/octet-stream") -> "Part":
        """Create single Part from bytes."""
        return cls(mime_type=mime_type, data=base64.b64encode(data))
```

### PartSource Implementations

Built-in PartSource implementations for common schemes:

```python
class FilePartSource:
    """Stream Parts from local files."""
    
    def __init__(self, allowed_dirs: list[Path]):
        self.allowed_dirs = allowed_dirs
    
    def __call__(self, url: Url) -> Enumerable[Part]:
        path = self._validate_path(url.path)
        
        # For most files, yield single Part
        if not self._is_streamable(path):
            data = path.read_bytes()
            mime_type = self._detect_mime_type(data, path)
            part = Part.from_bytes(data, mime_type)
            return from_iterable([part])
        
        # For streamable content (audio, video), yield chunks
        return self._stream_file(path)

class MicPartSource:
    """Stream audio Parts from microphone."""
    
    def __call__(self, url: Url) -> Enumerable[Part]:
        # Parse parameters from URL query string
        params = self._parse_audio_params(url)
        return self._stream_microphone(**params)

class HttpPartSource:  
    """Stream Parts from HTTP(S) URLs."""
    
    def __call__(self, url: Url) -> Enumerable[Part]:
        # Handle both single downloads and streaming content
        if self._supports_streaming(url):
            return self._stream_http(url)
        else:
            return self._download_http(url)

# Register built-in sources
PART_SOURCE_REGISTRY.register_part_source("file", FilePartSource([Path(".")]))
PART_SOURCE_REGISTRY.register_part_source("mic", MicPartSource())
PART_SOURCE_REGISTRY.register_part_source("http", HttpPartSource())
PART_SOURCE_REGISTRY.register_part_source("https", HttpPartSource())
```

### Enhanced Type System

```python
# Specific part types with rich type guards
AudioPart: TypeAlias = Part
ImagePart: TypeAlias = Part  
TextPart: TypeAlias = Part
PdfPart: TypeAlias = Part

def is_audio_part(part: Part) -> TypeGuard[AudioPart]:
    return part.mime_type.startswith("audio/")

def is_image_part(part: Part) -> TypeGuard[ImagePart]:
    return part.mime_type.startswith("image/")

def is_streamable_audio_part(part: Part) -> TypeGuard[AudioPart]:
    """Check if audio part supports streaming."""
    return (is_audio_part(part) and 
            part.mime_type in STREAMABLE_AUDIO_FORMATS)
```

### Streaming and Chunked Content

For streaming scenarios (like audio processing), return an Enumerable[Chunk]

## Migration Strategy (No Backwards Compatibility)

### Phase 1: Core Infrastructure

1. **Create `src/tidyllm/types/ext/audio.py`** - Move audio-specific PartSource implementations
2. **Update `src/tidyllm/types/part.py`** - Add PartSource registry and streaming support  
3. **Replace all Source usage** - No backwards compatibility, direct replacement

### Phase 2: Audio System Migration 

1. **Move audio streaming to `ext/audio.py`** - AudioPartSource, MicrophonePartSource
2. **Update `src/tidyllm/tools/audio.py`** - Replace Source with Enumerable[AudioPart]
3. **Fix missing functions** - Implement `audio_from_part` and proper streaming

### Phase 3: Application Updates

1. **Update `apps/transcribe_audio.py`** - Use AudioPart streams instead of Source
2. **Update `apps/pdf_to_note.py`** - Use Part instead of Source
3. **Update transcription tools** - Handle Part objects for audio input

### Phase 4: Complete Removal

1. **Delete `src/tidyllm/types/source.py`** - Complete removal
2. **Update all imports** - Fix all references to source.py
3. **Ensure tests pass** - Run `uv run pytest` until all green

## Detailed Examples

### Audio Processing Migration

**Before** (current broken state):
```python
def audio_stream(input_source: Source) -> Enumerable[AudioChunk]:
    audio_data, sr = librosa.load(input_source)  # Source not imported!
```

**After** (unified):
```python  
def audio_stream(input_part: AudioPart) -> AudioPartStream:
    """Stream audio from any Part source."""
    if input_part.mime_type == "audio/mic":
        return _stream_from_microphone(input_part)
    else:
        return _stream_from_bytes(input_part.bytes)
        
# Usage examples:
audio_stream("file://path/to/audio.wav")
audio_stream("mic://device/0?sample_rate=16000") 
audio_stream("data:audio/wav;base64,...")
```

### PDF Processing Migration

**Before**:
```python
def extract_pdf_images(pdf_source: Source) -> list[ImageData]:
    pdf_bytes = read_bytes(pdf_source)  # Manual byte reading
```

**After**:
```python
def extract_pdf_images(pdf_part: Part) -> list[ImagePart]:
    """Extract pages as image Parts."""
    if not is_pdf_part(pdf_part):
        raise ValueError("Expected PDF content")
    
    pdf_bytes = pdf_part.bytes
    # ... processing ...
    
    return [
        Part.from_bytes(img_bytes, "image/jpeg") 
        for img_bytes in extracted_images
    ]

# Usage:
extract_pdf_images("file://document.pdf")
extract_pdf_images("gdrive://path/to/document") 
extract_pdf_images("https://example.com/doc.pdf")
```

### Transcription Pipeline Enhancement

**Current** (with missing functions):
```python
audio_stream = audio_from_source(audio_source)  # Function doesn't exist!
```

**Proposed**:
```python
def transcribe_with_vad(audio_part: AudioPart) -> TranscriptionResult:
    """Transcribe audio with automatic VAD segmentation."""
    
    # Stream audio with VAD - all Part-based
    audio_chunks = (
        audio_stream(audio_part)
        .with_vad(min_speech_duration=Duration.from_ms(10000))
    )
    
    # Transcribe each chunk
    transcriptions = []
    for chunk in audio_chunks.with_progress("Transcribing"):
        # chunk is an AudioPart with timing metadata
        result = transcribe_audio_part(chunk)
        transcriptions.append(result)
    
    return TranscriptionResult(transcriptions=transcriptions)

# Supports all audio sources uniformly:
transcribe_with_vad("file://meeting.mp3")
transcribe_with_vad("mic://default?duration=60s")
transcribe_with_vad("https://archive.org/audio.mp3")
```

## Benefits of Unification

### 1. Consistency
- Single type system for all content
- Uniform URL-based addressing
- Consistent API patterns across tools

### 2. Extensibility  
- Easy to add new content sources (S3, Azure, etc.)
- Plugin-based architecture via registry
- Rich metadata support

### 3. Developer Experience
- No confusion between Source and Part
- Better IDE support with type guards
- Cleaner tool signatures

### 4. Feature Richness
- Built-in MIME type detection
- Streaming support where appropriate
- Proper serialization for CLI/API usage

### 5. Future-Proofing
- Aligned with modern AI/ML patterns (genai-processors)
- Easy integration with multimodal models
- Composable tool architectures

## Implementation Considerations

### Performance
- Lazy loading for large content
- Streaming for real-time data (audio, video)
- Efficient serialization for CLI/API

### Security
- URL scheme validation
- Sandboxed file access
- Content type verification

### Backward Compatibility
- Migration helpers during transition
- Clear deprecation warnings
- Comprehensive documentation

## Conclusion

This unification represents a significant architectural improvement that will:

1. **Eliminate confusion** between Source and Part types
2. **Enable rich multimedia tooling** with consistent patterns  
3. **Support future extensibility** through registry-based design
4. **Align with industry best practices** from genai-processors
5. **Fix current broken functionality** in audio processing

The migration can be done incrementally over 4 weeks with minimal disruption to existing functionality. The result will be a more elegant, powerful, and maintainable codebase that's ready for the next generation of AI tooling.