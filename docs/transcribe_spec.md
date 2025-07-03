
# Long-Form Audio Transcription & Vocabulary Extraction System

## Analysis of Existing Tools

### Current Capabilities
The tidyllm codebase already provides strong foundations for transcription:

**Audio Processing (`src/tidyllm/tools/audio.py`):**
- Voice Activity Detection (VAD) with Silero model
- Streaming audio processing with `Stream[AudioChunk]`
- File and microphone input support
- Audio format conversion and resampling
- `chunk_by_vad_stream()` - splits audio into speech segments

**Transcription (`src/tidyllm/tools/transcribe.py`):**
- Gemini Flash transcription via litellm
- Multiple audio format support (MP3, WAV, M4A, etc.)
- Word-level translation capabilities 
- Language detection and translation

**Vocabulary Management (`src/tidyllm/tools/vocab_table.py`):**
- SQLite-based vocabulary storage
- CRUD operations for vocab items
- Search by word, translation, or tags
- JSON-encoded examples and tags

**Caching System (`src/tidyllm/cache.py`):**
- Function-level caching with `@cached_function`
- SQLite persistence with hash-based keys
- Automatic cache table creation

**Stream Framework:**
- Type-safe iterator-based streaming
- Operations: map, filter, batch, window, reduce, collect
- Memory efficient for large audio files
- Context management and cleanup support

### Missing Components

1. **Pipeline Integration** - No coordination between VAD → Transcription
2. **Batch Transcription** - Current transcription only handles complete files
3. **Vocabulary Extraction** - No LLM-based vocab extraction from transcripts
4. **Vocabulary Diffing** - No comparison against existing vocab database
5. **Interactive Review UI** - No interface for vocabulary selection
6. **Progress Monitoring** - No real-time progress feedback
7. **Error Handling** - No retry logic or fallback mechanisms

## Architecture Overview
The system processes long audio recordings through a streaming pipeline:

Audio File → VAD Segmentation → Transcription → Vocabulary Extraction → Diff → Review UI
## Required New Components

### 1. Audio Segmentation Tool (audio_segment.py)
**Status**: ✅ **Already exists** - `chunk_by_vad_stream()` in `audio.py` provides this functionality

**Modification needed**: Update to return structured segments with timing info:

```python
class AudioSegment(BaseModel):
    """A segment of audio with timing information."""
    audio_data: bytes
    start_time: float  # seconds
    end_time: float
    duration: float
    segment_index: int

class AudioSegmentArgs(BaseModel):
    """Arguments for audio segmentation."""
    audio_path: str
    min_speech_duration: float = 0.5  # minimum speech duration in seconds
    speech_pad: float = 0.3  # padding around speech segments
    max_segment_duration: float = 30.0  # max duration to avoid API limits

@register()
def audio_segment(args: AudioSegmentArgs) -> Stream[AudioSegment]:
    """Split audio into speech segments using Silero VAD."""
    # Leverages existing chunk_by_vad_stream() with structured output
```
### 2. Batch Transcription Tool (batch_transcribe.py)
**Status**: ❌ **Needs to be built** - Integrates existing transcription with streaming

```python
class TranscriptionSegment(BaseModel):
    """Transcription result for a segment."""
    segment_index: int
    start_time: float
    end_time: float
    text: str
    words: list[TranscribedWord]  # Reuse from existing transcribe.py
    language: str

class BatchTranscribeArgs(BaseModel):
    """Arguments for batch transcription."""
    audio_segments: Stream[AudioSegment]
    target_language: str
    source_language: str | None = None
    batch_size: int = 5  # Process N segments concurrently
    
@register()
@cached_function  # Cache transcription results
def batch_transcribe(args: BatchTranscribeArgs) -> Stream[TranscriptionSegment]:
    """Transcribe audio segments in batches with rate limiting."""
    # Uses existing transcribe() function with streaming wrapper
```
### 3. Vocabulary Extraction Tool (vocab_extract.py)
**Status**: ❌ **Needs to be built** - New LLM-based vocabulary extraction

```python
class ExtractedVocab(BaseModel):
    """Extracted vocabulary item with context."""
    word: str
    translation: str
    example_source: str  # Example sentence in source language
    example_translated: str  # Example sentence in target language
    context: str  # Surrounding context
    confidence: float  # LLM confidence score
    segment_index: int
    timestamp: float

class VocabExtractArgs(BaseModel):
    """Arguments for vocabulary extraction."""
    transcriptions: Stream[TranscriptionSegment]
    source_language: str
    target_language: str
    extract_phrases: bool = True  # Extract phrases vs just words
    context_window: int = 50  # Characters of context

@register()
@cached_function  # Cache extraction results
def vocab_extract(args: VocabExtractArgs) -> Stream[ExtractedVocab]:
    """Extract vocabulary from transcriptions using LLM."""
    # Uses existing LLM infrastructure to extract meaningful vocabulary
```
## App Requirements

Based on user clarification:

### Input/Output Specifications
- **Audio formats**: All formats supported by transcribe.py (MP3, WAV, M4A, OGG, FLAC, AAC, WMA, WebM, MOV)  
- **Vocabulary diffing**: Simple word text matching against existing vocab database
- **CSV output**: New words that aren't in database yet
- **UI**: Basic vocabulary selection interface - users can select/deselect words to add

### Command Structure
Multiple separate commands for each stage:
1. `segment` - VAD segmentation of audio file  
2. `transcribe` - Transcribe VAD segments with caching
3. `extract` - Extract vocabulary from transcripts (built into transcribe.py already)
4. `diff` - Compare against existing vocab database  
5. `review` - Interactive selection UI
6. `pipeline` - Run all stages together

### Progress Monitoring
Per-segment progress tracking with stream utilities

## Simplified Implementation Plan

Since transcribe.py already extracts vocabulary with translations, the app becomes much simpler:

### 1. Stream Progress Utilities (stream_utils.py)
**Status**: ❌ **Needs to be built** - Add progress tracking to streams

```python
def progress_stream(
    stream: Stream[T], 
    description: str = "Processing",
    total: int | None = None
) -> Stream[T]:
    """Add progress bar to stream processing."""
    # Uses rich.progress for real-time progress updates
```

### 2. Vocabulary Diffing (In app)
**Status**: ❌ **Simple implementation needed**

```python
def find_new_words(transcribed_words: list[TranscribedWord]) -> list[TranscribedWord]:
    """Find words not in existing vocab database."""
    # Use vocab_search() to check existence
    # Return only new words
```

### 3. CSV Export (In app)  
**Status**: ❌ **Simple implementation needed**

```python
def export_new_words_csv(words: list[TranscribedWord], output_path: str):
    """Export new vocabulary to CSV."""
    # word, translation, source_language, target_language columns
```

### 4. Interactive Selection UI (In app)
**Status**: ❌ **Basic implementation needed**

```python  
def select_vocabulary(words: list[TranscribedWord]) -> list[TranscribedWord]:
    """Interactive selection of vocabulary to add."""
    # Rich-based table with checkboxes
    # Return selected words
```
## Final App Structure

**File**: `apps/transcribe_audio.py`

### Commands to implement:
1. `segment-audio` - VAD segmentation with progress
2. `transcribe-segments` - Batch transcription with caching  
3. `diff-vocab` - Find new words vs existing database
4. `review-vocab` - Interactive selection UI
5. `export-csv` - Export new words to CSV
6. `full-pipeline` - Run all stages together

### Key Implementation Notes:
- Use existing `chunk_by_vad_stream()` from audio.py for segmentation
- Use existing `transcribe()` from transcribe.py for transcription (already extracts vocab)
- Use existing `vocab_search()` from vocab_table.py for diffing
- Use existing `vocab_add()` from vocab_table.py to add selected words
- Add progress tracking with rich progress bars
- Cache transcription results using `@cached_function`
- Simple CSV export with word, translation, language columns
- Basic rich-based selection UI with checkboxes

The app becomes a thin orchestration layer over existing tools rather than complex new functionality.




