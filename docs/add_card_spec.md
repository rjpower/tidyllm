Specification: add_card Application for tidyllm

Overview

Create an "add_card" application that takes English or Japanese terms, generates
example sentences, produces TTS audio, and creates Anki flashcards. This will
integrate with the existing tidyllm framework while adding new language
processing capabilities.

New Tools to Add

1. src/tidyllm/tools/tts.py

See @audio.py for examples.

name: "generate_speech"

- generate_speech: Generate TTS audio for text in various languages
  - Use litellm for API abstraction (supports Google TTS, OpenAI TTS, etc.)
  - Cache results using @cached_function
  - Support language detection and voice selection
  - Return MP3 audio bytes with metadata

Example @cli:

uv run ...tts.py generate_speech --content='I hope when you come the weather will be clement' --voice=en-US-Neural2-C --provider=google
uv run ...tts.py generate_speech --content='穏やかな天気になることを願っています' --voice=ja-JP-Neural2-C --provider=google


2. src/tidyllm/tools/language.py

- detect_language(): Auto-detect language of input text

3. Enhanced src/tidyllm/tools/anki.py

class AddVocabCardRequest(BaseModel):
  term_en: str
  term_ja: str
  sentence_en: str
  sentence_ja: str
  audio_en: Path
  audio_ja: Path

  deck_name: str

- anki_add_vocab_card(req: AddVocabCardRequest):
  - creates a single card apkg with the appropriate deck
  - calls anki_import
- anki_create(): extend to handle front and back audio & content as well as template str
- anki_import(): Import .apkg files into Anki collection

Structural Changes

Application Structure

apps/add_card.py

Functions:
- add_single_card(): Add a single term with auto-generation
- add_from_csv(): Batch import from CSV
- add_from_text(): Extract and add vocabulary from text
- review_and_add(): Interactive review before adding

Testing Strategy

Unit Tests

1. test_tts.py: Mock TTS API calls, verify caching
2. test_language.py: Test example generation, language detection

Integration Tests

1. Test full pipeline: term → examples → TTS → Anki card
2. Test batch processing and error handling
3. Test language detection and switching

Test Fixtures

- Sample audio files for different languages
- Mock LLM responses for example generation
- Test Anki database structure

Implementation Details

TTS Integration

- Use litellm for provider abstraction
- Support fallback providers if primary fails
- Implement rate limiting and retry logic
- Cache audio by content hash

Example Generation

- Use structured prompts for consistent output
- Support difficulty levels (beginner/intermediate/advanced)
- Include grammar explanations when relevant
- Generate multiple examples per term

Anki Integration

- Extend existing genanki usage
- Support custom note types with audio fields
- Handle media file management
- Support deck hierarchies

Error Handling

- Graceful degradation if TTS fails
- Fallback to no examples if generation fails
- Clear error messages for missing dependencies
- Validation of language codes and audio formats

Performance Considerations

- Parallel TTS generation for batch operations
- Lazy loading of language models
- Efficient caching strategy
- Progress indicators for long operations

Dependencies to Add

# In pyproject.toml
google-cloud-texttospeech = "^2.14.0"  # For TTS
langdetect = "^1.0.9"  # For language detection

CLI Interface

# Single term
tidyllm add_card "hello" --target-lang ja

# From file
tidyllm add_card --csv vocabulary.csv

tidyllm add_card --word="考える" --source-lang ja --deck "Japanese::N5"
