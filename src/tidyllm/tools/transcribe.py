"""Audio transcription tool using Gemini via litellm."""

import base64
from pathlib import Path
from typing import cast

import litellm
import litellm.types
import litellm.types.utils
from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class TranscribedWord(BaseModel):
    """A word with its translation."""
    word_native: str
    word_translated: str | None = None


class TranscriptionResult(BaseModel):
    """Result of audio transcription."""
    transcription: str
    words: list[TranscribedWord] = Field(default_factory=list)


def get_audio_mime_type(file_path: Path) -> str:
    """Get MIME type for audio file."""
    suffix = file_path.suffix.lower()
    mime_types = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".wma": "audio/x-ms-wma",
        ".webm": "audio/webm",
        ".mov": "video/quicktime",
    }
    return mime_types.get(suffix, 'audio/mpeg')


@register()
@cached_function
def transcribe_bytes(
    audio_data: bytes,
    mime_type: str,
    source_language: str | None = None,
    target_language: str = "en",
) -> TranscriptionResult:
    """Transcribe audio from bytes using Gemini Flash via litellm.

    Args:
        audio_data: Audio file data as bytes
        mime_type: MIME type of the audio (e.g., "audio/wav", "audio/mp3")
        language: Language of the audio (auto-detect if not provided)
        translate_to: Target language for translation (default: "en")

    Example usage: transcribe_bytes(audio_bytes, "audio/wav", "es", "en")
    """
    ctx = get_tool_context()

    print(f"Transcribing: {len(audio_data)} bytes of data.")

    # Encode audio data as base64
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # Create structured output schema using Pydantic
    response_schema = TranscriptionResult.model_json_schema()

    # Build prompt
    language_instruction = ""
    if source_language:
        language_instruction = f"The audio is in {source_language}. "
    else:
        language_instruction = "Detect the language of the audio. "

    prompt = f"""Transcribe this audio file. {language_instruction}
For each key word or phrase in the transcription, provide a translation to {target_language}.

Return the results in the following JSON format:
{{
  "transcription": "Hola, ¿cómo estás? Me llamo María y vivo en Madrid.",
  "words": [
    {{
      "word_native": "hola",
      "word_translated": "hello"
    }},
    {{
      "word_native": "cómo estás",
      "word_translated": "how are you"
    }},
    {{
      "word_native": "me llamo",
      "word_translated": "my name is"
    }},
    {{
      "word_native": "Madrid",
      "word_translated": "Madrid"
    }}
  ]
}}

Only include key words/phrases that would benefit from translation, not every single word.
"""

    # Call Gemini via litellm
    response = litellm.completion(
        model=ctx.config.fast_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{audio_base64}",
                    },
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "transcription_result",
                "schema": response_schema,
                "strict": True,
            },
        },
    )

    response = cast(litellm.types.utils.ModelResponse, response)

    # Parse response
    if response.choices:
        choice = cast(litellm.types.utils.Choices, response.choices)[0]
        return TranscriptionResult.model_validate_json(choice.message.content)
    else:
        raise RuntimeError("No response from LLM")


@register()
def transcribe(
    audio_file_path: Path, language: str | None = None, translate_to: str = "en"
) -> TranscriptionResult:
    """Transcribe audio file using Gemini Flash via litellm.

    This is a wrapper around transcribe_bytes that reads the file and determines the MIME type.

    Args:
        audio_file_path: Path to audio file to transcribe
        language: Language of the audio (auto-detect if not provided)
        translate_to: Target language for translation (default: "en")

    Example usage: transcribe(Path("/path/to/audio.mp3"), "es", "en")
    """
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Read audio file and determine MIME type
    audio_data = audio_file_path.read_bytes()
    mime_type = get_audio_mime_type(audio_file_path)

    # Call the cached bytes-based function
    return transcribe_bytes(audio_data, mime_type, language, translate_to)


if __name__ == "__main__":
    cli_main(transcribe, context_cls=ToolContext)
