"""Audio transcription tool using Gemini via litellm."""

import base64
import json
from pathlib import Path
from typing import cast

import litellm
import litellm.types
import litellm.types.utils
from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class TranscribedWord(BaseModel):
    """A word with its translation."""
    word: str
    translation: str | None = None
    start_time: float | None = None
    end_time: float | None = None




class TranscriptionResult(BaseModel):
    """Result of audio transcription."""
    transcription: str
    language: str
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
def transcribe(audio_file_path: Path, language: str | None = None, translate_to: str = "en") -> TranscriptionResult:
    """Transcribe audio using Gemini Flash via litellm.
    
    Args:
        audio_file_path: Path to audio file to transcribe
        language: Language of the audio (auto-detect if not provided)
        translate_to: Target language for translation (default: "en")
    
    Example usage: transcribe(Path("/path/to/audio.mp3"), "es", "en")
    """
    ctx = get_tool_context()
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Read audio file and encode as base64
    audio_data = audio_file_path.read_bytes()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    mime_type = get_audio_mime_type(audio_file_path)

    # Create structured output schema
    response_schema = {
        "type": "object",
        "properties": {
            "transcription": {
                "type": "string",
                "description": "The full transcription of the audio",
            },
            "language": {
                "type": "string",
                "description": "The detected language of the audio (ISO 639-1 code)",
            },
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                    "required": ["word"],
                },
                "description": "Key words from the transcription with translations",
            },
        },
        "required": ["transcription", "language", "words"],
    }

    # Build prompt
    language_instruction = ""
    if language:
        language_instruction = f"The audio is in {language}. "
    else:
        language_instruction = "Detect the language of the audio. "

    prompt = f"""Transcribe this audio file. {language_instruction}
For each key word or phrase in the transcription, provide a translation to {translate_to}.
Return the results in the specified JSON format."""

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
        result_data = json.loads(choice.message.content)

        # Convert words to TranscribedWord objects
        words = []
        for word_data in result_data.get("words", []):
            words.append(
                TranscribedWord(
                    word=word_data["word"], translation=word_data.get("translation")
                )
            )

        return TranscriptionResult(
            transcription=result_data["transcription"],
            language=result_data["language"],
            words=words,
        )
    else:
        raise RuntimeError("No response from LLM")


if __name__ == "__main__":
    cli_main(transcribe, context_cls=ToolContext)
