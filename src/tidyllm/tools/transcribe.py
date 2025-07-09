"""Audio transcription tool using Gemini via litellm."""

import base64

import filetype
from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
from tidyllm.llm import completion_with_schema
from tidyllm.registry import register
from tidyllm.source import SourceLike, read_bytes
from tidyllm.tools.context import ToolContext


class TranscribedWord(BaseModel):
    """A word with its translation."""
    word_native: str
    word_translated: str | None = None


class TranscriptionResult(BaseModel):
    """Result of audio transcription."""
    transcription: str
    words: list[TranscribedWord] = Field(default_factory=list)


@register()
@cached_function
def transcribe_audio(
    audio_data: SourceLike,
    source_language: str | None = None,
    target_language: str = "en",
) -> TranscriptionResult:
    """Transcribe audio from bytes using Gemini Flash via litellm.

    Args:
        audio_data: Audio file data as bytes
        language: Language of the audio (auto-detect if not provided)
        translate_to: Target language for translation (default: "en")

    Example usage: transcribe_bytes(audio_bytes, "audio/wav", "es", "en")
    """
    ctx = get_tool_context()

    audio_data = read_bytes(audio_data)
    mime_type = filetype.guess_mime(audio_data)
    print(f"Transcribing: {len(audio_data)} bytes of data.")

    # Encode audio data as base64
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

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

    return completion_with_schema(
        model=ctx.config.fast_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:{mime_type};base64,{audio_base64}",
                        },
                    },
                ],
            }
        ],
        response_schema=TranscriptionResult,
    )


if __name__ == "__main__":
    cli_main(transcribe_audio, context_cls=ToolContext)
