from typing import Any

from pydantic.types import Base64Bytes

"""Text-to-speech tool using litellm."""

import base64
from enum import Enum
from pathlib import Path

import litellm
from pydantic import BaseModel

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class Voice(Enum):
    ZEPHYR = "Zephyr"
    PUCK = "Puck"
    CHARON = "Charon"
    KORE = "Kore"
    FENRIR = "Fenrir"
    LEDA = "Leda"
    ORUS = "Orus"
    AOEDE = "Aoede"
    CALLIRRHOE = "Callirrhoe"
    AUTONOE = "Autonoe"
    ENCELADUS = "Enceladus"
    IAPETUS = "Iapetus"
    UMBRIEL = "Umbriel"
    ALGIEBA = "Algieba"
    DESPINA = "Despina"
    ERINOME = "Erinome"
    ALGENIB = "Algenib"
    RASALGETHI = "Rasalgethi"
    LAOMEDEIA = "Laomedeia"
    ACHERNAR = "Achernar"
    ALNILAM = "Alnilam"
    SCHEDAR = "Schedar"
    GACRUX = "Gacrux"
    PULCHERRIMA = "Pulcherrima"
    ACHIRD = "Achird"
    ZUBENELGENUBI = "Zubenelgenubi"
    VINDEMIATRIX = "Vindemiatrix"
    SADACHBIA = "Sadachbia"
    SADALTAGER = "Sadaltager"
    SULAFAT = "Sulafat"


class SpeechResult(BaseModel):
    """Result of TTS generation."""
    audio_bytes: Base64Bytes
    content: str
    voice: Voice
    provider: str
    audio_format: str = "mp3"


@register()
@cached_function
def generate_speech(
    content: str,
    voice: Voice = Voice.ZEPHYR,
    language: str = "",
    model: str = "gemini/gemini-2.5-flash-preview-tts",
) -> SpeechResult:
    """Generate TTS audio for text using litellm.

    Args:
        content: Text to convert to speech
        voice: Voice to use (auto-detected if not provided)
        provider: TTS provider to use
        auto_detect_language: Whether to auto-detect language and voice

    Returns:
        SpeechResult with audio bytes and metadata

    Example usage: generate_speech("Hello world", "en-US-Neural2-C")
    """
    print(f"Generating speech: {len(content)} characters with voice {voice}")

    if language != "":
        content = f"Say the following in {language}: '{content}'"

    # Generate speech using litellm
    response: Any = litellm.speech(model=model, input=content, voice=voice)

    # Stream to BytesIO instead of file
    import io

    audio_buffer = io.BytesIO()
    for chunk in response.iter_bytes():
        audio_buffer.write(chunk)
    audio_bytes = audio_buffer.getvalue()
    result = SpeechResult(
        audio_bytes=base64.b64encode(audio_bytes),
        content=content,
        voice=voice,
        provider=model,
        audio_format="mp3",
    )

    print(len(result.audio_bytes))
    return result

if __name__ == "__main__":
    cli_main(generate_speech, context_cls=ToolContext)
