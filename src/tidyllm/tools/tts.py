from typing import Any

from pydantic.types import Base64Bytes

"""Text-to-speech tool using litellm."""

import tempfile
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import litellm
from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
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
    return SpeechResult(
        audio_bytes=response.content,
        content=content,
        voice=voice,
        provider=model,
        audio_format="mp3",
    )


@register()
def generate_speech_file(
    content: str,
    output_path: Path,
    voice: str | None = None,
    provider: str = "gemini",
    auto_detect_language: bool = True
) -> Path:
    """Generate TTS audio and save to file.
    
    Args:
        content: Text to convert to speech
        output_path: Path to save audio file
        voice: Voice to use (auto-detected if not provided)
        provider: TTS provider to use
        auto_detect_language: Whether to auto-detect language and voice
        
    Returns:
        Path to saved audio file
        
    Example usage: generate_speech_file("Hello", Path("hello.mp3"))
    """
    # Generate speech
    result = generate_speech(
        content=content,
        voice=voice,
        provider=provider,
        auto_detect_language=auto_detect_language
    )
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    output_path.write_bytes(result.audio_bytes)
    
    print(f"Speech saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    cli_main(generate_speech, context_cls=ToolContext)
