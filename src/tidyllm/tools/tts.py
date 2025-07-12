from enum import Enum
from typing import Any

from tidyllm.registry import register
from tidyllm.types.linq import Enumerable, Table
from tidyllm.types.part import AudioPart


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


@register()
def generate_speech(
    content: str,
    voice: Voice = Voice.ZEPHYR,
    language: str = "",
    model: str = "gemini/gemini-2.5-flash-preview-tts",
) -> Enumerable[AudioPart]:
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
    if language != "":
        content = f"Say the following in {language}: '{content}'"

    # Generate speech using litellm
    import litellm

    response: Any = litellm.speech(model=model, input=content, voice=voice)

    import io

    audio_buffer = io.BytesIO()
    for chunk in response.iter_bytes():
        audio_buffer.write(chunk)
    audio_bytes = audio_buffer.getvalue()
    audio_part = AudioPart.from_audio_bytes(audio_bytes)
    return Table.from_pydantic([audio_part])
