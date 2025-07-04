"""Text-to-speech tool using litellm."""

import tempfile
from pathlib import Path
from typing import BinaryIO

import litellm
from pydantic import BaseModel, Field

from tidyllm.adapters.cli import cli_main
from tidyllm.cache import cached_function
from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.tools.context import ToolContext


class SpeechResult(BaseModel):
    """Result of TTS generation."""
    audio_bytes: bytes
    content: str
    voice: str
    provider: str
    audio_format: str = "mp3"


def detect_language_and_voice(content: str) -> tuple[str, str]:
    """Detect language and select appropriate voice using LLM."""
    ctx = get_tool_context()
    
    response = litellm.completion(
        model=ctx.config.fast_model,
        messages=[
            {
                "role": "user",
                "content": f"""Detect the language of this text and suggest an appropriate voice code: "{content}"

Return only a JSON object with this format:
{{
    "language": "en",
    "voice": "en-US-Neural2-C"
}}

Common voice codes:
- English: en-US-Neural2-C, en-US-Neural2-D
- Japanese: ja-JP-Neural2-C, ja-JP-Neural2-D
- Spanish: es-ES-Neural2-C, es-ES-Neural2-D
- French: fr-FR-Neural2-C, fr-FR-Neural2-D
- German: de-DE-Neural2-C, de-DE-Neural2-D"""
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "language_detection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string"},
                        "voice": {"type": "string"}
                    },
                    "required": ["language", "voice"]
                },
                "strict": True
            }
        }
    )
    
    if response.choices:
        import json
        result = json.loads(response.choices[0].message.content)
        return result["language"], result["voice"]
    else:
        return "en", "en-US-Neural2-C"


@register()
@cached_function
def generate_speech(
    content: str,
    voice: str | None = None,
    provider: str = "gemini",
    auto_detect_language: bool = True
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
    # Auto-detect language and voice if not provided
    if voice is None and auto_detect_language:
        _, voice = detect_language_and_voice(content)
    elif voice is None:
        voice = "en-US-Neural2-C"
    
    # Determine model based on provider
    if provider == "gemini":
        model = "gemini/gemini-2.5-flash-preview-tts"
    else:
        model = f"{provider}-tts"
    
    print(f"Generating speech: {len(content)} characters with voice {voice}")
    
    # Generate speech using litellm
    response = litellm.speech(
        model=model,
        input=content,
        voice=voice
    )
    
    # Get audio bytes
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)
        response.stream_to_file(temp_path)
        audio_bytes = temp_path.read_bytes()
    
    return SpeechResult(
        audio_bytes=audio_bytes,
        content=content,
        voice=voice,
        provider=provider,
        audio_format="mp3"
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