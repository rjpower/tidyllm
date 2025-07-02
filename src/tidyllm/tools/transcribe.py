"""Audio transcription tool using Gemini via litellm."""

import base64
import json
from pathlib import Path

import litellm
from pydantic import BaseModel, Field

from tidyllm.cli import cli_main
from tidyllm.context import get_tool_context
from tidyllm.registry import register


class TranscribedWord(BaseModel):
    """A word with its translation."""
    word: str
    translation: str | None = None
    start_time: float | None = None
    end_time: float | None = None


class TranscribeArgs(BaseModel):
    """Arguments for audio transcription."""
    audio_file_path: Path = Field(description="Path to audio file to transcribe")
    language: str | None = Field(None, description="Language of the audio (auto-detect if not provided)")
    translate_to: str = Field("en", description="Target language for translation")


class TranscriptionResult(BaseModel):
    """Result of audio transcription."""
    transcription: str
    language: str
    words: list[TranscribedWord] = Field(default_factory=list)
    error: str | None = None




def get_audio_mime_type(file_path: Path) -> str:
    """Get MIME type for audio file."""
    suffix = file_path.suffix.lower()
    mime_types = {
        '.mp3': 'audio/mp3',
        '.wav': 'audio/wav',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.aac': 'audio/aac',
        '.wma': 'audio/x-ms-wma',
        '.webm': 'audio/webm',
    }
    return mime_types.get(suffix, 'audio/mpeg')


@register()
def transcribe(args: TranscribeArgs) -> TranscriptionResult:
    """Transcribe audio using Gemini Flash via litellm.
    
    Example usage: transcribe({"audio_file_path": "/path/to/audio.mp3", "language": "es", "translate_to": "en"})
    """
    ctx = get_tool_context()
    if not args.audio_file_path.exists():
        return TranscriptionResult(
            transcription="",
            language="unknown",
            error=f"Audio file not found: {args.audio_file_path}"
        )
    
    try:
        # Read audio file and encode as base64
        audio_data = args.audio_file_path.read_bytes()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        mime_type = get_audio_mime_type(args.audio_file_path)
        
        # Create structured output schema
        response_schema = {
            "type": "object",
            "properties": {
                "transcription": {
                    "type": "string",
                    "description": "The full transcription of the audio"
                },
                "language": {
                    "type": "string",
                    "description": "The detected language of the audio (ISO 639-1 code)"
                },
                "words": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string"},
                            "translation": {"type": "string"}
                        },
                        "required": ["word"]
                    },
                    "description": "Key words from the transcription with translations"
                }
            },
            "required": ["transcription", "language", "words"]
        }
        
        # Build prompt
        language_instruction = ""
        if args.language:
            language_instruction = f"The audio is in {args.language}. "
        else:
            language_instruction = "Detect the language of the audio. "
            
        prompt = f"""Transcribe this audio file. {language_instruction}
For each key word or phrase in the transcription, provide a translation to {args.translate_to}.
Return the results in the specified JSON format."""
        
        # Call Gemini via litellm
        response = litellm.completion(
            model=ctx.config.fast_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:{mime_type};base64,{audio_base64}"}
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "transcription_result",
                    "schema": response_schema,
                    "strict": True
                }
            }
        )
        
        # Parse response
        if response.choices and response.choices[0].message.content:
            result_data = json.loads(response.choices[0].message.content)
            
            # Convert words to TranscribedWord objects
            words = []
            for word_data in result_data.get("words", []):
                words.append(TranscribedWord(
                    word=word_data["word"],
                    translation=word_data.get("translation")
                ))
                
            return TranscriptionResult(
                transcription=result_data["transcription"],
                language=result_data["language"],
                words=words
            )
        else:
            return TranscriptionResult(
                transcription="",
                language="unknown",
                error="No response from LLM"
            )
            
    except Exception as e:
        return TranscriptionResult(
            transcription="",
            language="unknown",
            error=f"Transcription failed: {str(e)}"
        )


if __name__ == "__main__":
    cli_main(transcribe)