"""Audio processing functions for TidyLLM."""

import warnings

import numpy as np

from tidyllm.registry import register
from tidyllm.types.duration import Duration
from tidyllm.types.linq import Enumerable
from tidyllm.types.part import AudioPart

# VAD Configuration
VAD_SAMPLE_RATE = 16000
VAD_BOUNDARY_WINDOW = Duration.from_ms(1000)
MIN_SPEECH_DURATION = Duration.from_ms(500)
MIN_SILENCE_DURATION = Duration.from_ms(250)
SPEECH_THRESHOLD = 0.7
DEFAULT_CHUNK_DURATION = Duration.from_ms(1000)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
warnings.filterwarnings("ignore", module="sunau")
warnings.filterwarnings("ignore", module="audiooop")
warnings.filterwarnings("ignore", module="aifc")

def _load_vad_model():
    """Load the silero VAD model."""
    import torch

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )  # type: ignore
    return model

_VAD_MODEL = None


def find_voice_activity(
    audio_array: np.ndarray,
    sample_rate: int = VAD_SAMPLE_RATE,
    min_speech_duration: Duration = MIN_SPEECH_DURATION,
    min_silence_duration: Duration = MIN_SILENCE_DURATION,
    speech_threshold: float = SPEECH_THRESHOLD,
) -> list[tuple[Duration, Duration]]:
    """Run Voice Activity Detection on audio tensor.

    Args:
        audio_array: Numpy float32 array of audio
        vad_model: Loaded VAD model
        sample_rate: Audio sample rate
        min_speech_duration: Minimum speech duration
        min_silence_duration: Minimum silence duration
        speech_threshold: Speech detection threshold

    Returns:
        List of (start, end) Duration tuples for detected speech segments
    """
    import torch
    from silero_vad import get_speech_timestamps
    global _VAD_MODEL
    if _VAD_MODEL is None:
      _VAD_MODEL = _load_vad_model()

    audio_tensor = torch.from_numpy(audio_array)

    total_duration = min_speech_duration + min_silence_duration
    if Duration.from_samples(len(audio_tensor), sample_rate) < total_duration:
        return []

    timestamps = get_speech_timestamps(
        audio_tensor,
        model=_VAD_MODEL,
        sampling_rate=sample_rate,
        threshold=speech_threshold,
        min_silence_duration_ms=int(min_silence_duration.as_ms()),
        min_speech_duration_ms=int(min_speech_duration.as_ms()),
        return_seconds=True,
    )

    total_duration_audio = Duration.from_samples(len(audio_tensor), sample_rate)

    segments = []
    for ts in timestamps:
        start = Duration.from_sec(ts["start"])
        end = Duration.from_sec(ts["end"])
        # Skip segments that end too close to the audio end
        if end > total_duration_audio - Duration.from_sec(1):
            continue
        segments.append((start, end))

    return segments


@register()
def play_audio(data: Enumerable[AudioPart]):
    """Play audio from AudioPart data using sounddevice.
    
    Args:
        data: Enumerable of AudioPart objects containing audio data
        
    Example usage: play_audio(audio_parts)
    """
    try:
        import tempfile
        from pathlib import Path
        
        import sounddevice as sd
        import soundfile as sf
    except ImportError as e:
        raise ImportError("Audio playback requires sounddevice and soundfile. Install with: uv pip install 'tidyllm[audio]'") from e
    
    for audio_part in data:
        if not audio_part.mime_type.startswith("audio/"):
            continue
        
        # Write audio bytes to temporary file for soundfile to read
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_part.bytes)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Load audio data and play
            audio_data, file_sr = sf.read(str(tmp_path))
            sd.play(audio_data, samplerate=file_sr)
            sd.wait()  # Wait until audio finishes playing
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)
