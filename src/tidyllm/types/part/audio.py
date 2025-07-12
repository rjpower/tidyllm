"""Audio-specific Part sources and utilities."""

import base64
import io
import queue
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from pydantic_core import Url

from tidyllm.types.duration import Duration
from tidyllm.types.linq import Enumerable, Table
from tidyllm.types.part.lib import PART_SOURCE_REGISTRY, Part

# Top-level audio format utility functions


def resample_audio(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio data from one sample rate to another."""
    if from_rate == to_rate:
        return data
    return librosa.resample(data, orig_sr=from_rate, target_sr=to_rate, axis=-1)


def convert_to_mono(data: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
    if data.ndim == 1:
        return data  # Already mono
    return np.mean(data, axis=0)


def convert_to_stereo(data: np.ndarray) -> np.ndarray:
    """Convert mono audio to stereo by duplicating the channel."""
    if data.ndim == 2:
        return data  # Already stereo
    return np.stack([data, data], axis=0)


def samples_to_duration(samples: int, sample_rate: int) -> float:
    """Convert number of samples to duration in seconds."""
    return samples / sample_rate


def duration_to_samples(duration: float, sample_rate: int) -> int:
    """Convert duration in seconds to number of samples."""
    return int(duration * sample_rate)


class AudioPart(Part):
    """Audio Part with native numpy array storage and audio processing capabilities."""

    sample_rate: int
    channels: int
    timestamp: float = 0.0

    def model_post_init(self, __context) -> None:
        """Post-initialization to set up audio data."""
        if not hasattr(self, "_audio_data"):
            self._audio_data = np.array([], dtype=np.float32)

    def to_bytes(self, format="wav") -> bytes:
        """Convert to the given file format and return the audio data as bytes."""
        if self._audio_data.ndim == 1:
            audio_for_sf = self._audio_data.reshape(-1, 1)
        else:
            audio_for_sf = self._audio_data.T

        buffer = io.BytesIO()
        sf.write(
            buffer,
            audio_for_sf,
            self.sample_rate,
            format=format.upper(),
        )
        buffer.seek(0)
        return buffer.read()

    def to_base64(self, format="wav") -> str:
        return base64.b64encode(self.to_bytes(format)).decode()

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        sample_rate: int,
        channels: int | None = None,
        timestamp: float = 0.0,
    ) -> "AudioPart":
        """Create AudioPart from numpy array."""
        # Ensure audio_data is in correct format (channels, samples) or (samples,) for mono
        if data.ndim == 1:
            audio_data = data
            detected_channels = 1
        elif data.ndim == 2:
            if data.shape[0] <= 2:  # (channels, samples)
                audio_data = data
                detected_channels = data.shape[0]
            else:  # (samples, channels) - transpose
                audio_data = data.T
                detected_channels = data.shape[1]
        else:
            raise ValueError(f"Audio data must be 1D or 2D array, got {data.ndim}D")

        if channels is not None and channels != detected_channels:
            raise ValueError(
                f"Specified channels {channels} doesn't match data shape {detected_channels}"
            )

        # Create mime_type with metadata
        mime_type = f"audio/wav;rate={sample_rate};channels={detected_channels};timestamp={timestamp}"

        # Create instance with model_construct to bypass validation
        instance = cls.model_construct(
            mime_type=mime_type,
            sample_rate=sample_rate,
            channels=detected_channels,
            timestamp=timestamp,
        )
        instance._audio_data = audio_data

        return instance

    @classmethod
    def from_audio_bytes(
        cls, audio_bytes: bytes, timestamp: float = 0.0
    ) -> "AudioPart":
        """Create AudioPart from audio bytes (any format supported by librosa)."""
        audio_data, sample_rate = librosa.load(
            io.BytesIO(audio_bytes), sr=None, mono=False
        )
        return cls.from_array(audio_data, int(sample_rate), timestamp=timestamp)

    @property
    def samples(self) -> np.ndarray:
        """Access to raw numpy array."""
        return self._audio_data

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self._audio_data.ndim == 1:
            return len(self._audio_data) / self.sample_rate
        else:
            return self._audio_data.shape[1] / self.sample_rate

    def resample_to(self, new_rate: int) -> "AudioPart":
        """Return new AudioPart with different sample rate."""
        resampled_data = resample_audio(self._audio_data, self.sample_rate, new_rate)
        return AudioPart.from_array(
            resampled_data, new_rate, self.channels, self.timestamp
        )

    def extract_segment(self, start_sec: float, end_sec: float) -> "AudioPart":
        """Return AudioPart slice for specified time range."""
        start_sample = duration_to_samples(start_sec, self.sample_rate)
        end_sample = duration_to_samples(end_sec, self.sample_rate)

        if self._audio_data.ndim == 1:
            segment_data = self._audio_data[start_sample:end_sample]
        else:
            segment_data = self._audio_data[:, start_sample:end_sample]

        return AudioPart.from_array(
            segment_data, self.sample_rate, self.channels, self.timestamp + start_sec
        )


class AudioPartSource:
    """PartSource implementation that returns AudioPart instances for audio mime types."""

    def from_dict(self, d: dict) -> Part:
        """Create AudioPart from dictionary representation."""
        mime_type = d["mime_type"]
        data_bytes = base64.b64decode(d["data"])

        # Parse metadata from mime_type
        timestamp = 0.0
        if "timestamp=" in mime_type:
            timestamp_str = mime_type.split("timestamp=")[1].split(";")[0]
            timestamp = float(timestamp_str)

        return AudioPart.from_audio_bytes(data_bytes, timestamp=timestamp)


class AudioFileSource:
    """Stream audio Parts from file sources."""

    def __init__(self, allowed_dirs: list[Path] | None = None):
        self.allowed_dirs = allowed_dirs or [Path(".")]

    def __call__(self, url: Url) -> Enumerable[Part]:
        """Stream audio Parts from file URL."""
        path = Path(url.path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        resolved_path = path.resolve()
        for allowed_dir in self.allowed_dirs:
            if resolved_path.is_relative_to(allowed_dir.resolve()):
                params = self._parse_query_params(url.query)
                sample_rate = params.get("sample_rate")
                chunk_duration = Duration.from_ms(params.get("chunk_duration_ms", 1000))
                return self._stream_audio_file(path, sample_rate, chunk_duration)

        raise ValueError(f"Path {path} not in allowed directories")

    def _parse_query_params(self, query: str | None) -> dict[str, Any]:
        """Parse URL query parameters."""
        params = {}
        if not query:
            return params

        for param in query.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                if key == "sample_rate":
                    params[key] = int(value)
                elif key == "chunk_duration_ms":
                    params[key] = int(value)
                else:
                    params[key] = value
        return params

    def _stream_audio_file(
        self, path: Path, sample_rate: int | None, chunk_duration: Duration
    ) -> Enumerable[Part]:
        """Stream audio file as chunked audio Parts."""

        def file_generator():
            audio_data, file_sr = librosa.load(str(path), sr=sample_rate, mono=False)
            actual_sr = sample_rate or file_sr

            # Ensure 2D array (channels, samples)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.shape[0] > audio_data.shape[1]:
                audio_data = audio_data.T

            channels, total_samples = audio_data.shape
            samples_per_chunk = int(chunk_duration.as_sec() * actual_sr)

            for start_sample in range(0, total_samples, samples_per_chunk):
                end_sample = min(start_sample + samples_per_chunk, total_samples)
                chunk_data = audio_data[:, start_sample:end_sample]

                # Create AudioPart directly from numpy data
                timestamp_sec = start_sample / actual_sr
                audio_part = AudioPart.from_array(
                    data=chunk_data,
                    sample_rate=int(actual_sr),
                    channels=channels,
                    timestamp=timestamp_sec,
                )
                yield audio_part

        return Table.from_rows(file_generator())


class MicrophoneSource:
    """Stream audio Parts from microphone."""

    def __call__(self, url: Url) -> Enumerable[Part]:
        """Stream microphone audio as Parts."""
        params = self._parse_query_params(url.query)
        sample_rate = params.get("sample_rate", 16000)
        channels = params.get("channels", 1)
        chunk_duration_ms = params.get("chunk_duration_ms", 1000)
        device_id = params.get("device_id")
        duration_sec = params.get("duration_sec")

        return self._stream_microphone(
            sample_rate, channels, chunk_duration_ms, device_id, duration_sec
        )

    def _parse_query_params(self, query: str | None) -> dict[str, Any]:
        """Parse microphone URL query parameters."""
        params = {}
        if not query:
            return params

        for param in query.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                if key in [
                    "sample_rate",
                    "channels",
                    "chunk_duration_ms",
                    "device_id",
                    "duration_sec",
                ]:
                    params[key] = int(value) if value.isdigit() else value
                else:
                    params[key] = value
        return params

    def _stream_microphone(
        self,
        sample_rate: int,
        channels: int,
        chunk_duration_ms: int,
        device_id: int | None,
        duration_sec: int | None,
    ) -> Enumerable[Part]:
        """Stream microphone as audio Parts."""

        def mic_generator():
            import sounddevice as sd

            audio_queue = queue.Queue()
            start_time = time.time()
            frames_per_chunk = int(channels * (chunk_duration_ms / 1000) * sample_rate)

            def audio_callback(indata, _frames, _time_info, _status):
                audio_queue.put(indata.copy())

            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=np.float32,
                blocksize=frames_per_chunk,
                callback=audio_callback,
                device=device_id,
            )

            stream.start()

            try:
                while True:
                    # Check duration limit
                    if duration_sec and (time.time() - start_time) >= duration_sec:
                        break

                    try:
                        audio_data = audio_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    timestamp_sec = time.time() - start_time

                    # Create AudioPart directly from numpy data
                    audio_part = AudioPart.from_array(
                        data=audio_data,
                        sample_rate=sample_rate,
                        channels=channels,
                        timestamp=timestamp_sec,
                    )
                    yield audio_part

            except KeyboardInterrupt:
                return
            finally:
                stream.stop()
                stream.close()

        return Table.from_rows(mic_generator())


PART_SOURCE_REGISTRY.register_scheme("audio", AudioFileSource([Path(".")]))
PART_SOURCE_REGISTRY.register_scheme("mic", MicrophoneSource())

audio_part_source = AudioPartSource()
PART_SOURCE_REGISTRY.register_mimetype("audio/wav", audio_part_source)
PART_SOURCE_REGISTRY.register_mimetype("audio/pcm", audio_part_source)
PART_SOURCE_REGISTRY.register_mimetype("audio/mp3", audio_part_source)
PART_SOURCE_REGISTRY.register_mimetype("audio/ogg", audio_part_source)
