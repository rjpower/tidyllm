"""Audio processing functions for TidyLLM."""

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.stream import Stream, create_stream_from_iterator

# VAD Configuration
VAD_SAMPLE_RATE = 16000
VAD_BOUNDARY_WINDOW_MS = 1000
MIN_SPEECH_DURATION_MS = 1000
MIN_SILENCE_DURATION_MS = 250
SPEECH_THRESHOLD = 0.5


@dataclass
class AudioFormat:
    """Audio format descriptor with conversion capabilities."""

    sample_rate: int
    channels: int
    dtype: Literal["int16", "float32"] = "int16"

    def samples_to_ms(self, samples: int) -> float:
        """Convert sample count to milliseconds for this format."""
        return samples * 1000.0 / self.sample_rate

    def ms_to_samples(self, ms: float) -> int:
        """Convert milliseconds to sample count for this format."""
        return int(ms * self.sample_rate / 1000)

    def bytes_to_samples(self, data: bytes) -> int:
        """Get sample count from byte data."""
        bytes_per_sample = 2 if self.dtype == "int16" else 4
        return len(data) // (self.channels * bytes_per_sample)

    def duration(self, data: bytes) -> float:
        """Get duration in ms from byte data."""
        return self.samples_to_ms(self.bytes_to_samples(data)) / 1000

    def to_float32_array(self, data: bytes) -> np.ndarray:
        """Convert bytes to normalized float32 array."""
        if self.dtype == "int16":
            array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            array = np.frombuffer(data, dtype=np.float32)

        if self.channels == 2:
            array = array.reshape(-1, 2)
        return array

    def from_float32_array(self, array: np.ndarray) -> bytes:
        """Convert float32 array to bytes in this format."""
        if self.channels == 2 and array.ndim == 2:
            array = array.flatten()

        if self.dtype == "int16":
            return (array * 32767).astype(np.int16).tobytes()
        else:
            return array.astype(np.float32).tobytes()

    def extract_segment(self, data: bytes, start_ms: float, end_ms: float) -> bytes:
        """Extract time segment from audio data."""
        start_samples = self.ms_to_samples(start_ms)
        end_samples = self.ms_to_samples(end_ms)

        array = self.to_float32_array(data)
        if self.channels == 2:
            segment = array[start_samples:end_samples, :]
        else:
            segment = array[start_samples:end_samples]

        return self.from_float32_array(segment)

    def resample_to(self, data: bytes, target_format: "AudioFormat") -> bytes:
        """Resample audio data to target format."""
        if (
            self.sample_rate == target_format.sample_rate
            and self.channels == target_format.channels
        ):
            # No resampling needed, just convert data type if necessary
            if self.dtype == target_format.dtype:
                return data
            else:
                array = self.to_float32_array(data)
                return target_format.from_float32_array(array)

        import librosa

        # Convert to float32 array
        array = self.to_float32_array(data)

        # Handle stereo to mono or vice versa
        if self.channels == 2 and target_format.channels == 1:
            # Convert stereo to mono
            array = np.mean(array, axis=1)
        elif self.channels == 1 and target_format.channels == 2:
            # Convert mono to stereo (duplicate channel)
            array = np.column_stack([array, array])

        # Resample if needed
        if self.sample_rate != target_format.sample_rate:
            if array.ndim == 2:
                # Stereo - resample each channel
                resampled = np.zeros(
                    (int(len(array) * target_format.sample_rate / self.sample_rate), 2)
                )
                for ch in range(2):
                    resampled[:, ch] = librosa.resample(
                        array[:, ch],
                        orig_sr=self.sample_rate,
                        target_sr=target_format.sample_rate,
                    )
                array = resampled
            else:
                # Mono
                array = librosa.resample(
                    array, orig_sr=self.sample_rate, target_sr=target_format.sample_rate
                )

        return target_format.from_float32_array(array)


def _load_vad_model():
    """Load the silero VAD model."""
    import torch
    
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    ) # type: ignore
    return model, utils


def find_voice_activity(
    audio_tensor: torch.Tensor,
    vad_model,
    sample_rate: int = VAD_SAMPLE_RATE,
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
    speech_threshold: float = SPEECH_THRESHOLD,
) -> list[tuple[int, int]]:
    """Run Voice Activity Detection on audio tensor.

    Args:
        audio_tensor: Audio data as torch.Tensor
        vad_model: Loaded VAD model
        sample_rate: Audio sample rate
        min_speech_duration_ms: Minimum speech duration in ms
        min_silence_duration_ms: Minimum silence duration in ms
        speech_threshold: Speech detection threshold

    Returns:
        List of (start_ms, end_ms) tuples for detected speech segments
    """
    from silero_vad import get_speech_timestamps
    
    total_duration_ms = min_speech_duration_ms + min_silence_duration_ms
    if (len(audio_tensor) / sample_rate) < (total_duration_ms / 1000):
        return []

    timestamps = get_speech_timestamps(
        audio_tensor,
        model=vad_model,
        sampling_rate=sample_rate,
        threshold=speech_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        min_speech_duration_ms=min_speech_duration_ms,
        return_seconds=True,
    )

    total_ms = len(audio_tensor) / sample_rate * 1000

    segments = []
    for ts in timestamps:
        start_ms = int(ts["start"] * 1000)
        end_ms = int(ts["end"] * 1000)
        # Skip segments that end too close to the audio end
        if end_ms > total_ms - 1000:
            continue
        segments.append((start_ms, end_ms))

    return segments


class AudioChunk(BaseModel):
    """Represents a chunk of audio data."""

    data: bytes
    timestamp: float
    format: AudioFormat

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        return self.format.duration(self.data)

    @property
    def sample_rate(self) -> int:
        return self.format.sample_rate

    @property
    def channels(self) -> int:
        return self.format.channels

    def to_float32_array(self) -> np.ndarray:
        """Convert to normalized float32 array."""
        return self.format.to_float32_array(self.data)

    def extract_segment(self, start_ms: float, end_ms: float) -> "AudioChunk":
        """Extract time segment from this chunk."""
        segment_data = self.format.extract_segment(self.data, start_ms, end_ms)
        return AudioChunk(
            data=segment_data,
            timestamp=self.timestamp + start_ms / 1000,
            format=self.format,
        )

    def resample_to(self, target_format: AudioFormat) -> "AudioChunk":
        """Resample this chunk to target format."""
        resampled_data = self.format.resample_to(self.data, target_format)
        return AudioChunk(
            data=resampled_data, timestamp=self.timestamp, format=target_format
        )

    @classmethod
    def from_float32_array(
        cls, array: np.ndarray, timestamp: float, audio_format: AudioFormat
    ) -> "AudioChunk":
        """Create AudioChunk from float32 numpy array."""
        data = audio_format.from_float32_array(array)
        return cls(data=data, timestamp=timestamp, format=audio_format)


@register(
    name="audio.mic",
    description="Stream audio from microphone",
    tags=["audio", "source", "streaming"],
)
def mic(
    sample_rate: int = 16000,
    channels: int = 1,
    sample_size: float = 1.0,
    device_id: int | None = None,
) -> Stream[AudioChunk]:
    """Stream audio from microphone.

    Args:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        sample_size: Length of audio chunks in seconds
        device_id: Audio device ID

    Returns:
        A stream of audio chunks captured from the microphone
    """
    cleanup_ref = {"stream": None}
    audio_format = AudioFormat(
        sample_rate=sample_rate, channels=channels, dtype="int16"
    )

    def mic_generator():
        """Generator that yields audio chunks from microphone."""
        import sounddevice as sd

        audio_queue = queue.Queue()
        start_time = time.time()
        frames_per_chunk = channels * sample_size * sample_rate

        def audio_callback(indata, _frames, _time_info, status):
            """Callback function for sounddevice."""
            if status:
                print(f"Audio input status: {status}")
            audio_bytes = (indata * 32767).astype(np.int16).tobytes()
            audio_queue.put(audio_bytes)

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=np.float32,
            blocksize=frames_per_chunk,
            callback=audio_callback,
            device=device_id,
        )
        cleanup_ref["stream"] = stream
        stream.start()

        try:
            while True:
                audio_data = audio_queue.get()
                timestamp = time.time() - start_time

                yield AudioChunk(
                    data=audio_data,
                    timestamp=timestamp,
                    format=audio_format,
                )
        except KeyboardInterrupt:
            return
        finally:
            if stream:
                stream.stop()
                stream.close()

    def cleanup():
        """Cleanup function for the stream."""
        if cleanup_ref["stream"]:
            cleanup_ref["stream"].stop()
            cleanup_ref["stream"].close()

    return create_stream_from_iterator(mic_generator, cleanup=cleanup)


@register(
    name="audio.file",
    description="Stream audio from a file",
    tags=["audio", "source", "streaming"],
)
def file(
    file_path: Path,
    sample_rate: int | None = None,
    max_duration: float | None = None,
    seconds_per_chunk: float = 1.0,
) -> Stream[AudioChunk]:
    """Stream audio from a file.

    Args:
        file_path: Path to audio file
        sample_size: Length of each chunk in seconds.
        sample_rate: Override sample rate (uses file's rate if None)
        max_duration: Maximum duration in seconds to read

    Returns:
        A stream of audio chunks read from the file
    """

    def file_generator():
        """Generator that yields audio chunks from file."""
        import librosa

        # Load audio file with librosa, preserving original channel count
        audio_data, file_sample_rate = librosa.load(file_path, sr=sample_rate, mono=False)
        actual_sample_rate = sample_rate or file_sample_rate

        # Normalize audio_data to always be 2D: (channels, samples)
        if audio_data.ndim == 1:
            # Mono file - reshape to (1, samples)
            audio_data = audio_data.reshape(1, -1)
            channels = 1
        else:
            # Stereo file - ensure shape is (channels, samples)
            if audio_data.shape[0] > audio_data.shape[1]:
                audio_data = audio_data.T
            channels = audio_data.shape[0]

        audio_format = AudioFormat(
            sample_rate=actual_sample_rate, channels=channels, dtype="int16"
        )
        total_samples = audio_data.shape[1]
        samples_per_chunk = int(seconds_per_chunk * actual_sample_rate)

        # Apply max_duration limit
        if max_duration:
            max_samples = int(max_duration * actual_sample_rate)
            total_samples = min(total_samples, max_samples)

        samples_read = 0

        while samples_read < total_samples:
            # Read chunk - audio_data is always (channels, samples) now
            chunk_end = min(samples_read + samples_per_chunk, total_samples)
            chunk_data = audio_data[:, samples_read:chunk_end]

            # Convert to interleaved format: (channels, samples) -> (samples * channels,)
            chunk_flat = chunk_data.T.flatten()
            timestamp = samples_read / actual_sample_rate

            # Use AudioFormat to convert
            audio_bytes = audio_format.from_float32_array(chunk_flat)

            yield AudioChunk(
                data=audio_bytes,
                timestamp=timestamp,
                format=audio_format,
            )

            samples_read = chunk_end

    return create_stream_from_iterator(file_generator)


@register(
    name="audio.merge_chunks",
    description="Merge a list of audio chunks into one",
    tags=["audio", "transform"],
)
def merge_chunks(chunks: list[AudioChunk]) -> AudioChunk:
    """Merge multiple audio chunks into a single chunk.

    Args:
        chunks: List of audio chunks to merge

    Returns:
        A single merged audio chunk

    Raises:
        ValueError: If no chunks provided
    """
    if not chunks:
        raise ValueError("No chunks to merge")

    first = chunks[0]
    merged_data = b"".join(chunk.data for chunk in chunks)

    return AudioChunk(
        data=merged_data,
        timestamp=first.timestamp,
        format=first.format,
    )


class VADBuffer:
    """Helper class to manage audio buffering and VAD processing."""

    def __init__(
        self,
        vad_model,
        target_format: AudioFormat | None = None,
        min_buffer_ms: int = MIN_SPEECH_DURATION_MS + VAD_BOUNDARY_WINDOW_MS,
    ):
        self.vad_model = vad_model
        self.target_format = target_format or AudioFormat(
            sample_rate=VAD_SAMPLE_RATE, channels=1, dtype="int16"
        )
        self.min_buffer_ms = min_buffer_ms
        self.audio_data = np.array([], dtype=np.float32)
        self.start_timestamp = None

    def add_audio(self, chunk: AudioChunk):
        """Add audio chunk to buffer."""
        # Resample chunk to target format if needed
        if chunk.format != self.target_format:
            chunk = chunk.resample_to(self.target_format)

        if self.start_timestamp is None:
            self.start_timestamp = chunk.timestamp

        # Convert chunk to float32 and append to buffer
        chunk_array = chunk.to_float32_array()
        if chunk_array.ndim == 2:
            chunk_array = chunk_array.flatten()
        self.audio_data = np.concatenate([self.audio_data, chunk_array])

    def get_buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        return self.target_format.samples_to_ms(len(self.audio_data))

    def process_vad(
        self,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_threshold: float,
    ) -> list[AudioChunk]:
        """Process VAD and return speech segments."""
        if self.get_buffer_duration_ms() < self.min_buffer_ms:
            return []

        if len(self.audio_data) == 0 or self.start_timestamp is None:
            return []

        audio_tensor = torch.from_numpy(self.audio_data)
        segments = find_voice_activity(
            audio_tensor,
            self.vad_model,
            sample_rate=self.target_format.sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_threshold=speech_threshold,
        )

        if not segments:
            # Keep entire buffer when no segments found - this is critical!
            return []

        # Extract speech segments
        speech_chunks = []
        for start_ms, end_ms in segments:
            start_samples = self.target_format.ms_to_samples(start_ms)
            end_samples = self.target_format.ms_to_samples(end_ms)

            segment_audio = self.audio_data[start_samples:end_samples]

            speech_chunks.append(
                AudioChunk.from_float32_array(
                    array=segment_audio,
                    timestamp=self.start_timestamp + start_ms / 1000,
                    audio_format=self.target_format,
                )
            )

        # Keep remaining audio after last segment
        last_end_ms = segments[-1][1]
        cut_samples = self.target_format.ms_to_samples(last_end_ms)

        self.audio_data = self.audio_data[cut_samples:]
        self.start_timestamp = self.start_timestamp + last_end_ms / 1000
        print([s.timestamp for s in speech_chunks])
        return speech_chunks

    def get_remaining_chunk(self) -> AudioChunk | None:
        """Return any remaining audio in buffer as a single chunk."""
        if len(self.audio_data) == 0 or self.start_timestamp is None:
            return None

        remaining_chunk = AudioChunk.from_float32_array(
            array=self.audio_data.copy(),
            timestamp=self.start_timestamp,
            audio_format=self.target_format,
        )

        self._reset_buffer()
        return remaining_chunk

    def _reset_buffer(self):
        """Reset the buffer state."""
        self.audio_data = np.array([], dtype=np.float32)
        self.start_timestamp = None


def chunk_by_vad_stream(
    audio_stream: Stream[AudioChunk],
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
    speech_threshold: float = SPEECH_THRESHOLD,
) -> Stream[AudioChunk]:
    """Split audio stream into chunks based on voice activity detection.

    Args:
        audio_stream: Input stream of audio chunks
        min_speech_duration_ms: Minimum speech duration in ms
        min_silence_duration_ms: Minimum silence duration in ms
        speech_threshold: Speech detection threshold

    Returns:
        Stream of audio chunks segmented by voice activity
    """

    def vad_generator():
        context = get_tool_context()
        vad_model, _ = context.get_ref("vad_model", _load_vad_model)

        vad_buffer = VADBuffer(vad_model)

        for chunk in audio_stream:
            vad_buffer.add_audio(chunk)
            yield from vad_buffer.process_vad(
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_threshold=speech_threshold,
            )

        remaining_chunk = vad_buffer.get_remaining_chunk()
        if remaining_chunk:
            yield remaining_chunk

    return create_stream_from_iterator(vad_generator)


@register(
    name="audio.chunk_by_vad",
    description="Apply VAD to audio file and return speech segments as JSON",
    tags=["audio", "vad", "file"],
)
def chunk_by_vad(
    file_path: Path,
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
    speech_threshold: float = SPEECH_THRESHOLD,
) -> list[AudioChunk]:
    """Apply VAD to audio file and return speech segments.

    Args:
        file_path: Path to audio file
        min_speech_duration_ms: Minimum speech duration in ms
        min_silence_duration_ms: Minimum silence duration in ms
        speech_threshold: Speech detection threshold

    Returns:
        List of speech segments as dictionaries with data, timestamp, etc.
    """
    audio_stream = file(file_path)
    vad_stream = chunk_by_vad_stream(
        audio_stream,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_threshold=speech_threshold,
    )

    chunks = list(vad_stream)
    return chunks


def chunk_to_wav_bytes(chunk: AudioChunk) -> bytes:
    """Convert an AudioChunk to WAV format bytes.

    Args:
        chunk: AudioChunk containing PCM audio data

    Returns:
        WAV file as bytes
    """
    import io

    import soundfile as sf

    # Use AudioFormat to convert to float32 array
    audio_array = chunk.to_float32_array()

    # Write to in-memory WAV file
    with io.BytesIO() as wav_buffer:
        sf.write(
            wav_buffer,
            audio_array,
            chunk.format.sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        return wav_buffer.getvalue()


@register(
    name="audio.chunk_to_wav_bytes",
    description="Convert AudioChunk to WAV bytes",
    tags=["audio", "export", "bytes"],
)
def chunk_to_wav_bytes_tool(chunk: AudioChunk) -> bytes:
    """Convert an AudioChunk to WAV format bytes.

    Args:
        chunk: AudioChunk containing PCM audio data

    Returns:
        WAV file as bytes

    Example: wav_data = chunk_to_wav_bytes_tool(audio_chunk)
    """
    return chunk_to_wav_bytes(chunk)


@register(
    name="audio.chunk_to_wav_file",
    description="Convert AudioChunk to WAV file",
    tags=["audio", "export", "file"],
)
def chunk_to_wav_file(chunk: AudioChunk, output_path: str) -> str:
    """Convert an AudioChunk to a WAV file.

    Args:
        chunk: AudioChunk containing PCM audio data
        output_path: Path where to save the WAV file

    Returns:
        Path to the created WAV file

    Example: chunk_to_wav_file(audio_chunk, "/tmp/segment.wav")
    """
    wav_bytes = chunk_to_wav_bytes(chunk)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(wav_bytes)

    return str(output_path_obj)
