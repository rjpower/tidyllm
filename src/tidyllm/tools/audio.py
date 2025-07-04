"""Audio processing functions for TidyLLM."""

import io
import queue
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from tidyllm.context import get_tool_context
from tidyllm.duration import Duration
from tidyllm.registry import register
from tidyllm.stream import Stream, create_stream_from_iterator

# VAD Configuration
VAD_SAMPLE_RATE = 16000
VAD_BOUNDARY_WINDOW = Duration.from_ms(1000)
MIN_SPEECH_DURATION = Duration.from_ms(1000)
MIN_SILENCE_DURATION = Duration.from_ms(250)
SPEECH_THRESHOLD = 0.5
DEFAULT_CHUNK_DURATION = Duration.from_ms(1000)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
warnings.filterwarnings("ignore", module="sunau")
warnings.filterwarnings("ignore", module="audiooop")
warnings.filterwarnings("ignore", module="aifc")


@dataclass
class AudioFormat:
    """Audio format descriptor for normalized float32 audio arrays."""

    sample_rate: int
    channels: int

    def samples_to_duration(self, samples: int) -> Duration:
        """Convert sample count to duration for this format."""
        return Duration.from_samples(samples, self.sample_rate)

    def duration_to_samples(self, duration: Duration) -> int:
        """Convert duration to sample count for this format."""
        return duration.as_samples(self.sample_rate)

    def get_sample_count(self, data: np.ndarray) -> int:
        """Get sample count from audio array."""
        # For multi-channel: shape is (samples,) for mono, (samples, channels) for stereo
        return len(data)

    def duration(self, data: np.ndarray) -> Duration:
        """Get duration from audio array."""
        return self.samples_to_duration(self.get_sample_count(data))

    def extract_segment(
        self, data: np.ndarray, start: Duration, end: Duration
    ) -> np.ndarray:
        """Extract time segment from audio array."""
        start_samples = self.duration_to_samples(start)
        end_samples = self.duration_to_samples(end)
        return data[start_samples:end_samples]

    def resample_to(self, data: np.ndarray, target_format: "AudioFormat") -> np.ndarray:
        """Resample audio array to target format."""
        array = data

        # No resampling needed
        if (
            self.sample_rate == target_format.sample_rate
            and self.channels == target_format.channels
        ):
            return array

        import librosa

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

        return array


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
    audio_array: np.array,
    vad_model,
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
    audio_tensor = torch.from_numpy(audio_array)

    total_duration = min_speech_duration + min_silence_duration
    if Duration.from_samples(len(audio_tensor), sample_rate) < total_duration:
        return []

    timestamps = get_speech_timestamps(
        audio_tensor,
        model=vad_model,
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


class AudioChunk(BaseModel):
    """Represents a chunk of audio data."""

    data: list[float]  # Normalized float32 audio data as list
    timestamp: Duration
    format: AudioFormat

    @property
    def duration(self) -> Duration:
        """Get duration"""
        return self.format.duration(self.as_array())

    @property
    def sample_rate(self) -> int:
        return self.format.sample_rate

    @property
    def channels(self) -> int:
        return self.format.channels

    def as_array(self) -> np.ndarray:
        """Get data as numpy array."""
        return np.array(self.data, dtype=np.float32)

    def extract_segment(self, start: Duration, end: Duration) -> "AudioChunk":
        """Extract time segment from this chunk."""
        segment_array = self.format.extract_segment(self.as_array(), start, end)
        return AudioChunk(
            data=segment_array.tolist(),
            timestamp=self.timestamp + start,
            format=self.format,
        )

    def resample_to(self, target_format: AudioFormat) -> "AudioChunk":
        """Resample this chunk to target format."""
        resampled_array = self.format.resample_to(self.as_array(), target_format)
        return AudioChunk(
            data=resampled_array.tolist(),
            timestamp=self.timestamp,
            format=target_format,
        )

    @classmethod
    def from_array(
        cls, array: np.ndarray, timestamp: Duration, audio_format: AudioFormat
    ) -> "AudioChunk":
        """Create AudioChunk from numpy array."""
        # Ensure array is float32 and normalized
        if array.dtype != np.float32:
            if array.dtype == np.int16:
                array = array.astype(np.float32) / 32767.0
            else:
                array = array.astype(np.float32)
        return cls(data=array.tolist(), timestamp=timestamp, format=audio_format)


@register(
    name="audio.mic",
    description="Stream audio from microphone",
    tags=["audio", "source", "streaming"],
)
def mic(
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_duration=DEFAULT_CHUNK_DURATION,
    device_id: int | None = None,
) -> Stream[AudioChunk]:
    """Stream audio from microphone.

    Args:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        chunk_duration: Length of audio chunks
        device_id: Audio device ID

    Returns:
        A stream of audio chunks captured from the microphone
    """
    cleanup_ref: dict[str, object] = {"stream": None}
    audio_format = AudioFormat(sample_rate=sample_rate, channels=channels)

    def mic_generator():
        """Generator that yields audio chunks from microphone."""
        import sounddevice as sd

        audio_queue = queue.Queue()
        start_time = time.time()
        frames_per_chunk = int(channels * chunk_duration.as_sec() * sample_rate)

        def audio_callback(indata, _frames, _time_info, status):
            """Callback function for sounddevice."""
            if status:
                print(f"Audio input status: {status}")
            # Store as float32 array directly (already normalized)
            audio_queue.put(indata.copy())

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
                timestamp = Duration.from_sec(time.time() - start_time)

                yield AudioChunk.from_array(
                    array=audio_data,
                    timestamp=timestamp,
                    audio_format=audio_format,
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
    max_duration: Duration | None = None,
    chunk_duration=DEFAULT_CHUNK_DURATION,
) -> Stream[AudioChunk]:
    """Stream audio from a file.

    Args:
        file_path: Path to audio file
        chunk_duration: Length of each chunk.
        sample_rate: Override sample rate (uses file's rate if None)
        max_duration: Maximum duration to read

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
            sample_rate=int(actual_sample_rate), channels=channels
        )
        total_samples = audio_data.shape[1]
        samples_per_chunk = int(chunk_duration.as_sec() * actual_sample_rate)

        # Apply max_duration limit
        if max_duration:
            max_samples = int(max_duration.as_sec() * actual_sample_rate)
            total_samples = min(total_samples, max_samples)

        samples_read = 0

        while samples_read < total_samples:
            # Read chunk - audio_data is always (channels, samples) now
            chunk_end = min(samples_read + samples_per_chunk, total_samples)
            chunk_data = audio_data[:, samples_read:chunk_end]

            # Convert to interleaved format: (channels, samples) -> (samples * channels,)
            if channels == 1:
                chunk_flat = chunk_data[0]  # Take first (and only) channel
            else:
                chunk_flat = chunk_data.T.flatten()

            timestamp = Duration.from_sec(samples_read / actual_sample_rate)

            yield AudioChunk.from_array(
                array=chunk_flat,
                timestamp=timestamp,
                audio_format=audio_format,
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
    merged_arrays = [chunk.as_array() for chunk in chunks]
    merged_data = np.concatenate(merged_arrays)

    return AudioChunk(
        data=merged_data.tolist(),
        timestamp=first.timestamp,
        format=first.format,
    )


class VADBuffer:
    """Helper class to manage audio buffering and VAD processing."""

    def __init__(
        self,
        vad_model,
        target_format: AudioFormat | None = None,
        min_buffer_duration: Duration = MIN_SPEECH_DURATION + VAD_BOUNDARY_WINDOW,
    ):
        self.vad_model = vad_model
        self.target_format = target_format or AudioFormat(
            sample_rate=VAD_SAMPLE_RATE, channels=1
        )
        self.min_buffer_duration = min_buffer_duration
        self.audio_data = np.array([], dtype=np.float32)
        self.start_timestamp: Duration | None = None

    def add_audio(self, chunk: AudioChunk):
        """Add audio chunk to buffer."""
        # Resample chunk to target format if needed
        if chunk.format != self.target_format:
            chunk = chunk.resample_to(self.target_format)

        if self.start_timestamp is None:
            self.start_timestamp = chunk.timestamp

        # Append chunk data to buffer
        chunk_array = chunk.as_array()
        if chunk_array.ndim == 2:
            chunk_array = chunk_array.flatten()
        self.audio_data = np.concatenate([self.audio_data, chunk_array])

    def get_buffer_duration(self) -> Duration:
        """Get current buffer duration."""
        return self.target_format.samples_to_duration(len(self.audio_data))

    def process_vad(
        self,
        min_speech_duration: Duration,
        min_silence_duration: Duration,
        speech_threshold: float,
    ) -> list[AudioChunk]:
        """Process VAD and return speech segments."""
        if self.get_buffer_duration() < self.min_buffer_duration:
            return []

        if len(self.audio_data) == 0 or self.start_timestamp is None:
            return []

        segments = find_voice_activity(
            self.audio_data,
            self.vad_model,
            sample_rate=self.target_format.sample_rate,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            speech_threshold=speech_threshold,
        )

        if not segments:
            # Keep entire buffer when no segments found - this is critical!
            return []

        # Extract speech segments
        speech_chunks = []
        for start, end in segments:
            start_samples = self.target_format.duration_to_samples(start)
            end_samples = self.target_format.duration_to_samples(end)

            segment_audio = self.audio_data[start_samples:end_samples]

            speech_chunks.append(
                AudioChunk.from_array(
                    array=segment_audio,
                    timestamp=self.start_timestamp + start,
                    audio_format=self.target_format,
                )
            )

        # Keep remaining audio after last segment
        last_end = segments[-1][1]
        cut_samples = self.target_format.duration_to_samples(last_end)

        self.audio_data = self.audio_data[cut_samples:]
        self.start_timestamp = self.start_timestamp + last_end
        return speech_chunks

    def get_remaining_chunk(self) -> AudioChunk | None:
        """Return any remaining audio in buffer as a single chunk."""
        if len(self.audio_data) == 0 or self.start_timestamp is None:
            return None

        remaining_chunk = AudioChunk.from_array(
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
    min_speech_duration: Duration = MIN_SPEECH_DURATION,
    min_silence_duration: Duration = MIN_SILENCE_DURATION,
    speech_threshold: float = SPEECH_THRESHOLD,
) -> Stream[AudioChunk]:
    """Split audio stream into chunks based on voice activity detection.

    Args:
        audio_stream: Input stream of audio chunks
        min_speech_duration: Minimum speech duration
        min_silence_duration: Minimum silence duration
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
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
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
    min_speech_duration: Duration = MIN_SPEECH_DURATION,
    min_silence_duration: Duration = MIN_SILENCE_DURATION,
    speech_threshold: float = SPEECH_THRESHOLD,
) -> list[AudioChunk]:
    """Apply VAD to audio file and return speech segments.

    Args:
        file_path: Path to audio file
        min_speech_duration: Minimum speech duration
        min_silence_duration: Minimum silence duration
        speech_threshold: Speech detection threshold

    Returns:
        List of speech segments as dictionaries with data, timestamp, etc.
    """
    audio_stream = file(file_path)
    vad_stream = chunk_by_vad_stream(
        audio_stream,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
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

    # Get audio data as numpy array
    audio_array = chunk.as_array()

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
    name="audio.play",
    description="Play audio from a file or stdin",
    tags=["audio", "playback"],
)
def play(file_path: Path | None = None):
    """Play audio from a file or stdin.

    Args:
        file_path: Path to audio file to play, or None to read from stdin
    """
    import numpy as np
    import sounddevice as sd
    from pydub import AudioSegment

    if file_path is None:
        s = io.BytesIO(sys.stdin.buffer.read())
        audio = AudioSegment.from_file(s)
    else:
        audio = AudioSegment.from_file(file_path)

    # Convert to numpy array
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        audio_data = audio_data.reshape((-1, 2))
    audio_data = audio_data / 32768.0  # Normalize from int16 to float32
    sample_rate = audio.frame_rate

    sd.play(audio_data, sample_rate)
    sd.wait()
