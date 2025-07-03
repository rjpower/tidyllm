"""Audio processing functions for TidyLLM."""

import queue
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from tidyllm.context import get_tool_context
from tidyllm.registry import register
from tidyllm.stream import Stream, create_stream_from_iterator

# VAD Configuration
VAD_SAMPLE_RATE = 16000
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 100
SPEECH_THRESHOLD = 0.5


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
    sample_rate: int
    channels: int
    format: Literal["pcm"] = "pcm"


@register(
    name="audio.mic",
    description="Stream audio from microphone",
    tags=["audio", "source", "streaming"],
)
def mic(
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_size: int = 1024,
    format: Literal["pcm"] = "pcm",
    device_id: int | None = None,
) -> Stream[AudioChunk]:
    """Stream audio from microphone.

    Args:
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        chunk_size: Size of audio chunks in bytes
        format: Audio format
        device_id: Audio device ID

    Returns:
        A stream of audio chunks captured from the microphone
    """
    cleanup_ref = {"stream": None}

    def mic_generator():
        """Generator that yields audio chunks from microphone."""
        import sounddevice as sd

        audio_queue = queue.Queue()
        start_time = time.time()
        frames_per_chunk = chunk_size // (2 * channels)

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
                    sample_rate=sample_rate,
                    channels=channels,
                    format=format,
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
    file_path: str,
    sample_size: int = 1024,
    sample_rate: int | None = None,
    max_duration: float | None = None,
) -> Stream[AudioChunk]:
    """Stream audio from a file.

    Args:
        file_path: Path to audio file
        sample_size: Number of samples to read per chunk
        sample_rate: Override sample rate (uses file's rate if None)
        max_duration: Maximum duration in seconds to read

    Returns:
        A stream of audio chunks read from the file
    """

    def file_generator():
        """Generator that yields audio chunks from file."""
        import librosa
        
        start_time = time.time()
        
        # Load audio file with librosa
        audio_data, file_sample_rate = librosa.load(file_path, sr=sample_rate, mono=False)
        
        # Handle mono/stereo
        if audio_data.ndim == 1:
            channels = 1
            audio_data = audio_data.reshape(1, -1)
        else:
            channels = audio_data.shape[0]
            
        actual_sample_rate = sample_rate or file_sample_rate
        total_samples = audio_data.shape[1]
        
        # Apply max_duration limit
        if max_duration:
            max_samples = int(max_duration * actual_sample_rate)
            total_samples = min(total_samples, max_samples)
            
        samples_read = 0
        
        while samples_read < total_samples:
            # Read chunk
            chunk_end = min(samples_read + sample_size, total_samples)
            chunk_data = audio_data[:, samples_read:chunk_end]
            
            if chunk_data.shape[1] == 0:
                break
                
            # Convert to interleaved format for multi-channel
            if channels == 1:
                chunk_flat = chunk_data[0]
            else:
                chunk_flat = chunk_data.T.flatten()
                
            # Convert to 16-bit PCM bytes
            audio_bytes = (chunk_flat * 32767).astype(np.int16).tobytes()
            timestamp = time.time() - start_time

            yield AudioChunk(
                data=audio_bytes,
                timestamp=timestamp,
                sample_rate=actual_sample_rate,
                channels=channels,
                format="pcm",
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
        sample_rate=first.sample_rate,
        channels=first.channels,
        format=first.format,
    )


@register(
    name="audio.create_resampler",
    description="Create a resampler function for a specific sample rate",
    tags=["audio", "transform", "factory"],
)
def create_resampler(target_sample_rate: int) -> Callable[[AudioChunk], AudioChunk]:
    """Create a resampler function for a specific sample rate.

    Args:
        target_sample_rate: The target sample rate in Hz

    Returns:
        A function that resamples audio chunks to the target rate
    """

    def resample(chunk: AudioChunk) -> AudioChunk:
        """Resample audio chunk to target sample rate."""
        if chunk.sample_rate == target_sample_rate:
            return chunk

        import librosa
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Handle mono/stereo
        if chunk.channels == 2:
            audio_array = audio_array.reshape(-1, 2).T
        
        # Resample using librosa
        resampled = librosa.resample(
            audio_array, 
            orig_sr=chunk.sample_rate, 
            target_sr=target_sample_rate
        )
        
        # Convert back to int16 bytes
        if resampled.ndim == 2:
            resampled = resampled.T.flatten()
        
        resampled_bytes = (resampled * 32767).astype(np.int16).tobytes()

        return AudioChunk(
            data=resampled_bytes,
            timestamp=chunk.timestamp,
            sample_rate=target_sample_rate,
            channels=chunk.channels,
            format=chunk.format,
        )

    return resample


def audio_duration(chunks: list[AudioChunk]) -> float:
    """Calculate total duration of audio chunks in seconds.

    Args:
        chunks: List of audio chunks

    Returns:
        Total duration in seconds
    """
    if not chunks:
        return 0.0

    first = chunks[0]
    bytes_per_second = first.sample_rate * first.channels * 2
    total_bytes = sum(len(chunk.data) for chunk in chunks)

    return total_bytes / bytes_per_second


def has_silence_gap(chunks: list[AudioChunk], max_gap: float = 0.5) -> bool:
    """Check if there's a silence gap in audio chunks.

    Args:
        chunks: List of audio chunks
        max_gap: Maximum allowed gap in seconds

    Returns:
        True if there's a gap larger than max_gap
    """
    if len(chunks) < 2:
        return False

    for i in range(1, len(chunks)):
        gap = chunks[i].timestamp - chunks[i - 1].timestamp
        expected_duration = len(chunks[i - 1].data) / (
            chunks[i - 1].sample_rate * chunks[i - 1].channels * 2
        )
        if gap > expected_duration + max_gap:
            return True

    return False


def chunk_by_vad_stream(
    audio_stream: Stream[AudioChunk],
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
    speech_threshold: float = SPEECH_THRESHOLD,
    buffer_duration_ms: int = 5000,
) -> Stream[AudioChunk]:
    """Split audio stream into chunks based on voice activity detection.

    Args:
        audio_stream: Input stream of audio chunks
        min_speech_duration_ms: Minimum speech duration in ms
        min_silence_duration_ms: Minimum silence duration in ms
        speech_threshold: Speech detection threshold
        buffer_duration_ms: Buffer duration for VAD processing

    Returns:
        Stream of audio chunks segmented by voice activity
    """
    def vad_generator():
        context = get_tool_context()
        vad_model, _ = context.get_ref("vad_model", _load_vad_model)
        
        buffer = []
        buffer_start_time = None
        
        for chunk in audio_stream:
            # Ensure chunk is at VAD sample rate
            if chunk.sample_rate != VAD_SAMPLE_RATE:
                # Simple resampling placeholder - in production use proper resampling
                chunk = AudioChunk(
                    data=chunk.data,
                    timestamp=chunk.timestamp,
                    sample_rate=VAD_SAMPLE_RATE,
                    channels=chunk.channels,
                    format=chunk.format,
                )
            
            if buffer_start_time is None:
                buffer_start_time = chunk.timestamp
            
            buffer.append(chunk)
            
            # Check if buffer has enough data
            buffer_duration = (chunk.timestamp - buffer_start_time) * 1000
            if buffer_duration >= buffer_duration_ms:
                # Process buffer through VAD
                merged_chunk = merge_chunks(buffer)
                
                # Convert to float32 numpy array
                audio_array = np.frombuffer(merged_chunk.data, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_array)
                
                # Find speech segments
                segments = find_voice_activity(
                    audio_tensor,
                    vad_model,
                    sample_rate=VAD_SAMPLE_RATE,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_threshold=speech_threshold,
                )
                
                # Yield segments as AudioChunks
                for start_ms, end_ms in segments:
                    start_samples = int(start_ms * VAD_SAMPLE_RATE / 1000)
                    end_samples = int(end_ms * VAD_SAMPLE_RATE / 1000)
                    
                    segment_audio = audio_array[start_samples:end_samples]
                    segment_bytes = (segment_audio * 32767).astype(np.int16).tobytes()
                    
                    yield AudioChunk(
                        data=segment_bytes,
                        timestamp=buffer_start_time + start_ms / 1000,
                        sample_rate=VAD_SAMPLE_RATE,
                        channels=merged_chunk.channels,
                        format="pcm",
                    )
                
                # Reset buffer
                buffer = []
                buffer_start_time = None
        
        # Handle remaining buffer
        if buffer:
            merged_chunk = merge_chunks(buffer)
            audio_array = np.frombuffer(merged_chunk.data, dtype=np.int16).astype(np.float32) / 32767.0
            audio_tensor = torch.from_numpy(audio_array)
            
            segments = find_voice_activity(
                audio_tensor,
                vad_model,
                sample_rate=VAD_SAMPLE_RATE,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_threshold=speech_threshold,
            )
            
            for start_ms, end_ms in segments:
                start_samples = int(start_ms * VAD_SAMPLE_RATE / 1000)
                end_samples = int(end_ms * VAD_SAMPLE_RATE / 1000)
                
                segment_audio = audio_array[start_samples:end_samples]
                segment_bytes = (segment_audio * 32767).astype(np.int16).tobytes()
                
                yield AudioChunk(
                    data=segment_bytes,
                    timestamp=buffer_start_time + start_ms / 1000,
                    sample_rate=VAD_SAMPLE_RATE,
                    channels=merged_chunk.channels,
                    format="pcm",
                )

    return create_stream_from_iterator(vad_generator)


@register(
    name="audio.chunk_by_vad",
    description="Apply VAD to audio file and return speech segments as JSON",
    tags=["audio", "vad", "file"],
)
def chunk_by_vad(
    file_path: str,
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    min_silence_duration_ms: int = MIN_SILENCE_DURATION_MS,
    speech_threshold: float = SPEECH_THRESHOLD,
    buffer_duration_ms: int = 5000,
    max_chunks: int = 10,
) -> list[dict]:
    """Apply VAD to audio file and return speech segments.

    Args:
        file_path: Path to audio file
        min_speech_duration_ms: Minimum speech duration in ms
        min_silence_duration_ms: Minimum silence duration in ms  
        speech_threshold: Speech detection threshold
        buffer_duration_ms: Buffer duration for VAD processing
        max_chunks: Maximum number of chunks to return

    Returns:
        List of speech segments as dictionaries with data, timestamp, etc.
    """
    audio_stream = file(file_path, sample_size=1024)
    vad_stream = chunk_by_vad_stream(
        audio_stream,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_threshold=speech_threshold,
        buffer_duration_ms=buffer_duration_ms,
    )
    
    chunks = list(vad_stream.take(max_chunks))
    
    return [
        {
            "data_hex": chunk.data.hex(),
            "data_length": len(chunk.data),
            "timestamp": chunk.timestamp,
            "sample_rate": chunk.sample_rate,
            "channels": chunk.channels,
            "format": chunk.format,
        }
        for chunk in chunks
    ]
