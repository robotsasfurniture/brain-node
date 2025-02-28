from collections import deque
from typing import List, Optional, Protocol
from .audio_service import AudioChunk
import numpy as np


class AudioBuffer(Protocol):
    """Protocol defining the interface for audio buffers."""

    def add(self, chunk: AudioChunk) -> None:
        """Add an audio chunk to the buffer."""
        ...

    def get_chunks_between(self, start_ms: int, end_ms: int) -> List[AudioChunk]:
        """Retrieve audio chunks between start and end times in milliseconds."""
        ...

    def get_concatenated_audio(
        self, start_ms: int, end_ms: int
    ) -> Optional[np.ndarray]:
        """Get concatenated audio data between times in milliseconds."""
        ...

    def clear(self) -> None:
        """Clear all chunks from the buffer."""
        ...

    def get_duration(self) -> int:
        """Get total duration of audio in buffer in milliseconds."""
        ...

    def __len__(self) -> int:
        """Get number of chunks in buffer."""
        ...


class AudioRingBuffer(AudioBuffer):
    def __init__(self, max_size: int = 1000):
        """Initialize ring buffer with maximum size.

        Args:
            max_size: Maximum number of audio chunks to store (default: 1000 chunks)
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, chunk: AudioChunk) -> None:
        """Add an audio chunk to the buffer."""
        self.buffer.append(chunk)

    def get_chunks_between(self, start_ms: int, end_ms: int) -> List[AudioChunk]:
        """Retrieve audio chunks between start and end times in milliseconds."""
        return [chunk for chunk in self.buffer if start_ms <= chunk.time_ms <= end_ms]

    def get_concatenated_audio(
        self, start_ms: int, end_ms: int
    ) -> Optional[np.ndarray]:
        """Get concatenated audio data between times in milliseconds.

        Returns:
            numpy.ndarray: Concatenated audio data, or None if no chunks found
        """
        chunks = self.get_chunks_between(start_ms, end_ms)
        if not chunks:
            return None

        # Ensure all chunks have the same format
        if not all(
            chunk.sample_rate == chunks[0].sample_rate
            and chunk.channels == chunks[0].channels
            for chunk in chunks
        ):
            raise ValueError("Inconsistent audio format in chunks")

        # Concatenate audio data
        return np.concatenate([chunk.data for chunk in chunks], axis=0)

    def clear(self) -> None:
        """Clear all chunks from the buffer."""
        self.buffer.clear()

    def get_duration(self) -> int:
        """Get total duration of audio in buffer in milliseconds."""
        if not self.buffer:
            return 0

        first_chunk = self.buffer[0]
        last_chunk = self.buffer[-1]
        return last_chunk.time_ms - first_chunk.time_ms

    def __len__(self) -> int:
        """Get number of chunks in buffer."""
        return len(self.buffer)
