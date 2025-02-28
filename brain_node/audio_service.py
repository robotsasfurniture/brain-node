import sounddevice as sd
import numpy as np
import threading
from queue import Queue
from typing import Optional, Callable, Protocol
from dataclasses import dataclass
import soundfile as sf
from datetime import datetime
from .logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__, log_file="logs/audio_service.log")


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""

    data: np.ndarray
    sample_rate: int
    channels: int
    time_ms: int  # Time in milliseconds from start (cumulative)


class AudioProvider(Protocol):
    """Protocol defining the interface for audio providers."""

    sample_rate: int
    channels: int
    interval: float

    def start_sampling(self) -> None:
        """Start sampling audio."""
        ...

    def stop(self) -> None:
        """Stop sampling audio."""
        ...


class AudioSampler(AudioProvider):
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 4,
        interval: float = 0.25,
        on_audio: Optional[Callable[[AudioChunk], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.interval = interval
        self.device: Optional[int] = None
        self.on_audio = on_audio
        self.should_stop = False
        self.current_time_ms = 0  # Keep track of cumulative time
        self.callback_queue = Queue()
        self.callback_thread = None
        self.stream = None
        self.audio_buffer = []
        self.buffer_samples = int(sample_rate * interval)  # samples per interval
        self.all_chunks = []
        self._find_device()

    def _find_device(self):
        """Find the first device that supports 4 channels and print its details."""
        devices = sd.query_devices()
        logger.info("\nAvailable audio devices:")
        logger.info("-" * 80)
        for i, device in enumerate(devices):
            logger.info(f"Device {i}: {device['name']}")
            logger.info(f"  - Max Input Channels: {device['max_input_channels']}")
            logger.info(f"  - Max Output Channels: {device['max_output_channels']}")
            logger.info(f"  - Default Sample Rate: {device['default_samplerate']} Hz")
            if "hostapi" in device:
                hostapi = sd.query_hostapis(device["hostapi"])
                logger.info(f"  - Host API: {hostapi['name']}")
            logger.info("-" * 80)

        for i, device in enumerate(devices):
            if device["max_input_channels"] >= self.channels:
                self.device = i
                logger.info("\nSelected audio device:")
                logger.info("-" * 80)
                logger.info(f"Device ID: {i}")
                logger.info(f"Name: {device['name']}")
                logger.info(f"Input Channels: {device['max_input_channels']}")
                logger.info(f"Default Sample Rate: {device['default_samplerate']} Hz")
                if "hostapi" in device:
                    hostapi = sd.query_hostapis(device["hostapi"])
                    logger.info(f"Host API: {hostapi['name']}")
                logger.info(
                    f"Low Input Latency: {device.get('default_low_input_latency', 'N/A')}"
                )
                logger.info(
                    f"High Input Latency: {device.get('default_high_input_latency', 'N/A')}"
                )
                logger.info("Using Configuration:")
                logger.info(f"  - Sample Rate: {self.sample_rate} Hz")
                logger.info(f"  - Channels: {self.channels}")
                logger.info(f"  - Interval: {self.interval} seconds")
                logger.info("-" * 80)
                return

        raise RuntimeError("No device found with 4 or more channels")

    def _callback_worker(self):
        """Worker thread for processing callbacks."""
        while not self.should_stop:
            try:
                chunk = self.callback_queue.get(timeout=1.0)  # 1 second timeout
                if self.on_audio:
                    self.on_audio(chunk)
            except:
                continue

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio streaming."""
        if status:
            logger.warning(f"Status: {status}")

        # Add incoming data to buffer
        self.audio_buffer.extend(indata.copy())

        # If we have enough samples, create a chunk
        while len(self.audio_buffer) >= self.buffer_samples:
            # Take the first interval's worth of samples
            chunk_data = np.array(self.audio_buffer[: self.buffer_samples])
            self.audio_buffer = self.audio_buffer[self.buffer_samples :]

            # Create audio chunk
            chunk = AudioChunk(
                data=chunk_data.reshape(-1, self.channels),
                sample_rate=self.sample_rate,
                channels=self.channels,
                time_ms=self.current_time_ms,
            )

            # Store chunk for WAV file
            self.all_chunks.append(chunk)

            # Update time and send to callback if needed
            self.current_time_ms += int(self.interval * 1000)
            if self.on_audio:
                self.callback_queue.put(chunk)

    def start_sampling(self):
        """Start continuous sampling from the audio device."""
        if self.device is None:
            raise RuntimeError("No suitable device found")

        logger.info(f"Starting streaming from device {self.device}")

        # Start callback thread if we have a callback
        if self.on_audio:
            self.callback_thread = threading.Thread(target=self._callback_worker)
            self.callback_thread.daemon = True
            self.callback_thread.start()

        try:
            # Start the stream
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                dtype=np.float32,
                blocksize=int(
                    self.sample_rate * 0.01
                ),  # 10ms blocks for smoother processing
            )

            with self.stream:
                logger.info("Stream started")
                while not self.should_stop:
                    sd.sleep(100)  # Sleep and check stop flag every 100ms

        except KeyboardInterrupt:
            logger.info("\nStopping audio streaming...")
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            self.should_stop = True
            if self.callback_thread:
                self.callback_thread.join(timeout=1.0)

    def stop(self):
        """Stop the audio sampling."""
        self.should_stop = True
        # Don't close the stream here, let the context manager handle it


if __name__ == "__main__":
    sampler = AudioSampler()
    sampler.start_sampling()
