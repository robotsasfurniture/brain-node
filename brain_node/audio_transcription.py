import numpy as np
import base64
import asyncio
from typing import Optional, Callable, Dict, Any
from openai import AsyncOpenAI
from scipy.signal import resample_poly
from .audio_service import AudioSampler, AudioChunk, AudioProvider
from .audio_buffer import AudioBuffer, AudioRingBuffer
from .sound_localization import SoundLocalizer, LocalizationResult
from .visualization import LocalizationVisualizer
from .websocket_service import WebSocketProvider, Ros2WebSocketService
import soundfile as sf
from .logging_config import setup_logger
import os
from datetime import datetime

# Set up logger
logger = setup_logger(__name__, log_file="logs/audio_transcription.log")


class ChatGptService:
    # Static target channel for audio processing
    TARGET_CHANNEL = 0

    def __init__(
        self,
        audio_provider: AudioProvider,
        websocket_service: Optional[WebSocketProvider] = None,
        audio_buffer: Optional[AudioBuffer] = None,
        buffer_duration: int = 60,  # 1 minute of audio history
        on_transcription: Optional[Callable[[str], None]] = None,
        localizer: Optional[SoundLocalizer] = None,
        websocket_uri: str = "ws://localhost:9090",
        visualizer: Optional[LocalizationVisualizer] = None,
    ):
        self.audio_provider = audio_provider
        self.audio_queue = asyncio.Queue()
        self.should_stop = False
        self.on_transcription = on_transcription or self._default_transcription_callback
        self.localizer = localizer
        self.client = AsyncOpenAI()
        self.connection = None
        self._loop = None
        self.websocket_service = websocket_service or Ros2WebSocketService(
            uri=websocket_uri
        )
        self.visualizer = visualizer

        # Set up audio buffer
        if audio_buffer is None:
            # Calculate buffer size based on duration and interval
            interval = getattr(audio_provider, "interval", 0.25)
            max_chunks = int(buffer_duration / interval)
            self.audio_buffer = AudioRingBuffer(max_size=max_chunks)
        else:
            self.audio_buffer = audio_buffer

    def _default_transcription_callback(self, text: str):
        """Default callback for transcription results."""
        logger.info(f"Transcription: {text}")

    def _handle_audio_chunk_sync(self, chunk: AudioChunk):
        """Synchronous wrapper for handling audio chunks."""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        # Add to buffer immediately (this is thread-safe)
        self.audio_buffer.add(chunk)

        # Schedule the async operation in the event loop
        asyncio.run_coroutine_threadsafe(self.audio_queue.put(chunk), self._loop)

    async def _handle_audio_chunk(self, chunk: AudioChunk):
        """Handle incoming audio chunks from the sampler."""
        await self.audio_queue.put(chunk)
        self.audio_buffer.add(chunk)  # Add to ring buffer

    async def _handle_move_to_command(self, start_ms: int, end_ms: int):
        """Handle MOVE_TO command from ChatGPT."""
        try:
            if start_ms >= end_ms:
                logger.warning(f"Invalid time range: {start_ms}ms to {end_ms}ms")
                return

            # Get audio data from buffer
            audio_data = self.audio_buffer.get_concatenated_audio(start_ms, end_ms)
            if audio_data is None:
                logger.warning(f"No audio found between {start_ms}ms and {end_ms}ms")
                return

            # Save the audio into a wav file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("audio"):
                os.makedirs("audio")
            wav_file_path = f"audio/audio_{timestamp}_{start_ms}_{end_ms}.wav"
            sf.write(wav_file_path, audio_data, self.audio_provider.sample_rate)

            # Perform localization if localizer is available
            location = None
            if self.localizer is not None:
                try:
                    # Calculate samples for 500ms segments
                    samples_per_segment = int(
                        self.audio_provider.sample_rate * 0.5
                    )  # 0.5s = 500ms
                    total_samples = audio_data.shape[0]
                    num_complete_segments = total_samples // samples_per_segment
                    logger.info(f"Number of complete segments: {num_complete_segments}")

                    # Lists to store results
                    angles = []
                    distances = []
                    x_coords = []
                    y_coords = []

                    # Process each complete segment
                    for i in range(num_complete_segments):
                        start_idx = i * samples_per_segment
                        end_idx = start_idx + samples_per_segment
                        # Keep all channels, just slice the time dimension
                        segment = audio_data[start_idx:end_idx, :]

                        result = self.localizer.localize_from_arrays(segment)
                        if result:
                            angles.append(result.angle)
                            distances.append(result.distance)
                            x_coords.append(result.x)
                            y_coords.append(result.y)

                    logger.info(f"Number of results: {len(angles)}")

                    # Calculate mean results if we have any valid segments
                    if angles:
                        mean_angle = np.mean(angles)
                        mean_distance = np.mean(distances)
                        mean_x = np.mean(x_coords)
                        mean_y = np.mean(y_coords)
                        std_angle = np.std(angles)
                        std_distance = np.std(distances)

                        location_info = (
                            f" Sound source detected at:\n"
                            f"  - Angle: {mean_angle:.1f}° (±{std_angle:.1f}°)\n"
                            f"  - Distance: {mean_distance:.1f}m (±{std_distance:.1f}m)\n"
                            f"  - Position: ({mean_x:.1f}m, {mean_y:.1f}m)\n"
                            f"  - Segments analyzed: {len(angles)}/{num_complete_segments}"
                        )
                        logger.info(location_info)

                        location = LocalizationResult(
                            angle=mean_angle,
                            distance=mean_distance,
                        )

                        # Update visualization if enabled
                        if self.visualizer is not None:
                            self.visualizer.update_plot(location)
                    else:
                        logger.warning("No valid localization results in any segment")

                except Exception as e:
                    logger.error(f"Localization error: {e}")

            if location:
                # Send location to ROS2
                await self.websocket_service.send_location(location)

        except (ValueError, KeyError) as e:
            logger.error(f"Error processing MOVE_TO command: {e}")

    async def process_audio(self):
        """Process audio chunks from the queue and send to OpenAI."""
        logger.info("Starting audio processing loop")
        while not self.should_stop:
            try:
                chunk = await self.audio_queue.get()

                if not self.connection:
                    logger.warning(
                        "OpenAI connection not established, skipping audio chunk"
                    )
                    continue

                # Extract the target channel
                single_channel = chunk.data[:, self.TARGET_CHANNEL]

                # Convert to int16 and resample to 24kHz
                resampled = resample_poly(single_channel, 24000, chunk.sample_rate)
                audio_bytes = (resampled * 32767).astype(np.int16).tobytes()

                try:
                    # Convert audio bytes to base64 string
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                    # Send audio chunk
                    await self.connection.input_audio_buffer.append(audio=audio_base64)
                except Exception as e:
                    logger.error(f"Error sending audio to OpenAI: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

            except asyncio.CancelledError:
                logger.info("Audio processing task cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                import traceback

                logger.error(traceback.format_exc())

        logger.info("Audio processing loop ended")

    async def handle_events(self):
        """Handle events from the OpenAI connection."""
        logger.info("Handling events")
        speech_time_queue = []
        async for event in self.connection:
            if event.type == "error":
                logger.error(f"Error: {event.error}")
            elif event.type == "input_audio_buffer.speech_started":
                logger.info(f"Speech started at {event.audio_start_ms}ms")
                speech_time_queue.append(
                    [event.audio_start_ms, None]
                )  # Start time, stop time initially None
            elif event.type == "input_audio_buffer.speech_stopped":
                logger.info(f"Speech stopped at {event.audio_end_ms}ms")
                # Find the most recent speech segment without a stop time and add it
                for segment in reversed(speech_time_queue):
                    if segment[1] is None:
                        segment[1] = event.audio_end_ms
                        logger.info(
                            f"Speech segment recorded: {segment[0]}ms to {segment[1]}ms"
                        )
                        break
            elif event.type == "response.text.done":
                text = event.text.strip()
                logger.info(f"Text done: {text}")
                if text and text.strip() == "MOVE_TO" and speech_time_queue:
                    # Get the most recent complete speech segment
                    for segment in reversed(speech_time_queue):
                        if segment[1] is not None:  # Found a complete segment
                            start_ms, end_ms = segment
                            start_ms -= 300  # For VAD padding
                            end_ms -= 500  # For VAD padding
                            logger.info(
                                f"Using speech segment: {start_ms}ms to {end_ms}ms"
                            )
                            await self._handle_move_to_command(start_ms, end_ms)
                            break

    async def start(self):
        """Start the audio sampling and transcription service."""
        logger.info(f"Using channel {self.TARGET_CHANNEL} for transcription")

        try:
            # Store the event loop for the audio callback
            self._loop = asyncio.get_running_loop()
            logger.debug("Got event loop")

            # Connect to WebSocket and wait for connection
            logger.info("Connecting to WebSocket...")
            await self.websocket_service.connect()
            logger.info("WebSocket connection established")

            logger.info("Connecting to OpenAI...")
            async with self.client.beta.realtime.connect(
                model="gpt-4o-realtime-preview"
            ) as connection:
                logger.info("Connected to OpenAI")
                self.connection = connection

                # Initialize the session
                logger.info("Initializing session...")
                await connection.session.update(
                    session={
                        "modalities": ["text"],
                        "instructions": """
    The instructor robot will receive audio input to determine movement actions based on command cues. For each command, the robot should perform a corresponding movement action as follows:

- **Audio cues for 'Table Bot come here'** – MOVE_TO
- **Audio cues for 'Table Bot over here'** – MOVE_TO


The robot should only respond using these commands. The robot should analyze audio input continuously and prioritize the most recent command. If ambiguous commands are detected (e.g., unclear or overlapping), the robot should remain in its last known state until a clearer command is received.
    """,
                        "temperature": 0.6,
                        "turn_detection": {
                            "type": "server_vad",
                        },
                    }
                )
                logger.info("Session updated successfully")

                # Start the audio processing task
                logger.info("Creating tasks...")
                audio_task = asyncio.create_task(self.process_audio())
                events_task = asyncio.create_task(self.handle_events())
                logger.info("Tasks created successfully")

                # Set up the audio provider callback with the sync wrapper
                if isinstance(self.audio_provider, AudioSampler):
                    logger.info("Setting up audio provider callback...")
                    self.audio_provider.on_audio = self._handle_audio_chunk_sync
                    logger.info("Audio provider callback set")

                # Start the audio provider in a separate thread to not block the event loop
                logger.info("Starting audio provider...")
                import threading

                provider_thread = threading.Thread(
                    target=self.audio_provider.start_sampling
                )
                provider_thread.start()
                logger.info("Audio provider started in separate thread")

                try:
                    # Wait for tasks to complete or until interrupted
                    logger.info("Waiting for tasks to complete...")
                    await asyncio.gather(audio_task, events_task)
                except asyncio.CancelledError:
                    logger.info("Tasks cancelled")
                    raise
                finally:
                    logger.info("Stopping audio provider...")
                    self.audio_provider.stop()
                    if provider_thread.is_alive():
                        provider_thread.join()
                    logger.info("Audio provider stopped")

        except KeyboardInterrupt:
            logger.info("\nStopping transcription service...")
        except Exception as e:
            logger.error(f"Error during service operation: {e}")
            import traceback

            logger.error(traceback.format_exc())
        finally:
            self.should_stop = True
            self.audio_provider.stop()
            if self.connection:
                await self.connection.close()
            await self.websocket_service.close()
            if self.visualizer:
                self.visualizer.close()
            logger.info("Service stopped")


def run_service(service: ChatGptService):
    """Run the service in the asyncio event loop."""
    asyncio.run(service.start())
