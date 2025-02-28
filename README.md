# Brain Node

A real-time audio transcription service that captures audio from a 4-channel device and transcribes one channel using OpenAI's Realtime API via WebSocket.

## Features

- Modular design with dependency injection
- Pluggable audio providers and buffer implementations
- Automatically detects and uses 4-channel audio devices
- Samples audio at configurable intervals (default: 0.25 seconds)
- Real-time transcription using OpenAI's GPT-4o Realtime model
- WebSocket-based streaming for instant transcription
- Configurable target channel selection
- Custom transcription callback support
- Audio history buffer with MOVE_TO command support

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd brain-node
```

2. Install using Poetry:

```bash
poetry install
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Command Line Interface

Run the service using Poetry with default settings:

```bash
poetry run brain-node
```

Available command-line options:

```
--sample-rate RATE    Audio sample rate in Hz (default: 44100)
--channels NUM        Number of audio channels (default: 4)
--interval SEC        Sampling interval in seconds (default: 0.25)
--target-channel NUM  Which audio channel to transcribe (default: 0)
--buffer-duration SEC Duration of audio history to maintain in seconds (default: 300)
--api-key KEY        OpenAI API key (can also be set via OPENAI_API_KEY environment variable)
```

Example with custom settings:

```bash
poetry run brain-node --sample-rate 48000 --channels 4 --target-channel 1 --buffer-duration 600
```

### Programmatic Usage

You can use the service programmatically with custom audio providers, buffers, and callbacks:

```python
from brain_node import ChatGptService, AudioSampler, AudioRingBuffer

# Create an audio provider
audio_sampler = AudioSampler(
    sample_rate=44100,
    channels=4,
    interval=0.25
)

# Create a custom buffer (optional)
audio_buffer = AudioRingBuffer(max_size=2000)

# Create a callback for transcriptions
def my_transcription_callback(text: str):
    print(f"Received transcription: {text}")

# Create and start the service
service = ChatGptService(
    audio_provider=audio_sampler,
    audio_buffer=audio_buffer,  # Optional: uses default if not provided
    target_channel=0,
    buffer_duration=300,  # Only used if audio_buffer not provided
    on_transcription=my_transcription_callback
)
service.start()
```

### Custom Components

#### Custom Audio Providers

You can create custom audio providers by implementing the `AudioProvider` protocol:

```python
from brain_node import AudioProvider

class MyCustomAudioProvider(AudioProvider):
    def start_sampling(self) -> None:
        # Implement your audio sampling logic
        pass

    def stop(self) -> None:
        # Implement your cleanup logic
        pass
```

#### Custom Audio Buffers

You can create custom audio buffers by implementing the `AudioBuffer` protocol:

```python
from brain_node import AudioBuffer, AudioChunk
from typing import List, Optional
import numpy as np

class MyCustomBuffer(AudioBuffer):
    def add(self, chunk: AudioChunk) -> None:
        # Implement chunk storage logic
        pass

    def get_chunks_between(self, start_time: float, end_time: float) -> List[AudioChunk]:
        # Implement chunk retrieval logic
        pass

    def get_concatenated_audio(self, start_time: float, end_time: float) -> Optional[np.ndarray]:
        # Implement audio concatenation logic
        pass

    def clear(self) -> None:
        # Implement buffer clearing logic
        pass

    def get_duration(self) -> float:
        # Implement duration calculation
        pass

    def __len__(self) -> int:
        # Implement length calculation
        pass
```

Use your custom components:

```python
service = ChatGptService(
    audio_provider=MyCustomAudioProvider(),
    audio_buffer=MyCustomBuffer(),
    target_channel=0
)
service.start()
```

## Audio History and MOVE_TO Commands

The service maintains a configurable buffer of audio history. When ChatGPT sends a MOVE_TO command, the service can retrieve and replay specific segments of audio from this history.

Example MOVE_TO command response:

```json
{
  "type": "command.move_to",
  "start_time": 1234567890.5,
  "end_time": 1234567892.0
}
```

## Configuration Options

You can customize the service by modifying the parameters when creating the `AudioTranscriptionService`:

- `sample_rate`: Audio sample rate (default: 44100 Hz)
- `channels`: Number of input channels (default: 4)
- `interval`: Sampling interval in seconds (default: 0.25)
- `target_channel`: Which channel to transcribe (default: 0)
- `buffer_duration`: Duration of audio history to maintain in seconds (default: 300)
- `on_transcription`: Callback function for transcription results (optional)

## Requirements

- Python 3.12 or higher
- A 4-channel audio input device
- OpenAI API key with access to GPT-4o Realtime model
