import argparse
import os
import sys
from typing import Optional

from .sound_localization import SoundLocalizer
from .audio_service import AudioSampler
from .audio_transcription import ChatGptService, run_service
from .websocket_service import Ros2WebSocketService, MockWebSocketService
from .visualization import LocalizationVisualizer


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Brain Node - Real-time audio transcription service with ChatGPT"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate in Hz (default: 44100)",
    )

    parser.add_argument(
        "--channels", type=int, default=4, help="Number of audio channels (default: 4)"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Sampling interval in seconds (default: 0.25)",
    )

    parser.add_argument(
        "--buffer-duration",
        type=int,
        default=60,
        help="Duration of audio history to maintain in seconds (default: 60)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)",
    )

    parser.add_argument(
        "--mock-websocket",
        action="store_true",
        help="Use mock WebSocket service instead of real ROS2 connection",
    )

    parser.add_argument(
        "--websocket-uri",
        type=str,
        default="ws://localhost:9090",
        help="WebSocket URI for ROS2 bridge (default: ws://localhost:9090)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable real-time visualization of sound source localization",
    )

    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """Validate command line arguments and return error message if invalid."""
    if args.sample_rate <= 0:
        return "Sample rate must be positive"

    if args.channels <= 0:
        return "Number of channels must be positive"

    if args.interval <= 0:
        return "Sampling interval must be positive"

    if args.buffer_duration <= 0:
        return "Buffer duration must be positive"

    if not args.api_key and not os.getenv("OPENAI_API_KEY"):
        return "OpenAI API key must be provided either via --api-key or OPENAI_API_KEY environment variable"

    return None


def main():
    """Main entry point for the brain-node command."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    error = validate_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    # Set API key if provided via command line
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    try:
        # Create audio sampler
        audio_sampler = AudioSampler(
            sample_rate=args.sample_rate, channels=args.channels, interval=args.interval
        )

        # Create WebSocket service
        websocket_service = (
            MockWebSocketService()
            if args.mock_websocket
            else Ros2WebSocketService(uri=args.websocket_uri)
        )

        # Create visualizer if enabled
        visualizer = LocalizationVisualizer() if args.visualize else None

        # Create the service
        service = ChatGptService(
            audio_provider=audio_sampler,
            websocket_service=websocket_service,
            buffer_duration=args.buffer_duration,
            localizer=SoundLocalizer(model_path="models/model_directivity.pth"),
            visualizer=visualizer,
        )

        print("Starting Brain Node transcription service...")
        print("Press Ctrl+C to stop")

        # Run the service in the event loop
        run_service(service)

    except KeyboardInterrupt:
        print("\nShutting down...")
        if args.visualize and visualizer:
            visualizer.close()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.visualize and visualizer:
            visualizer.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
