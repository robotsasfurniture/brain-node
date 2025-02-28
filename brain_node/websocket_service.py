import asyncio
import websockets
import json
from typing import Optional, Protocol
from .sound_localization import LocalizationResult
from .logging_config import setup_logger

logger = setup_logger(__name__, log_file="logs/websocket.log")


class WebSocketProvider(Protocol):
    """Protocol defining the interface for WebSocket communication."""

    async def connect(self) -> None:
        """Establish connection to the WebSocket server."""
        ...

    async def send_location(self, location: LocalizationResult) -> None:
        """Send location data via WebSocket.

        Args:
            location: LocationResult containing angle and distance information
        """
        ...

    async def close(self) -> None:
        """Close the WebSocket connection."""
        ...


class Ros2WebSocketService(WebSocketProvider):
    """ROS2-specific WebSocket implementation."""

    def __init__(self, uri: str = "ws://localhost:9090"):
        """Initialize WebSocket service for ROS2 communication.

        Args:
            uri: WebSocket URI for ROS2 bridge (default: ws://localhost:9090)
        """
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_delay = 1  # Start with 1 second delay
        self._max_reconnect_delay = 30  # Maximum delay between reconnection attempts

    async def connect(self) -> None:
        """Establish WebSocket connection with reconnection logic."""
        while not self._connected:
            try:
                self.websocket = await websockets.connect(self.uri)
                self._connected = True
                self._reconnect_delay = 1  # Reset delay on successful connection
                logger.info(f"Connected to ROS2 WebSocket at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to WebSocket: {e}")
                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff with maximum delay
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def send_location(self, location: LocalizationResult) -> None:
        """Send location data to ROS2 via WebSocket.

        Args:
            location: LocationResult containing angle and distance information
        """
        if not self._connected:
            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Failed to establish connection: {e}")
                return

        try:
            # Create ROS2 message format
            message = {
                "op": "publish",
                "topic": "/audio/location",
                "msg": {
                    "header": {
                        "frame_id": "base_link",
                        "stamp": {"sec": 0, "nanosec": 0},
                    },
                    "angle": location.angle,
                    "distance": location.distance,
                    "x": location.x,
                    "y": location.y,
                },
            }

            await self.websocket.send(json.dumps(message))
            logger.debug(
                f"Sent location data: angle={location.angle:.2f}°, distance={location.distance:.2f}m"
            )

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed, attempting to reconnect...")
            self._connected = False
            await self.connect()
        except Exception as e:
            logger.error(f"Error sending location data: {e}")
            self._connected = False

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.websocket:
            try:
                await self.websocket.close()
                self._connected = False
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")


class MockWebSocketService(WebSocketProvider):
    """Mock WebSocket service for testing."""

    async def connect(self) -> None:
        logger.info("Mock WebSocket connected")

    async def send_location(self, location: LocalizationResult) -> None:
        logger.info(
            f"Mock sending location: angle={location.angle:.2f}°, distance={location.distance:.2f}m"
        )

    async def close(self) -> None:
        logger.info("Mock WebSocket closed")
