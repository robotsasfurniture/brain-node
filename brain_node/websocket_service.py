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
        attempt = 1
        while not self._connected:
            try:
                logger.info(
                    f"Attempting to connect to ROS2 WebSocket at {self.uri} (attempt {attempt})"
                )
                self.websocket = await websockets.connect(self.uri)
                self._connected = True
                self._reconnect_delay = 1  # Reset delay on successful connection
                logger.info(f"Successfully connected to ROS2 WebSocket at {self.uri}")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {attempt} failed: {e}")
                logger.info(f"Retrying in {self._reconnect_delay} seconds...")
                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff with maximum delay
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
                attempt += 1

    async def send_location(self, location: LocalizationResult) -> None:
        """Send location data to ROS2 via WebSocket.

        Args:
            location: LocationResult containing angle and distance information
        """
        if not self._connected:
            logger.error("Cannot send location: WebSocket not connected")
            return

        try:
            # Create ROS2 message format
            message = {
                "angle": float(location.angle),
                "distance": float(location.distance),
            }

            await self.websocket.send(json.dumps(message))
            logger.debug(
                f"Sent location data: angle={location.angle:.2f}°, distance={location.distance:.2f}m"
            )

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False
            raise ConnectionError("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error sending location data: {e}")
            self._connected = False
            raise

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

    def __init__(self):
        self._connected = False

    async def connect(self) -> None:
        logger.info("Mock WebSocket connecting...")
        await asyncio.sleep(0.5)  # Simulate connection delay
        self._connected = True
        logger.info("Mock WebSocket connected")

    async def send_location(self, location: LocalizationResult) -> None:
        if not self._connected:
            raise ConnectionError("Mock WebSocket not connected")
        logger.info(
            f"Mock sending location: angle={location.angle:.2f}°, distance={location.distance:.2f}m"
        )

    async def close(self) -> None:
        self._connected = False
        logger.info("Mock WebSocket closed")
