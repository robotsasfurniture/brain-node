import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scipy.signal import spectrogram
from dataclasses import dataclass
from typing import Optional
from .logging_config import setup_logger

logger = setup_logger(__name__, log_file="logs/sound_localization.log")


@dataclass
class LocalizationResult:
    """Represents the result of sound localization."""

    angle: float
    distance: float
    x: float
    y: float

    def __init__(self, angle: float, distance: float):
        self.angle = angle
        self.distance = distance
        # Calculate x, y coordinates from polar coordinates
        angle_rad = np.deg2rad(angle)
        self.x = distance * np.cos(angle_rad)
        self.y = distance * np.sin(angle_rad)


class SoundLocalizationNet(nn.Module):
    """Neural network for sound source localization."""

    def __init__(self):
        super(SoundLocalizationNet, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(
            4, 32, kernel_size=3, stride=1, padding=1
        )  # 4 channels for 4 microphones
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Transformer encoder
        self.transformer_layer = TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=64, dropout=0, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_layer, num_layers=2
        )

        # Fully connected layers
        self.fc1 = nn.Linear(192 * 64, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 3)  # Output: 3D coordinates (x, y, z)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers with ReLU activation
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Reshape for transformer
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Flatten and apply fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class SoundLocalizer:
    """Class for performing sound source localization."""

    def __init__(self, model_path: str, sample_rate: int = 44100):
        """
        Initialize the sound localizer.

        Args:
            model_path: Path to the pre-trained PyTorch model
            sample_rate: Audio sample rate (default: 44100)
        """
        self.model = SoundLocalizationNet()
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.sample_rate = sample_rate

    def _generate_spectrogram(self, audio_data: np.ndarray) -> torch.Tensor:
        """
        Generate spectrogram from audio data.

        Args:
            audio_data: Input audio data
        Returns:
            Normalized spectrogram as torch tensor
        """
        _, _, sxx = spectrogram(audio_data, fs=self.sample_rate)
        sxx_normalized = (sxx - np.mean(sxx)) / (np.std(sxx) + 1e-8)
        return torch.from_numpy(sxx_normalized).float()

    def localize_from_arrays(self, streams: np.ndarray) -> Optional[LocalizationResult]:
        """
        Localize sound source from array of audio streams.

        Args:
            streams: Array of shape (num_channels, num_samples)
        Returns:
            LocalizationResult or None if localization fails
        """
        if streams.ndim != 2:
            raise ValueError("Input streams must be a 2D array")
        if streams.shape[0] < 4:
            raise ValueError("At least 4 channels required")

        streams = streams.T
        logger.debug(f"Localizing sound in shape of {streams.shape}")

        try:
            # Process each channel
            spectrograms = [
                self._generate_spectrogram(streams[i, :])
                for i in range(streams.shape[0])
            ]
            spectrograms = torch.stack(spectrograms, dim=0)
            spectrograms = spectrograms.unsqueeze(0)  # Add batch dimension

            # Get predictions
            with torch.no_grad():
                predictions = self.model(spectrograms)

            # Extract coordinates
            x, y, _ = predictions.squeeze(0).numpy()

            # Calculate polar coordinates
            distance = np.sqrt(x**2 + y**2)
            angle = np.arctan2(y, x) * (180 / np.pi)

            return LocalizationResult(angle=angle, distance=distance)

        except Exception as e:
            logger.error(f"Error during localization: {e}")
            return None
