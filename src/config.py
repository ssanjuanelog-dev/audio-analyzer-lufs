"""
Configuration Module for Audio Analyzer LUFS Application

Manages application settings, audio parameters, and user preferences.
Stores configuration in JSON format for persistence across sessions.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 48000
    buffer_size: int = 2048
    fft_size: int = 4096
    hop_length: int = 1024
    channels: int = 2
    bits_per_sample: int = 24


@dataclass
class LUFSConfig:
    """LUFS metering configuration."""
    target_lufs: float = -23.0
    gate_threshold: float = -70.0
    true_peak_limit: float = 0.0
    standard: str = "ITU-R BS.1770-4"


class AppConfig:
    """Application configuration manager."""

    def __init__(self, config_file: str = 'config.json'):
        """Initialize application configuration.
        
        Args:
            config_file: Path to configuration JSON file
        """
        self.config_file = Path(config_file)
        self.audio = AudioConfig()
        self.lufs = LUFSConfig()
        self.load()

    def load(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.audio = AudioConfig(**data.get('audio', {}))
                self.lufs = LUFSConfig(**data.get('lufs', {}))

    def save(self) -> None:
        """Save configuration to file."""
        config_data = {
            'audio': asdict(self.audio),
            'lufs': asdict(self.lufs)
        }
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    def reset_defaults(self) -> None:
        """Reset configuration to default values."""
        self.audio = AudioConfig()
        self.lufs = LUFSConfig()
        self.save()
