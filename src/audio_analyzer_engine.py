"""Audio Analyzer Engine - Main integration module.

Integrates all audio analysis components into a unified engine for
real-time analysis and batch processing with comprehensive metrics.

Author: ssanjuanelog
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime

# Import all analysis modules
from audio_acquisition import AudioAcquisition
from fft_analyzer import FFTAnalyzer
from lufs_meter import LUFSMeter
from audio_analysis import AudioFeatureExtractor
from export import DataExporter
from database import AnalysisDatabase


class AnalysisMode(Enum):
    """Audio analysis modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class AnalysisResult:
    """Complete analysis result from all modules."""
    timestamp: str
    filename: Optional[str]
    duration: float
    sample_rate: int
    channels: int
    
    # LUFS metrics
    integrated_lufs: float
    short_term_lufs: float
    momentary_lufs: float
    true_peak: float
    loudness_range: float
    
    # Spectrum data
    spectrum_linear: np.ndarray
    spectrum_log: np.ndarray
    spectrum_mel: np.ndarray
    spectrum_bark: np.ndarray
    frequency_bins: np.ndarray
    
    # Time-domain features
    rms_energy: np.ndarray
    zero_crossing_rate: np.ndarray
    energy_entropy: np.ndarray
    
    # Spectral features
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_flux: np.ndarray
    
    # Phase and onset detection
    phase_information: Dict[str, np.ndarray]
    onset_times: np.ndarray
    onset_strengths: np.ndarray
    
    # Quality metrics
    snr_db: float
    thd_percent: float
    papr_db: float
    
    # Compliance indicators
    standards_compliance: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
        return result_dict


class AudioAnalyzerEngine:
    """Central audio analysis engine integrating all components."""
    
    def __init__(self, sample_rate: int = 48000, frame_size: int = 2048,
                 hop_size: int = 512, log_level: str = 'INFO'):
        """Initialize audio analyzer engine.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: FFT frame size
            hop_size: Hop size between frames
            log_level: Logging level
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Initialize all analysis modules
        self.audio_acq = AudioAcquisition(sample_rate)
        self.fft_analyzer = FFTAnalyzer(sample_rate, frame_size, hop_size)
        self.lufs_meter = LUFSMeter(sample_rate)
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.exporter = DataExporter()
        self.database = AnalysisDatabase()
        
        # Analysis state
        self.current_audio = None
        self.current_result = None
        
        self.logger.info(f"Audio Analyzer Engine initialized (SR: {sample_rate} Hz, "
                        f"Frame: {frame_size}, Hop: {hop_size})")
    
    def analyze_file(self, filepath: str) -> AnalysisResult:
        """Analyze complete audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Complete analysis result
        """
        self.logger.info(f"Starting file analysis: {filepath}")
        
        # Load audio
        audio, sr = self.audio_acq.load_audio(filepath)
        if sr != self.sample_rate:
            self.logger.warning(f"Sample rate mismatch: {sr} vs {self.sample_rate}")
        
        # Perform analysis
        result = self._analyze_audio(audio, sr, filepath)
        
        # Store in database
        self.database.store_analysis(result)
        
        self.logger.info(f"File analysis completed")
        return result
    
    def analyze_stream(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Analyze streaming audio chunk for real-time processing.
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            Real-time metrics dictionary
        """
        metrics = {}
        
        # LUFS measurement
        lufs_data = self.lufs_meter.measure(audio_chunk)
        metrics['momentary_lufs'] = lufs_data.get('momentary', -np.inf)
        
        # FFT analysis
        spectrum_data = self.fft_analyzer.analyze(audio_chunk)
        metrics['spectrum'] = {
            'linear': spectrum_data['spectrum_linear'],
            'frequency': spectrum_data['frequencies']
        }
        
        # Feature extraction
        features = self.feature_extractor.extract_all_metrics(audio_chunk)
        metrics['features'] = features
        
        return metrics
    
    def analyze_batch(self, file_list: List[str]) -> List[AnalysisResult]:
        """Analyze multiple audio files in batch.
        
        Args:
            file_list: List of file paths
            
        Returns:
            List of analysis results
        """
        results = []
        for filepath in file_list:
            try:
                result = self.analyze_file(filepath)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing {filepath}: {e}")
        return results
    
    def _analyze_audio(self, audio: np.ndarray, sr: int,
                       filename: Optional[str] = None) -> AnalysisResult:
        """Internal method to perform complete audio analysis.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            filename: Source filename
            
        Returns:
            Complete analysis result
        """
        # Ensure stereo for processing
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        duration = len(audio) / sr
        channels = audio.shape[1] if audio.ndim > 1 else 1
        
        # Process each channel
        audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        
        # LUFS measurement
        lufs_data = self.lufs_meter.measure(audio_mono)
        
        # FFT analysis
        spectrum_data = self.fft_analyzer.analyze(audio_mono)
        
        # Feature extraction
        features = self.feature_extractor.extract_all_metrics(audio_mono)
        
        # Compliance check
        standards_compliance = self._check_compliance(lufs_data)
        
        # Create result object
        result = AnalysisResult(
            timestamp=datetime.now().isoformat(),
            filename=filename,
            duration=duration,
            sample_rate=sr,
            channels=channels,
            
            # LUFS metrics
            integrated_lufs=lufs_data.get('integrated', -np.inf),
            short_term_lufs=lufs_data.get('short_term', -np.inf),
            momentary_lufs=lufs_data.get('momentary', -np.inf),
            true_peak=lufs_data.get('true_peak', -np.inf),
            loudness_range=lufs_data.get('lra', 0.0),
            
            # Spectrum data
            spectrum_linear=spectrum_data['spectrum_linear'],
            spectrum_log=spectrum_data['spectrum_log'],
            spectrum_mel=spectrum_data['spectrum_mel'],
            spectrum_bark=spectrum_data['spectrum_bark'],
            frequency_bins=spectrum_data['frequencies'],
            
            # Time-domain features
            rms_energy=features.get('rms_energy', np.array([])),
            zero_crossing_rate=features.get('zero_crossing_rate', np.array([])),
            energy_entropy=features.get('energy_entropy', np.array([])),
            
            # Spectral features
            spectral_centroid=features.get('spectral_centroid', np.array([])),
            spectral_bandwidth=features.get('spectral_bandwidth', np.array([])),
            spectral_rolloff=features.get('spectral_rolloff', np.array([])),
            spectral_flux=features.get('spectral_flux', np.array([])),
            
            # Phase and onset
            phase_information=features.get('phase_information', {}),
            onset_times=features.get('onset_times', np.array([])),
            onset_strengths=features.get('onset_strengths', np.array([])),
            
            # Quality metrics
            snr_db=features.get('snr', -np.inf),
            thd_percent=features.get('thd', 0.0),
            papr_db=features.get('papr', 0.0),
            
            # Standards compliance
            standards_compliance=standards_compliance
        )
        
        self.current_result = result
        return result
    
    def _check_compliance(self, lufs_data: Dict[str, float]) -> Dict[str, bool]:
        """Check compliance with audio standards.
        
        Args:
            lufs_data: LUFS measurement data
            
        Returns:
            Dictionary of standard compliance status
        """
        integrated = lufs_data.get('integrated', -np.inf)
        
        return {
            'ITU-R BS.1770-4': -23 <= integrated <= -19,
            'EBU R128': -23 <= integrated <= -19,
            'LKFS': -16 <= integrated <= -14,
            'ATSC A/85': -27 <= integrated <= -24
        }
    
    def export_result(self, result: AnalysisResult, format: str = 'json',
                     output_path: Optional[str] = None) -> str:
        """Export analysis result.
        
        Args:
            result: Analysis result to export
            format: Export format ('json', 'pdf', 'png')
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        return self.exporter.export(result.to_dict(), format, output_path)
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve analysis history.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of previous analysis results
        """
        return self.database.get_history(limit)
    
    def clear_history(self) -> None:
        """Clear analysis history."""
        self.database.clear_history()
        self.logger.info("Analysis history cleared")


def create_engine(config_path: Optional[str] = None) -> AudioAnalyzerEngine:
    """Factory function to create configured analyzer engine.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured AudioAnalyzerEngine instance
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return AudioAnalyzerEngine(**config)
    return AudioAnalyzerEngine()


if __name__ == "__main__":
    # Example usage
    engine = create_engine()
    print(f"Audio Analyzer Engine ready")
    print(f"Sample rate: {engine.sample_rate} Hz")
    print(f"Frame size: {engine.frame_size}")
    print(f"Hop size: {engine.hop_size}")
