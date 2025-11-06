"""Mixing Analysis Module - Professional mixing quality assessment.

Provides detailed analysis for mixing tasks including EQ balance,
dynamic range, frequency distribution, and frequency-dependent metrics.

Author: ssanjuanelog
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.fftpack import fft


@dataclass
class FrequencyBand:
    """Frequency band analysis result."""
    name: str
    frequency_range: Tuple[float, float]
    energy_db: float
    peak_frequency: float
    bandwidth: float
    quality_factor: float


@dataclass
class MixingMetrics:
    """Complete mixing analysis metrics."""
    # Frequency balance
    sub_bass: FrequencyBand  # 20-80 Hz
    bass: FrequencyBand     # 80-250 Hz
    low_mids: FrequencyBand  # 250-800 Hz
    mids: FrequencyBand     # 800-2500 Hz
    high_mids: FrequencyBand # 2500-8000 Hz
    presence: FrequencyBand  # 8000-16000 Hz
    air: FrequencyBand      # 16000-20000 Hz
    
    # Dynamic metrics
    crest_factor: float
    dynamic_range: float
    rms_level: float
    peak_level: float
    
    # Frequency balance score
    frequency_balance_score: float
    
    # Clipping detection
    has_clipping: bool
    clipping_percentage: float
    
    # Noise floor
    noise_floor_db: float
    snr: float


class MixingAnalyzer:
    """Analyze audio for mixing optimization."""
    
    # Standard frequency bands for professional mixing
    FREQUENCY_BANDS = {
        'sub_bass': (20, 80),
        'bass': (80, 250),
        'low_mids': (250, 800),
        'mids': (800, 2500),
        'high_mids': (2500, 8000),
        'presence': (8000, 16000),
        'air': (16000, 20000)
    }
    
    def __init__(self, sample_rate: int = 48000, frame_size: int = 2048,
                 log_level: str = 'INFO'):
        """Initialize mixing analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: FFT frame size for analysis
            log_level: Logging level
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
    
    def analyze_mixing(self, audio: np.ndarray) -> MixingMetrics:
        """Perform comprehensive mixing analysis.
        
        Args:
            audio: Audio signal
            
        Returns:
            Complete mixing metrics
        """
        # Normalize to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Compute frequency bands
        bands = self._analyze_frequency_bands(audio)
        
        # Compute dynamic metrics
        crest_factor = self._compute_crest_factor(audio)
        dynamic_range = self._compute_dynamic_range(audio)
        rms_level = self._compute_rms_level(audio)
        peak_level = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        
        # Check for clipping
        has_clipping, clipping_pct = self._detect_clipping(audio)
        
        # Compute noise floor
        noise_floor = self._estimate_noise_floor(audio)
        snr = rms_level - noise_floor
        
        # Frequency balance score
        freq_score = self._compute_frequency_balance_score(bands)
        
        return MixingMetrics(
            sub_bass=bands['sub_bass'],
            bass=bands['bass'],
            low_mids=bands['low_mids'],
            mids=bands['mids'],
            high_mids=bands['high_mids'],
            presence=bands['presence'],
            air=bands['air'],
            crest_factor=crest_factor,
            dynamic_range=dynamic_range,
            rms_level=rms_level,
            peak_level=peak_level,
            frequency_balance_score=freq_score,
            has_clipping=has_clipping,
            clipping_percentage=clipping_pct,
            noise_floor_db=noise_floor,
            snr=snr
        )
    
    def _analyze_frequency_bands(self, audio: np.ndarray) -> Dict[str, FrequencyBand]:
        """Analyze energy in standard frequency bands.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of band analyses
        """
        # Compute FFT
        n = len(audio)
        freq_response = np.abs(fft(audio, n=self.frame_size))
        freqs = np.fft.fftfreq(self.frame_size, 1/self.sample_rate)[:self.frame_size//2]
        freq_response = freq_response[:self.frame_size//2]
        freq_response_db = 20 * np.log10(freq_response + 1e-10)
        
        bands = {}
        for band_name, (low_freq, high_freq) in self.FREQUENCY_BANDS.items():
            # Get indices for frequency range
            idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
            
            if len(idx) > 0:
                band_energy = freq_response[idx]
                band_energy_db = 20 * np.log10(np.mean(band_energy) + 1e-10)
                peak_idx = np.argmax(band_energy)
                peak_freq = freqs[idx[peak_idx]]
                
                # Compute bandwidth (3dB point)
                peak_level = band_energy_db[peak_idx]
                half_power = peak_level - 3
                above_half_power = np.sum(band_energy_db > half_power)
                bandwidth = (above_half_power / len(idx)) * (high_freq - low_freq)
                
                # Compute Q factor
                q_factor = peak_freq / (bandwidth + 1e-10)
                
                bands[band_name] = FrequencyBand(
                    name=band_name,
                    frequency_range=(low_freq, high_freq),
                    energy_db=band_energy_db,
                    peak_frequency=peak_freq,
                    bandwidth=bandwidth,
                    quality_factor=q_factor
                )
        
        return bands
    
    def _compute_crest_factor(self, audio: np.ndarray) -> float:
        """Compute crest factor (peak/RMS ratio).
        
        Args:
            audio: Audio signal
            
        Returns:
            Crest factor in dB
        """
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        if rms > 0:
            return 20 * np.log10(peak / rms)
        return 0.0
    
    def _compute_dynamic_range(self, audio: np.ndarray, threshold_db: float = -80) -> float:
        """Compute dynamic range.
        
        Args:
            audio: Audio signal
            threshold_db: Noise threshold in dB
            
        Returns:
            Dynamic range in dB
        """
        peak_level = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        return peak_level - threshold_db
    
    def _compute_rms_level(self, audio: np.ndarray) -> float:
        """Compute RMS level in dB.
        
        Args:
            audio: Audio signal
            
        Returns:
            RMS level in dB
        """
        rms = np.sqrt(np.mean(audio ** 2))
        return 20 * np.log10(rms + 1e-10)
    
    def _detect_clipping(self, audio: np.ndarray, threshold: float = 0.99) -> Tuple[bool, float]:
        """Detect clipping in audio.
        
        Args:
            audio: Audio signal (assumed normalized -1 to 1)
            threshold: Clipping threshold
            
        Returns:
            Tuple of (has_clipping, clipping_percentage)
        """
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        clipping_pct = (clipped_samples / len(audio)) * 100
        has_clipping = clipping_pct > 0.01  # More than 0.01% clipping
        return has_clipping, clipping_pct
    
    def _estimate_noise_floor(self, audio: np.ndarray, percentile: float = 5) -> float:
        """Estimate noise floor level.
        
        Args:
            audio: Audio signal
            percentile: Percentile for noise floor estimation
            
        Returns:
            Noise floor in dB
        """
        magnitudes_db = 20 * np.log10(np.abs(audio) + 1e-10)
        noise_floor = np.percentile(magnitudes_db, percentile)
        return noise_floor
    
    def _compute_frequency_balance_score(self, bands: Dict[str, FrequencyBand]) -> float:
        """Compute frequency balance quality score (0-100).
        
        Args:
            bands: Dictionary of frequency bands
            
        Returns:
            Balance score 0-100 (100 is ideal)
        """
        # Get energy values
        energies = np.array([bands[name].energy_db for name in 
                           ['sub_bass', 'bass', 'low_mids', 'mids', 
                            'high_mids', 'presence', 'air']])
        
        # Ideal is relatively flat (minimal variance)
        energy_std = np.std(energies)
        energy_variance_penalty = min(energy_std * 2, 30)  # Max penalty 30 points
        
        # Penalize if bass is too quiet or too loud
        bass_level = (bands['bass'].energy_db + bands['sub_bass'].energy_db) / 2
        bass_penalty = abs(bass_level - (-6)) / 4  # Ideal bass -6dB
        bass_penalty = min(bass_penalty, 20)
        
        # Penalize if presence is too quiet (should be fairly present)
        presence_level = bands['presence'].energy_db
        presence_penalty = max(0, (presence_level + 12) / 3)  # Ideal presence -12dB
        presence_penalty = min(presence_penalty, 15)
        
        total_penalty = energy_variance_penalty + bass_penalty + presence_penalty
        score = max(0, 100 - total_penalty)
        
        return score
    
    def get_mixing_recommendations(self, metrics: MixingMetrics) -> List[str]:
        """Get recommendations for mix improvement.
        
        Args:
            metrics: Mixing metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check clipping
        if metrics.has_clipping:
            recommendations.append(f"CRITICAL: Clipping detected ({metrics.clipping_percentage:.2f}%) - reduce level")
        
        # Check dynamic range
        if metrics.dynamic_range < 12:
            recommendations.append("Low dynamic range - increase compression control")
        elif metrics.dynamic_range > 30:
            recommendations.append("High dynamic range - may need compression for consistency")
        
        # Check frequency balance
        if metrics.frequency_balance_score < 60:
            recommendations.append(f"Poor frequency balance (score: {metrics.frequency_balance_score:.1f}) - apply EQ corrections")
        
        # Bass analysis
        if metrics.sub_bass.energy_db > 2:
            recommendations.append("Sub-bass is too loud - consider reducing 20-80Hz")
        elif metrics.sub_bass.energy_db < -20:
            recommendations.append("Sub-bass too quiet - boost 20-80Hz for power")
        
        # Mid analysis  
        if metrics.mids.energy_db > 3:
            recommendations.append("Mids too prominent - slightly reduce 800-2500Hz")
        
        # Presence analysis
        if metrics.presence.energy_db < -20:
            recommendations.append("Lack of presence - boost 8-16kHz for clarity")
        
        # SNR analysis
        if metrics.snr < 30:
            recommendations.append(f"Noisy mix (SNR: {metrics.snr:.1f}dB) - improve recording quality or use noise gate")
        
        return recommendations if recommendations else ["Mix looks balanced - good work!"]


if __name__ == "__main__":
    analyzer = MixingAnalyzer()
    print("Mixing Analyzer initialized")
    print(f"Standard mixing frequency bands: {list(MixingAnalyzer.FREQUENCY_BANDS.keys())}")
