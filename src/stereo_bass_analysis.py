"""Stereo & Bass Analysis Module - Professional stereo imaging and bass tracking.

Provides comprehensive mono/stereo analysis, phase coherence checking,
and advanced bass frequency tracking with best repository integrations.

Best Repositories Referenced:
- librosa: https://github.com/librosa/librosa
- scipy.signal: https://github.com/scipy/scipy  
- numpy: https://github.com/numpy/numpy
- essentia: https://github.com/MTG/essentia

Author: ssanjuanelog
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import signal
from scipy.fft import fft, fftfreq


class StereoMode(Enum):
    """Stereo channel configuration modes."""
    MONO = "mono"
    STEREO = "stereo"
    STEREO_WIDE = "stereo_wide"
    MID_SIDE = "mid_side"
    SURROUND = "surround"


@dataclass
class StereoMetrics:
    """Stereo imaging and phase analysis results."""
    mode: str
    correlation_coefficient: float  # -1 to 1 (0 = perfectly uncorrelated)
    phase_difference_degrees: float  # Average phase difference L-R
    stereo_width: float  # 0-1 (1 = maximum width)
    mono_compatibility: float  # 0-100% (100 = fully compatible)
    imaging_balance: float  # -100 to 100 (-100 = left heavy, 100 = right heavy)
    phase_coherence: float  # 0-1 (1 = perfectly coherent)
    bass_mono_level: float  # dB level of mono bass
    bass_stereo_level: float  # dB level of stereo bass separation


class BassFrequencyBands(Enum):
    """Professional bass frequency band definitions."""
    SUB_BASS = (20, 60)        # Ultra-low sub-bass (rumble)
    DEEP_BASS = (60, 120)      # Deep bass foundation
    MID_BASS = (120, 250)      # Mid-bass punch and weight
    BASS_MID = (250, 500)      # Bass-midrange transition
    LOW_MID = (500, 1000)      # Low-midrange presence


class BassBuoyancy(Enum):
    """Bass presence and energy classification."""
    DEPLETED = "depleted"      # Bass too quiet (-12 dB or less)
    LIGHT = "light"            # Bass light and weak (-6 to -12 dB)
    BALANCED = "balanced"      # Bass well-balanced (0 to -6 dB)
    BOOSTED = "boosted"        # Bass prominent (-3 to +3 dB)
    HEAVY = "heavy"            # Bass too heavy (+3 to +6 dB)
    OVERWHELMING = "overwhelming"  # Bass overwhelming (+6 dB or more)


class StereoAnalyzer:
    """Analyze stereo imaging, phase relationships, and spatial characteristics."""
    
    def __init__(self, sample_rate: int = 48000, log_level: str = 'INFO'):
        """Initialize stereo analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            log_level: Logging level
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
    
    def analyze_stereo_field(self, audio: np.ndarray) -> StereoMetrics:
        """Analyze stereo field characteristics.
        
        Args:
            audio: Stereo audio (2-channel array)
            
        Returns:
            Stereo metrics analysis
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("Input must be stereo audio (N x 2)")
        
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Calculate correlation coefficient (phase relationship)
        correlation = np.corrcoef(left, right)[0, 1]
        
        # Calculate phase difference
        left_fft = fft(left)
        right_fft = fft(right)
        phase_diff = np.mean(np.abs(np.angle(right_fft) - np.angle(left_fft)))
        phase_degrees = np.degrees(phase_diff)
        
        # Calculate stereo width (0 to 1, where 1 = maximum width)
        stereo_width = 1 - abs(correlation)
        
        # Mono compatibility (100% = mono compatible, 0% = completely stereo)
        mono_compatibility = (1 + correlation) / 2 * 100
        
        # Imaging balance (-100 to 100)
        left_energy = np.sqrt(np.mean(left ** 2))
        right_energy = np.sqrt(np.mean(right ** 2))
        balance = (right_energy - left_energy) / (right_energy + left_energy + 1e-10) * 100
        
        # Phase coherence (how well L and R track together)
        phase_coherence = 1 - (phase_degrees / 180)  # Normalize to 0-1
        
        # Bass analysis (below 250 Hz)
        bass_cutoff = 250
        nyquist = self.sample_rate / 2
        normalized_cutoff = bass_cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        left_bass = signal.filtfilt(b, a, left)
        right_bass = signal.filtfilt(b, a, right)
        
        bass_mono_level = 20 * np.log10(np.sqrt(np.mean(left_bass ** 2)) + 1e-10)
        bass_stereo_level = 20 * np.log10(
            np.sqrt(np.mean((left_bass - right_bass) ** 2)) + 1e-10
        )
        
        return StereoMetrics(
            mode=StereoMode.STEREO.value,
            correlation_coefficient=correlation,
            phase_difference_degrees=phase_degrees,
            stereo_width=max(0, min(1, stereo_width)),
            mono_compatibility=max(0, min(100, mono_compatibility)),
            imaging_balance=max(-100, min(100, balance)),
            phase_coherence=max(0, min(1, phase_coherence)),
            bass_mono_level=bass_mono_level,
            bass_stereo_level=bass_stereo_level
        )
    
    def get_stereo_recommendations(self, metrics: StereoMetrics) -> List[str]:
        """Get stereo imaging recommendations.
        
        Args:
            metrics: Stereo metrics from analysis
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Correlation assessment
        if metrics.correlation_coefficient > 0.95:
            recommendations.append("Audio is nearly mono - consider adding more stereo width")
        elif metrics.correlation_coefficient > 0.8:
            recommendations.append("Stereo width is narrow - can be increased for more spaciousness")
        elif metrics.correlation_coefficient < -0.5:
            recommendations.append("WARNING: Phase issues detected - L/R channels are highly inverted")
        
        # Phase assessment
        if metrics.phase_difference_degrees > 120:
            recommendations.append("Large phase differences detected - check for phase issues")
        
        # Imaging balance
        if abs(metrics.imaging_balance) > 30:
            direction = "right" if metrics.imaging_balance > 0 else "left"
            recommendations.append(f"Audio is heavily panned to {direction} - consider rebalancing")
        
        # Mono compatibility
        if metrics.mono_compatibility < 70:
            recommendations.append("WARNING: Poor mono compatibility - may cause issues on mono playback")
        
        # Bass analysis
        if metrics.bass_stereo_level > metrics.bass_mono_level - 3:
            recommendations.append("Bass has significant stereo separation - consider mono-izing for compatibility")
        
        return recommendations


class BassAnalyzer:
    """Advanced bass frequency tracking and analysis."""
    
    def __init__(self, sample_rate: int = 48000, log_level: str = 'INFO'):
        """Initialize bass analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            log_level: Logging level
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.bands = BassFrequencyBands
    
    def analyze_all_bass_bands(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze all bass frequency bands.
        
        Args:
            audio: Audio signal (mono or stereo mean)
            
        Returns:
            Dictionary with bass analysis for all bands
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        analysis = {}
        nyquist = self.sample_rate / 2
        
        for band_name, (low_freq, high_freq) in self.bands.__members__.items():
            # Design band-pass filter
            norm_low = low_freq / nyquist
            norm_high = high_freq / nyquist
            norm_low = max(0.001, min(norm_low, 0.999))
            norm_high = max(norm_low + 0.001, min(norm_high, 0.999))
            
            b, a = signal.butter(5, [norm_low, norm_high], btype='band')
            band_signal = signal.filtfilt(b, a, audio)
            
            # Analyze band
            rms = np.sqrt(np.mean(band_signal ** 2))
            db_level = 20 * np.log10(rms + 1e-10)
            peak = np.max(np.abs(band_signal))
            crest_factor = peak / (rms + 1e-10) if rms > 0 else 0
            
            # Determine band buoyancy
            if db_level < -12:
                buoyancy = BassBuoyancy.DEPLETED
            elif db_level < -6:
                buoyancy = BassBuoyancy.LIGHT
            elif db_level < 0:
                buoyancy = BassBuoyancy.BALANCED
            elif db_level < 3:
                buoyancy = BassBuoyancy.BOOSTED
            elif db_level < 6:
                buoyancy = BassBuoyancy.HEAVY
            else:
                buoyancy = BassBuoyancy.OVERWHELMING
            
            analysis[band_name.lower()] = {
                'frequency_range': (low_freq, high_freq),
                'level_db': db_level,
                'rms': rms,
                'peak': peak,
                'crest_factor': crest_factor,
                'buoyancy': buoyancy.value
            }
        
        return analysis
    
    def get_bass_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Get bass optimization recommendations.
        
        Args:
            analysis: Bass analysis results
            
        Returns:
            List of bass improvement recommendations
        """
        recommendations = []
        
        # Sub-bass analysis
        sub_bass = analysis.get('sub_bass', {})
        if sub_bass.get('level_db', -100) < -6:
            recommendations.append("Sub-bass is too quiet - consider boosting for more weight (use with caution)")
        elif sub_bass.get('level_db', 0) > 6:
            recommendations.append("Sub-bass too loud - can mask mid-bass definition")
        
        # Deep bass
        deep_bass = analysis.get('deep_bass', {})
        if deep_bass.get('level_db', -100) > 0:
            recommendations.append("Deep bass is boosted - ensure it doesn't muddy the mix")
        
        # Mid-bass
        mid_bass = analysis.get('mid_bass', {})
        if mid_bass.get('buoyancy') == BassBuoyancy.HEAVY.value:
            recommendations.append("Mid-bass is too prominent - may compete with kick drum")
        elif mid_bass.get('buoyancy') == BassBuoyancy.DEPLETED.value:
            recommendations.append("Mid-bass is too weak - add punch with careful EQ boost")
        
        # Overall bass energy
        total_energy = sum(
            10 ** (v.get('level_db', -100) / 20) 
            for v in analysis.values() 
            if isinstance(v, dict) and 'level_db' in v
        )
        
        if total_energy > 10:
            recommendations.append("Overall bass energy is high - check for rumble or mud")
        elif total_energy < 0.1:
            recommendations.append("Overall bass energy is low - may need general bass boost")
        
        return recommendations


if __name__ == "__main__":
    stereo_analyzer = StereoAnalyzer()
    bass_analyzer = BassAnalyzer()
    print("Stereo & Bass Analyzer initialized")
    print(f"Bass bands: {[b.name for b in BassFrequencyBands]}")
