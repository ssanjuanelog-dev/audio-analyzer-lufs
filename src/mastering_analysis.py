"""Mastering Analysis Module - Professional audio mastering standards and optimization.

Provides comprehensive mastering analysis including loudness targets,
headroom calculations, dynamic range analysis, and platform-specific optimization.

Author: ssanjuanelog
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class MasteringStandard(Enum):
    """Professional mastering standards."""
    BROADCAST = "broadcast"
    STREAMING = "streaming"
    CD_AUDIO = "cd_audio"
    VINYL = "vinyl"
    CINEMA = "cinema"
    PODCAST = "podcast"


@dataclass
class MasteringTargets:
    """Mastering loudness and technical targets."""
    integrated_lufs: float
    short_term_lufs: float
    loudness_range_min: float
    loudness_range_max: float
    true_peak_max: float
    headroom_db: float
    dynamic_range_min: float
    dynamic_range_max: float
    
    def __str__(self) -> str:
        return f"Integrated: {self.integrated_lufs}LUFS | TP: {self.true_peak_max}dBFS | HR: {self.headroom_db}dB"


class MasteringStandardsDatabase:
    """Database of professional mastering standards."""
    
    TARGETS: Dict[MasteringStandard, MasteringTargets] = {
        MasteringStandard.BROADCAST: MasteringTargets(
            integrated_lufs=-23,
            short_term_lufs=-20,
            loudness_range_min=0,
            loudness_range_max=15,
            true_peak_max=-1.0,
            headroom_db=1.0,
            dynamic_range_min=8,
            dynamic_range_max=30
        ),
        MasteringStandard.STREAMING: MasteringTargets(
            integrated_lufs=-14,
            short_term_lufs=-11,
            loudness_range_min=0,
            loudness_range_max=12,
            true_peak_max=-1.0,
            headroom_db=1.0,
            dynamic_range_min=4,
            dynamic_range_max=20
        ),
        MasteringStandard.CD_AUDIO: MasteringTargets(
            integrated_lufs=-9,
            short_term_lufs=-6,
            loudness_range_min=0,
            loudness_range_max=18,
            true_peak_max=-0.3,
            headroom_db=0.3,
            dynamic_range_min=6,
            dynamic_range_max=24
        ),
        MasteringStandard.VINYL: MasteringTargets(
            integrated_lufs=-6,
            short_term_lufs=-3,
            loudness_range_min=0,
            loudness_range_max=14,
            true_peak_max=-3.0,
            headroom_db=3.0,
            dynamic_range_min=8,
            dynamic_range_max=30
        ),
        MasteringStandard.CINEMA: MasteringTargets(
            integrated_lufs=-27,
            short_term_lufs=-24,
            loudness_range_min=0,
            loudness_range_max=20,
            true_peak_max=-2.0,
            headroom_db=2.0,
            dynamic_range_min=10,
            dynamic_range_max=40
        ),
        MasteringStandard.PODCAST: MasteringTargets(
            integrated_lufs=-16,
            short_term_lufs=-13,
            loudness_range_min=0,
            loudness_range_max=10,
            true_peak_max=-1.0,
            headroom_db=1.0,
            dynamic_range_min=3,
            dynamic_range_max=15
        )
    }


class MasteringAnalyzer:
    """Analyze audio for mastering and provide optimization recommendations."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize mastering analyzer.
        
        Args:
            log_level: Logging level
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.db = MasteringStandardsDatabase()
    
    def analyze_for_standard(self, audio: np.ndarray, sr: int,
                           standard: MasteringStandard,
                           integrated_lufs: float,
                           short_term_lufs: float,
                           true_peak: float) -> Dict[str, Any]:
        """Analyze audio for specific mastering standard.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            standard: Mastering standard target
            integrated_lufs: Current integrated LUFS
            short_term_lufs: Current short-term LUFS
            true_peak: Current true peak
            
        Returns:
            Analysis result with recommendations
        """
        targets = self.db.TARGETS[standard]
        
        # Calculate headroom
        headroom = -(true_peak)  # Distance from 0dBFS
        
        # Calculate dynamic range
        dynamic_range = self._calculate_dynamic_range(audio)
        
        # Assess compliance
        loudness_compliant = abs(integrated_lufs - targets.integrated_lufs) <= 0.5
        tp_compliant = true_peak <= targets.true_peak_max
        headroom_compliant = headroom >= targets.headroom_db
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            integrated_lufs, short_term_lufs, true_peak,
            targets, standard
        )
        
        return {
            'standard': standard.value,
            'targets': asdict(targets),
            'current': {
                'integrated_lufs': integrated_lufs,
                'short_term_lufs': short_term_lufs,
                'true_peak': true_peak,
                'headroom': headroom,
                'dynamic_range': dynamic_range
            },
            'compliance': {
                'loudness': loudness_compliant,
                'true_peak': tp_compliant,
                'headroom': headroom_compliant,
                'overall': loudness_compliant and tp_compliant and headroom_compliant
            },
            'adjustments_needed': {
                'loudness_adjustment': targets.integrated_lufs - integrated_lufs,
                'headroom_adjustment': targets.headroom_db - headroom
            },
            'recommendations': recommendations
        }
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate Crest Factor / dynamic range.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dynamic range in dB
        """
        if len(audio) == 0:
            return 0.0
        
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms == 0:
            return 0.0
        
        return 20 * np.log10(peak / rms) if rms > 0 else 0.0
    
    def _generate_recommendations(self, integrated: float, short_term: float,
                                true_peak: float, targets: MasteringTargets,
                                standard: MasteringStandard) -> List[str]:
        """Generate mastering recommendations.
        
        Args:
            integrated: Current integrated LUFS
            short_term: Current short-term LUFS
            true_peak: Current true peak
            targets: Target values
            standard: Mastering standard
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Loudness recommendations
        loudness_diff = abs(integrated - targets.integrated_lufs)
        if loudness_diff > 0.5:
            if integrated > targets.integrated_lufs:
                db_to_reduce = integrated - targets.integrated_lufs
                recommendations.append(
                    f"Reduce loudness by {db_to_reduce:.1f}dB to meet {standard.value} target"
                )
            else:
                db_to_increase = targets.integrated_lufs - integrated
                recommendations.append(
                    f"Increase loudness by {db_to_increase:.1f}dB to meet {standard.value} target"
                )
        else:
            recommendations.append(f"Loudness level meets {standard.value} standard")
        
        # True Peak recommendations
        if true_peak > targets.true_peak_max:
            tp_reduction = true_peak - targets.true_peak_max
            recommendations.append(
                f"Reduce true peak by {tp_reduction:.1f}dB (current: {true_peak}dBFS, max: {targets.true_peak_max}dBFS)"
            )
        else:
            recommendations.append("True peak level is compliant")
        
        # Headroom recommendations
        headroom = -true_peak
        if headroom < targets.headroom_db:
            headroom_needed = targets.headroom_db - headroom
            recommendations.append(
                f"Increase headroom by {headroom_needed:.1f}dB for safety margin"
            )
        
        # Dynamic range feedback
        dr = self._calculate_dynamic_range(np.array([true_peak]) if true_peak != 0 else np.array([0.001]))
        if dr < targets.dynamic_range_min:
            recommendations.append(
                "Audio appears heavily compressed. Consider adding dynamics for clarity."
            )
        elif dr > targets.dynamic_range_max:
            recommendations.append(
                "Audio has excessive dynamic range. Consider gentle compression."
            )
        
        return recommendations
    
    def get_platform_optimization(self, integrated_lufs: float,
                                true_peak: float) -> Dict[str, Dict[str, Any]]:
        """Get optimization suggestions for multiple streaming platforms.
        
        Args:
            integrated_lufs: Current integrated LUFS
            true_peak: Current true peak
            
        Returns:
            Dictionary with per-platform optimization data
        """
        platforms = {
            'Spotify': {'target': -14, 'max_tp': -1.0},
            'Apple Music': {'target': -16, 'max_tp': -1.0},
            'YouTube': {'target': -13, 'max_tp': -1.0},
            'Amazon Music': {'target': -14, 'max_tp': -1.0},
            'Tidal': {'target': -14, 'max_tp': -1.0},
            'Podcast': {'target': -16, 'max_tp': -1.0}
        }
        
        result = {}
        for platform, specs in platforms.items():
            compliant = (
                abs(integrated_lufs - specs['target']) <= 0.5 and
                true_peak <= specs['max_tp']
            )
            result[platform] = {
                'target_lufs': specs['target'],
                'max_true_peak': specs['max_tp'],
                'compliant': compliant,
                'loudness_adjustment': specs['target'] - integrated_lufs,
                'status': 'Optimized' if compliant else 'Needs adjustment'
            }
        
        return result
    
    def estimate_clipping_risk(self, audio: np.ndarray) -> Dict[str, Any]:
        """Estimate risk of digital clipping.
        
        Args:
            audio: Audio signal
            
        Returns:
            Clipping risk analysis
        """
        peak_value = np.max(np.abs(audio))
        clipping_samples = np.sum(np.abs(audio) >= 0.99)
        total_samples = len(audio)
        
        return {
            'peak_level': peak_value,
            'clipping_samples': int(clipping_samples),
            'clipping_percentage': (clipping_samples / total_samples * 100) if total_samples > 0 else 0,
            'clipping_risk': 'HIGH' if clipping_samples > total_samples * 0.001 else 'LOW',
            'recommendation': 'Apply limiter' if clipping_samples > 0 else 'No clipping risk'
        }


if __name__ == "__main__":
    analyzer = MasteringAnalyzer()
    print("Mastering Analysis Module initialized")
    print(f"Supported standards: {len(analyzer.db.TARGETS)}")
    for standard in MasteringStandard:
        print(f"  - {standard.value}: {analyzer.db.TARGETS[standard]}")
