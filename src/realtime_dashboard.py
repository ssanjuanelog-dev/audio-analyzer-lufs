"""Real-time Audio Metrics Dashboard - Live monitoring module.

Provides real-time visualization and monitoring of all audio metrics
with 60fps update capability and professional audio industry display.

Author: ssanjuanelog
License: MIT
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import threading
import logging
from enum import Enum


class MetricStatus(Enum):
    """Status indicators for metrics compliance."""
    OPTIMAL = "optimal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a specific time."""
    timestamp: float
    integrated_lufs: float
    short_term_lufs: float
    momentary_lufs: float
    true_peak: float
    lra: float
    spectrum_magnitude: np.ndarray
    rms_energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    onset_detected: bool


class RealtimeMetricsBuffer:
    """Circular buffer for real-time metrics with configurable window."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize metrics buffer.
        
        Args:
            buffer_size: Maximum number of snapshots to maintain
        """
        self.buffer_size = buffer_size
        self.snapshots: deque = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
    
    def append(self, snapshot: MetricSnapshot) -> None:
        """Add metric snapshot to buffer.
        
        Args:
            snapshot: Metric snapshot to add
        """
        with self.lock:
            self.snapshots.append(snapshot)
    
    def get_latest(self) -> Optional[MetricSnapshot]:
        """Get most recent snapshot.
        
        Returns:
            Latest metric snapshot or None if empty
        """
        with self.lock:
            if self.snapshots:
                return self.snapshots[-1]
        return None
    
    def get_history(self, duration_seconds: float) -> List[MetricSnapshot]:
        """Get snapshots within time window.
        
        Args:
            duration_seconds: Duration window in seconds
            
        Returns:
            List of snapshots within window
        """
        with self.lock:
            if not self.snapshots:
                return []
            
            latest_time = self.snapshots[-1].timestamp
            cutoff_time = latest_time - duration_seconds
            
            return [s for s in self.snapshots if s.timestamp >= cutoff_time]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics for current buffer.
        
        Returns:
            Dictionary of statistics (mean, min, max, std)
        """
        with self.lock:
            if not self.snapshots:
                return {}
            
            lufs_values = [s.momentary_lufs for s in self.snapshots]
            
            return {
                'mean_lufs': float(np.mean(lufs_values)),
                'min_lufs': float(np.min(lufs_values)),
                'max_lufs': float(np.max(lufs_values)),
                'std_lufs': float(np.std(lufs_values)),
                'num_samples': len(lufs_values)
            }
    
    def clear(self) -> None:
        """Clear all buffered data."""
        with self.lock:
            self.snapshots.clear()


class ComplianceMonitor:
    """Monitor audio metrics compliance with professional standards."""
    
    # Standard target ranges (LUFS)
    STANDARDS = {
        'ITU-R BS.1770-4': (-23, -19),
        'EBU R128': (-23, -19),
        'LKFS': (-16, -14),
        'ATSC A/85': (-27, -24),
        'Spotify': (-14, -12),
        'Apple Music': (-16, -14),
        'YouTube': (-13, -11)
    }
    
    def __init__(self):
        """Initialize compliance monitor."""
        self.logger = logging.getLogger(__name__)
        self.current_status = {}
    
    def evaluate(self, lufs_value: float) -> Dict[str, Any]:
        """Evaluate LUFS against all standards.
        
        Args:
            lufs_value: Current integrated LUFS
            
        Returns:
            Dictionary with compliance status for each standard
        """
        compliance = {}
        
        for standard, (lower, upper) in self.STANDARDS.items():
            if lower <= lufs_value <= upper:
                status = MetricStatus.OPTIMAL
            elif lower - 1 <= lufs_value <= upper + 1:
                status = MetricStatus.WARNING
            else:
                status = MetricStatus.CRITICAL
            
            compliance[standard] = {
                'status': status.value,
                'target_range': (lower, upper),
                'current': lufs_value,
                'in_range': status == MetricStatus.OPTIMAL
            }
        
        self.current_status = compliance
        return compliance
    
    def get_recommendations(self, lufs_value: float) -> List[str]:
        """Get adjustment recommendations.
        
        Args:
            lufs_value: Current integrated LUFS
            
        Returns:
            List of adjustment recommendations
        """
        recommendations = []
        
        if lufs_value < -30:
            recommendations.append("Audio is too quiet - increase volume")
        elif lufs_value > -10:
            recommendations.append("Audio is too loud - reduce volume")
        elif lufs_value < -25:
            recommendations.append("Audio is moderately quiet")
        elif lufs_value > -15:
            recommendations.append("Audio is moderately loud")
        
        # Platform-specific recommendations
        spotify_range = self.STANDARDS['Spotify']
        apple_range = self.STANDARDS['Apple Music']
        
        if not (spotify_range[0] <= lufs_value <= spotify_range[1]):
            recommendations.append(f"Not optimal for Spotify (target: {spotify_range[0]} to {spotify_range[1]} LUFS)")
        
        if not (apple_range[0] <= lufs_value <= apple_range[1]):
            recommendations.append(f"Not optimal for Apple Music (target: {apple_range[0]} to {apple_range[1]} LUFS)")
        
        return recommendations


class AnalysisMetricsAggregator:
    """Aggregate and compute derived metrics for dashboard display."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.logger = logging.getLogger(__name__)
        self.buffer = RealtimeMetricsBuffer(buffer_size=2000)
        self.compliance = ComplianceMonitor()
    
    def record_metrics(self, metrics: Dict[str, Any]) -> MetricSnapshot:
        """Record new metrics snapshot.
        
        Args:
            metrics: Dictionary of current metrics
            
        Returns:
            Created metric snapshot
        """
        snapshot = MetricSnapshot(
            timestamp=datetime.now().timestamp(),
            integrated_lufs=metrics.get('integrated_lufs', -np.inf),
            short_term_lufs=metrics.get('short_term_lufs', -np.inf),
            momentary_lufs=metrics.get('momentary_lufs', -np.inf),
            true_peak=metrics.get('true_peak', -np.inf),
            lra=metrics.get('lra', 0.0),
            spectrum_magnitude=metrics.get('spectrum', np.array([])),
            rms_energy=metrics.get('rms_energy', 0.0),
            spectral_centroid=metrics.get('spectral_centroid', 0.0),
            zero_crossing_rate=metrics.get('zcr', 0.0),
            onset_detected=metrics.get('onset_detected', False)
        )
        
        self.buffer.append(snapshot)
        return snapshot
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data.
        
        Returns:
            Dictionary of dashboard display data
        """
        latest = self.buffer.get_latest()
        if not latest:
            return {}
        
        stats = self.buffer.get_statistics()
        compliance = self.compliance.evaluate(latest.integrated_lufs)
        recommendations = self.compliance.get_recommendations(latest.integrated_lufs)
        
        return {
            'current': {
                'integrated_lufs': latest.integrated_lufs,
                'short_term_lufs': latest.short_term_lufs,
                'momentary_lufs': latest.momentary_lufs,
                'true_peak': latest.true_peak,
                'lra': latest.lra,
                'rms_energy': latest.rms_energy,
                'spectral_centroid': latest.spectral_centroid,
                'zcr': latest.zero_crossing_rate,
                'onset_detected': latest.onset_detected,
                'timestamp': latest.timestamp
            },
            'statistics': stats,
            'compliance': compliance,
            'recommendations': recommendations,
            'history_size': len(self.buffer.snapshots)
        }
    
    def get_loudness_history(self, duration_seconds: float = 10) -> Dict[str, Any]:
        """Get loudness history for graph plotting.
        
        Args:
            duration_seconds: Duration window in seconds
            
        Returns:
            Dictionary with time series data for plotting
        """
        history = self.buffer.get_history(duration_seconds)
        
        if not history:
            return {}
        
        base_time = history[0].timestamp
        times = [(s.timestamp - base_time) for s in history]
        integrated = [s.integrated_lufs for s in history]
        momentary = [s.momentary_lufs for s in history]
        short_term = [s.short_term_lufs for s in history]
        
        return {
            'times': times,
            'integrated': integrated,
            'momentary': momentary,
            'short_term': short_term,
            'duration': duration_seconds
        }
    
    def get_spectrum_data(self, frequency_scale: str = 'linear') -> Dict[str, Any]:
        """Get spectrum data for frequency analysis display.
        
        Args:
            frequency_scale: Scale type ('linear', 'log', 'mel', 'bark')
            
        Returns:
            Dictionary with spectrum visualization data
        """
        latest = self.buffer.get_latest()
        if not latest or len(latest.spectrum_magnitude) == 0:
            return {}
        
        spectrum = latest.spectrum_magnitude
        
        # Compute frequency-weighted spectrum for visualization
        if frequency_scale == 'log':
            # Log scaling for frequency
            spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-10))
        else:
            spectrum_db = spectrum
        
        return {
            'magnitude': spectrum_db.tolist() if hasattr(spectrum_db, 'tolist') else spectrum_db,
            'scale': frequency_scale,
            'peak_frequency': float(np.argmax(spectrum)),
            'peak_magnitude': float(np.max(spectrum))
        }
    
    def export_session_report(self) -> Dict[str, Any]:
        """Export complete session analysis report.
        
        Returns:
            Dictionary with complete session statistics
        """
        stats = self.buffer.get_statistics()
        history = self.buffer.get_history(duration_seconds=float('inf'))
        
        if not history:
            return {}
        
        duration = history[-1].timestamp - history[0].timestamp
        onset_count = sum(1 for s in history if s.onset_detected)
        
        return {
            'session_duration': duration,
            'total_samples': len(history),
            'loudness_statistics': stats,
            'onset_count': onset_count,
            'average_spectrum_centroid': float(np.mean([s.spectral_centroid for s in history])),
            'average_zcr': float(np.mean([s.zero_crossing_rate for s in history])),
            'peak_rms_energy': float(np.max([s.rms_energy for s in history]))
        }
    
    def clear_session(self) -> None:
        """Clear all session data."""
        self.buffer.clear()
        self.logger.info("Session data cleared")


if __name__ == "__main__":
    # Example usage
    aggregator = AnalysisMetricsAggregator()
    print("Real-time Metrics Dashboard initialized")
    print(f"Compliance Standards: {len(aggregator.compliance.STANDARDS)}")
    for standard in aggregator.compliance.STANDARDS:
        print(f"  - {standard}")
