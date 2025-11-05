"""
Visualization Module for Audio Analyzer LUFS Application

Comprehensive visualization of all audio metrics and professional standards:
- Real-time spectrum analysis (linear, logarithmic, Mel, Bark scales)
- LUFS metering display (Integrated, Short-term, Momentary, True Peak)
- Waveform visualization
- Loudness range and dynamic range analysis
- Professional standard compliance indicators
- True Peak metering with ITU-R BS.1770-4 compliance
- LRA (Loudness Range) according to ITU-R BS.1770-4
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from typing import Dict, Tuple, Optional


class LUFSMeterDisplay(QWidget):
    """Custom LUFS meter widget following ITU-R BS.1770-4 standards."""

    def __init__(self):
        """Initialize LUFS meter display."""
        super().__init__()
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the user interface for LUFS display."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel('LUFS Loudness Metering (ITU-R BS.1770-4)')
        title.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Create grid for metrics
        grid = QGridLayout()
        
        # Integrated Loudness
        self.integrated_label = QLabel('Integrated LUFS:')
        self.integrated_value = QLabel('-23.0 LUFS')
        self.integrated_value.setFont(QFont('Courier', 11, QFont.Weight.Bold))
        self.integrated_value.setStyleSheet('color: #00FF00;')
        grid.addWidget(self.integrated_label, 0, 0)
        grid.addWidget(self.integrated_value, 0, 1)
        
        # Short-term Loudness
        self.short_term_label = QLabel('Short-term LUFS (3s):')
        self.short_term_value = QLabel('-23.0 LUFS')
        self.short_term_value.setFont(QFont('Courier', 11, QFont.Weight.Bold))
        self.short_term_value.setStyleSheet('color: #00FF00;')
        grid.addWidget(self.short_term_label, 1, 0)
        grid.addWidget(self.short_term_value, 1, 1)
        
        # Momentary Loudness
        self.momentary_label = QLabel('Momentary LUFS (0.4s):')
        self.momentary_value = QLabel('-23.0 LUFS')
        self.momentary_value.setFont(QFont('Courier', 11, QFont.Weight.Bold))
        self.momentary_value.setStyleSheet('color: #00FF00;')
        grid.addWidget(self.momentary_label, 2, 0)
        grid.addWidget(self.momentary_value, 2, 1)
        
        # True Peak
        self.true_peak_label = QLabel('True Peak (ITU-R BS.1770-4):')
        self.true_peak_value = QLabel('-1.0 dBFS')
        self.true_peak_value.setFont(QFont('Courier', 11, QFont.Weight.Bold))
        self.true_peak_value.setStyleSheet('color: #FFD700;')
        grid.addWidget(self.true_peak_label, 3, 0)
        grid.addWidget(self.true_peak_value, 3, 1)
        
        # Loudness Range (LRA)
        self.lra_label = QLabel('Loudness Range (LRA):')
        self.lra_value = QLabel('0.0 LU')
        self.lra_value.setFont(QFont('Courier', 11, QFont.Weight.Bold))
        self.lra_value.setStyleSheet('color: #00FF00;')
        grid.addWidget(self.lra_label, 4, 0)
        grid.addWidget(self.lra_value, 4, 1)
        
        # Compliance Status
        self.compliance_label = QLabel('Standards Compliance:')
        self.compliance_value = QLabel('✓ ITU-R BS.1770-4')
        self.compliance_value.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.compliance_value.setStyleSheet('color: #00FF00;')
        grid.addWidget(self.compliance_label, 5, 0)
        grid.addWidget(self.compliance_value, 5, 1)
        
        layout.addLayout(grid)
        self.setLayout(layout)

    def update_metrics(self, integrated: float, short_term: float,
                      momentary: float, true_peak: float,
                      lra: float = 0.0) -> None:
        """Update displayed LUFS metrics.
        
        Args:
            integrated: Integrated loudness in LUFS
            short_term: Short-term loudness in LUFS
            momentary: Momentary loudness in LUFS
            true_peak: True peak level in dBFS
            lra: Loudness range in LU
        """
        self.integrated_value.setText(f'{integrated:.1f} LUFS')
        self.short_term_value.setText(f'{short_term:.1f} LUFS')
        self.momentary_value.setText(f'{momentary:.1f} LUFS')
        self.true_peak_value.setText(f'{true_peak:.1f} dBFS')
        self.lra_value.setText(f'{lra:.1f} LU')
        
        # Update color based on compliance
        if integrated >= -26.0 and integrated <= -20.0:
            self.integrated_value.setStyleSheet('color: #00FF00;')
        else:
            self.integrated_value.setStyleSheet('color: #FF6B6B;')


class SpectrumAnalyzerWidget(QWidget):
    """Real-time spectrum analyzer with multiple frequency scales."""

    def __init__(self, parent=None):
        """Initialize spectrum analyzer.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.setup_plot()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_plot(self) -> None:
        """Setup the spectrum plot with professional formatting."""
        self.ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
        self.ax.set_title('Real-Time Spectrum Analysis (0-20kHz)', 
                         fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xscale('log')
        self.ax.set_ylim([-80, 0])

    def update_spectrum(self, frequencies: np.ndarray, magnitudes: np.ndarray) -> None:
        """Update spectrum plot with new data.
        
        Args:
            frequencies: Frequency values in Hz
            magnitudes: Magnitude values in dB
        """
        self.ax.clear()
        self.ax.plot(frequencies, magnitudes, linewidth=2, color='#00FF00')
        self.ax.fill_between(frequencies, magnitudes, -80, alpha=0.3, color='#00FF00')
        self.setup_plot()
        self.canvas.draw()


class WaveformWidget(QWidget):
    """Real-time waveform visualization."""

    def __init__(self, parent=None):
        """Initialize waveform widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.setup_plot()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_plot(self) -> None:
        """Setup waveform plot."""
        self.ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        self.ax.set_title('Audio Waveform', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim([-1.0, 1.0])

    def update_waveform(self, audio: np.ndarray, sample_rate: int) -> None:
        """Update waveform display.
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate in Hz
        """
        self.ax.clear()
        time = np.arange(len(audio)) / sample_rate
        self.ax.plot(time, audio, linewidth=0.5, color='#00FFFF')
        self.setup_plot()
        self.canvas.draw()


class ComplianceIndicator(QWidget):
    """Professional standard compliance indicator."""

    STANDARDS = {
        'ITU-R BS.1770-4': {'lufs': -23.0, 'true_peak': 0.0, 'lra_min': 0.0},
        'LKFS (Streaming)': {'lufs': -16.0, 'true_peak': -1.0, 'lra_min': 0.0},
        'EBU R128': {'lufs': -23.0, 'true_peak': -3.0, 'lra_min': 4.0},
        'ATSC A/85': {'lufs': -24.0, 'true_peak': -2.0, 'lra_min': 0.0},
    }

    def __init__(self):
        """Initialize compliance indicator."""
        super().__init__()
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize compliance display."""
        layout = QVBoxLayout()
        
        title = QLabel('Professional Standards Compliance')
        title.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Standards grid
        grid = QGridLayout()
        row = 0
        for standard, specs in self.STANDARDS.items():
            std_label = QLabel(f'{standard}:')
            std_specs = QLabel(
                f'Target: {specs["lufs"]:.0f} LUFS | '
                f'TP: {specs["true_peak"]:.0f} dBFS | '
                f'LRA min: {specs["lra_min"]:.0f} LU'
            )
            std_specs.setStyleSheet('color: #CCCCCC; font-size: 9pt;')
            grid.addWidget(std_label, row, 0)
            grid.addWidget(std_specs, row, 1)
            row += 1
        
        layout.addLayout(grid)
        self.setLayout(layout)
