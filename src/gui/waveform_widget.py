"""Waveform Display Widget - Real-time audio waveform visualization.

Provides a real-time waveform display with zoom, pan, and marker capabilities.
Supports both mono and stereo audio display with professional styling.

Best Repositories Referenced:
- matplotlib: https://github.com/matplotlib/matplotlib
- PyQt6: https://github.com/PyQt/PyQt6
- numpy: https://github.com/numpy/numpy

Author: ssanjuanelog
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams

# Configure matplotlib for dark theme
rcParams['figure.facecolor'] = '#1e1e1e'
rcParams['axes.facecolor'] = '#2d2d2d'
rcParams['axes.edgecolor'] = '#404040'
rcParams['axes.labelcolor'] = '#00ff00'
rcParams['xtick.color'] = '#00ff00'
rcParams['ytick.color'] = '#00ff00'
rcParams['text.color'] = '#00ff00'


class WaveformWidget(QWidget):
    """Real-time audio waveform display widget.
    
    Displays waveform with zoom, pan, and marker capabilities.
    Supports mono and stereo visualization.
    """
    
    # Signals
    marker_added = pyqtSignal(float)  # Time position in seconds
    range_selected = pyqtSignal(float, float)  # Start and end time
    
    def __init__(self, sample_rate: int = 44100, parent=None):
        """Initialize waveform widget.
        
        Args:
            sample_rate: Audio sample rate in Hz (default 44100)
            parent: Parent widget
        """
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.audio_data = None
        self.is_stereo = False
        self.zoom_level = 1.0
        self.pan_offset = 0.0
        self.markers = []
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        
        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Mouse tracking
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def set_audio_data(self, audio: np.ndarray) -> None:
        """Set audio data for display.
        
        Args:
            audio: Audio data (mono or stereo)
        """
        self.audio_data = audio
        self.is_stereo = audio.ndim == 2 and audio.shape[1] == 2
        self.pan_offset = 0.0
        self.zoom_level = 1.0
        self.update_display()
        
    def update_display(self) -> None:
        """Update waveform display."""
        if self.audio_data is None:
            return
            
        self.axes.clear()
        
        # Calculate time axis
        duration = len(self.audio_data) / self.sample_rate
        time = np.linspace(0, duration, len(self.audio_data))
        
        # Apply zoom and pan
        total_duration = duration / self.zoom_level
        start_time = self.pan_offset
        end_time = start_time + total_duration
        
        # Clamp to valid range
        start_time = max(0, min(start_time, duration - total_duration))
        end_time = min(duration, start_time + total_duration)
        
        # Find sample indices
        start_idx = int(start_time * self.sample_rate)
        end_idx = int(end_time * self.sample_rate)
        
        if self.is_stereo:
            # Stereo display
            left = self.audio_data[start_idx:end_idx, 0]
            right = self.audio_data[start_idx:end_idx, 1]
            time_zoom = time[start_idx:end_idx]
            
            # Plot left and right channels
            self.axes.plot(time_zoom, left, color='#00ff00', alpha=0.7, linewidth=0.5, label='Left')
            self.axes.plot(time_zoom, right, color='#ff00ff', alpha=0.7, linewidth=0.5, label='Right')
            self.axes.legend(loc='upper right')
        else:
            # Mono display
            mono = self.audio_data[start_idx:end_idx]
            time_zoom = time[start_idx:end_idx]
            self.axes.plot(time_zoom, mono, color='#00ff00', alpha=0.8, linewidth=0.5)
        
        # Draw markers
        for marker_time in self.markers:
            if start_time <= marker_time <= end_time:
                self.axes.axvline(x=marker_time, color='#ffff00', linestyle='--', alpha=0.5)
        
        # Configure axes
        self.axes.set_xlabel('Time (s)', color='#00ff00')
        self.axes.set_ylabel('Amplitude', color='#00ff00')
        self.axes.set_ylim(-1.0, 1.0)
        self.axes.grid(True, alpha=0.2, color='#404040')
        
        self.canvas.draw()
        
    def on_scroll(self, event) -> None:
        """Handle scroll wheel for zoom."""
        if event.inaxes != self.axes:
            return
            
        if event.button == 'up':
            self.zoom_level *= 1.2  # Zoom in
        elif event.button == 'down':
            self.zoom_level /= 1.2  # Zoom out
        
        self.zoom_level = max(1.0, min(self.zoom_level, 100.0))
        self.update_display()
        
    def on_press(self, event) -> None:
        """Handle mouse press."""
        if event.inaxes != self.axes:
            return
        if event.xdata is None:
            return
            
        if event.button == 1:  # Left click
            # Add marker
            self.markers.append(event.xdata)
            self.marker_added.emit(event.xdata)
            self.update_display()
        elif event.button == 3:  # Right click
            # Remove nearest marker
            if self.markers:
                nearest = min(self.markers, key=lambda m: abs(m - event.xdata))
                if abs(nearest - event.xdata) < 0.1:
                    self.markers.remove(nearest)
                    self.update_display()
    
    def on_motion(self, event) -> None:
        """Handle mouse motion for panning."""
        pass
    
    def sizeHint(self) -> QSize:
        """Return preferred size."""
        return QSize(1200, 300)
    
    def clear(self) -> None:
        """Clear waveform display."""
        self.audio_data = None
        self.markers = []
        self.axes.clear()
        self.canvas.draw()
