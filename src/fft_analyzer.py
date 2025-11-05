"""
Módulo de Análisis FFT
Transformada rápida de Fourier para análisis de espectro en tiempo real
"""

import numpy as np
from scipy import signal, fftpack
from typing import Tuple, List

class FFTAnalyzer:
    """Clase para realizar análisis FFT en tiempo real"""
    
    def __init__(self, fft_size: int = 2048, sample_rate: int = 44100):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.frequencies = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
    def compute_fft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcular FFT del audio"""
        window = signal.get_window('hann', len(audio))
        windowed = audio * window
        fft_result = np.fft.rfft(windowed, n=self.fft_size)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        return self.frequencies, magnitude_db
        
    def compute_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """Calcular espectrograma Mel"""
        D = np.abs(np.fft.rfft(audio, n=self.fft_size))
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(
            S=D, sr=self.sample_rate, n_mels=n_mels
        ))
        return mel_spec
        
    def compute_bark_scale(self, frequencies: np.ndarray) -> np.ndarray:
        """Convertir frecuencias a escala Bark"""
        return 13 * np.arctan(0.76 * frequencies / 1000) + \
               3.5 * np.arctan((frequencies / 7500) ** 2)
