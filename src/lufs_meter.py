"""
Módulo de Medición LUFS (ITU-R BS.1770-4)
Implementación de loudness metering conforme a estándares internacionales
"""

import numpy as np
import pyloudnorm
from scipy import signal
from typing import Dict, Tuple

class LUFSMeter:
    """Clase para medir loudness en LUFS (ITU-R BS.1770-4)"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.meter = pyloudnorm.Meter(sample_rate)
        
    def measure_loudness(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Medir loudness del audio
        Returns: Dict con LUFS, LRA, true_peak
        """
        loudness = self.meter.integrated_loudness(audio)
        return {"LUFS": loudness}
        
    def measure_short_term(self, audio: np.ndarray, block_size: int = 3) -> float:
        """Medir loudness de corto plazo (3 segundos)"""
        meter_st = pyloudnorm.Meter(self.sample_rate, block_size=block_size)
        return meter_st.integrated_loudness(audio)
        
    def measure_momentary(self, audio: np.ndarray) -> float:
        """Medir loudness momentáneo (0.4 segundos)"""
        meter_m = pyloudnorm.Meter(self.sample_rate, block_size=0.4)
        return meter_m.integrated_loudness(audio)
        
    def compute_true_peak(self, audio: np.ndarray) -> float:
        """
        Calcular True Peak (pico verdadero)
        Sobre muestreo y búsqueda de picos
        """
        L = int(4 * len(audio))  # Over-sample 4x
        resampled = signal.resample(audio, L)
        return np.max(np.abs(resampled))
        
    def normalize_loudness(self, audio: np.ndarray, target_lufs: float = -23) -> np.ndarray:
        """Normalizar audio a loudness objetivo"""
        loudness = self.meter.integrated_loudness(audio)
        normalized = pyloudnorm.normalize(audio, loudness, target_lufs)
        return normalized
