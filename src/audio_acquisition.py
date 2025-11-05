"""
Módulo de Adquisición de Audio
Captura de audio en tiempo real desde micrófono y lectura de archivos
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Tuple, Optional, Callable
import threading

class AudioAcquisition:
    """Clase para capturar y procesar audio en tiempo real"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2, block_size: int = 2048):
        """
        Inicializar adquisición de audio
        
        Args:
            sample_rate: Frecuencia de muestreo (Hz)
            channels: Número de canales (1=mono, 2=estéreo)
            block_size: Tamaño del bloque de audio
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        
    def start_microphone_stream(self, callback: Callable):
        """Iniciar captura desde micrófono"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio Status: {status}")
            callback(indata.copy())
            
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=audio_callback
        )
        self.stream.start()
        self.is_recording = True
        
    def stop_microphone_stream(self):
        """Detener captura desde micrófono"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            
    def load_audio_file(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Cargar archivo de audio
        
        Returns:
            audio_data: Array de audio
            sample_rate: Frecuencia de muestreo
        """
        audio_data, sr = sf.read(filepath)
        return audio_data, sr
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalizar audio al rango [-1, 1]"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
        
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Remuestrear audio a frecuencia objetivo"""
        if orig_sr == target_sr:
            return audio
        
        num_samples = int(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(audio), num_samples),
            np.arange(len(audio)),
            audio
        )
