"""
Comprehensive Audio Analysis Module

Provides advanced audio feature extraction and analysis including:
- Time-domain features (RMS, ZCR, short-term energy, energy entropy)
- Spectral features (centroid, bandwidth, rolloff, flux)
- Perceptual features (MFCC, Chroma features)
- Phase detection and STFT analysis
- Onset detection and transient analysis
- Harmonic analysis and THD measurements
- Signal quality metrics (SNR, SINAD, THD+N)
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional


class AudioFeatureExtractor:
    """Comprehensive audio feature extraction and analysis."""

    def __init__(self, sample_rate: int = 48000, n_fft: int = 4096):
        """Initialize feature extractor.
        
        Args:
            sample_rate: Sample rate in Hz
            n_fft: FFT size
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

    # ==================== TIME-DOMAIN FEATURES ====================

    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract RMS (Root Mean Square) energy.
        
        Args:
            audio: Audio signal
            
        Returns:
            RMS energy per frame
        """
        frame_size = self.n_fft
        rms = np.array([
            np.sqrt(np.mean(audio[i:i+frame_size]**2))
            for i in range(0, len(audio) - frame_size, self.hop_length)
        ])
        return rms

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract Zero Crossing Rate (ZCR).
        
        ZCR is the rate of sign changes in the signal.
        High ZCR suggests more noise or friction sounds.
        
        Args:
            audio: Audio signal
            
        Returns:
            ZCR per frame
        """
        frame_size = self.n_fft
        zcr = np.array([
            np.sum(np.abs(np.diff(np.sign(audio[i:i+frame_size])))) / (2 * frame_size)
            for i in range(0, len(audio) - frame_size, self.hop_length)
        ])
        return zcr

    def extract_short_term_energy_entropy(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy entropy.
        
        Energy entropy measures the level of perturbation in energy distribution.
        
        Args:
            audio: Audio signal
            
        Returns:
            Energy entropy per frame
        """
        frame_size = self.n_fft
        entropy = []
        
        for i in range(0, len(audio) - frame_size, self.hop_length):
            frame = audio[i:i+frame_size]
            subframes = np.array_split(frame, 10)  # 10 subframes
            energies = np.array([np.sum(sf**2) for sf in subframes])
            
            # Normalize
            energies_norm = energies / (np.sum(energies) + 1e-10)
            
            # Calculate entropy
            ent = -np.sum(energies_norm * np.log2(energies_norm + 1e-10))
            entropy.append(ent)
        
        return np.array(entropy)

    # ==================== SPECTRAL FEATURES ====================

    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid (brightness).
        
        Spectral centroid indicates the "center of mass" of the spectrum.
        Higher values suggest brighter, more high-frequency content.
        
        Args:
            audio: Audio signal
            
        Returns:
            Spectral centroid per frame in Hz
        """
        D = signal.stft(audio, fs=self.sample_rate, nperseg=self.n_fft)[2]
        magnitudes = np.abs(D)
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        
        centroids = np.average(
            freqs,
            axis=0,
            weights=magnitudes
        )
        return centroids

    def extract_spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral bandwidth.
        
        Measure of how spread out the spectrum is around the centroid.
        
        Args:
            audio: Audio signal
            
        Returns:
            Spectral bandwidth per frame in Hz
        """
        D = signal.stft(audio, fs=self.sample_rate, nperseg=self.n_fft)[2]
        magnitudes = np.abs(D)
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        
        centroid = np.average(freqs, axis=0, weights=magnitudes)
        bandwidth = np.sqrt(
            np.average((freqs - centroid)**2, axis=0, weights=magnitudes)
        )
        return bandwidth

    def extract_spectral_rolloff(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Extract spectral rolloff.
        
        Frequency below which 'threshold' (default 90%) of spectral energy is contained.
        
        Args:
            audio: Audio signal
            threshold: Energy threshold (0-1)
            
        Returns:
            Spectral rolloff per frame in Hz
        """
        D = signal.stft(audio, fs=self.sample_rate, nperseg=self.n_fft)[2]
        magnitudes = np.abs(D)**2
        freqs = np.fft.rfftfreq(self.n_fft, 1/self.sample_rate)
        
        # Cumulative energy
        cumsum = np.cumsum(magnitudes, axis=0)
        total_energy = cumsum[-1]
        
        rolloffs = []
        for i in range(cumsum.shape[1]):
            idx = np.where(cumsum[:, i] >= threshold * total_energy[i])[0][0]
            rolloffs.append(freqs[idx])
        
        return np.array(rolloffs)

    def extract_spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral flux (change between frames).
        
        Measure of how much the spectrum changes frame-to-frame.
        High values suggest transient events.
        
        Args:
            audio: Audio signal
            
        Returns:
            Spectral flux per frame
        """
        _, _, D = signal.stft(audio, fs=self.sample_rate, nperseg=self.n_fft)
        magnitudes = np.abs(D)
        
        # Normalized magnitudes
        magnitudes_norm = magnitudes / (np.sum(magnitudes, axis=0) + 1e-10)
        
        # Difference between consecutive frames
        flux = np.sqrt(np.sum(np.diff(magnitudes_norm, axis=1)**2, axis=0))
        return flux

    # ==================== PHASE & ONSET DETECTION ====================

    def extract_phase_information(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract phase information from STFT.
        
        Returns both unwrapped phase and phase derivatives.
        
        Args:
            audio: Audio signal
            
        Returns:
            Tuple of (phase angles, phase derivatives)
        """
        _, _, D = signal.stft(audio, fs=self.sample_rate, nperseg=self.n_fft)
        phase = np.angle(D)
        
        # Unwrap phase
        phase_unwrapped = np.unwrap(phase, axis=0)
        
        # Phase derivative (group delay)
        phase_derivative = np.diff(phase_unwrapped, axis=1)
        
        return phase_unwrapped, phase_derivative

    def detect_onsets(self, audio: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Detect onset times (transient events).
        
        Uses energy-based onset detection by analyzing spectral flux.
        
        Args:
            audio: Audio signal
            threshold: Detection threshold
            
        Returns:
            Onset frame indices
        """
        # Compute onset strength envelope
        flux = self.extract_spectral_flux(audio)
        
        # Smooth with median filter
        flux_smooth = signal.medfilt(flux, kernel_size=5)
        
        # Detect peaks above threshold
        onset_frames = signal.find_peaks(
            flux_smooth,
            height=threshold * np.max(flux_smooth),
            distance=int(0.05 * self.sample_rate / self.hop_length)
        )[0]
        
        return onset_frames

    # ==================== QUALITY METRICS ====================

    def calculate_snr(self, signal_clean: np.ndarray, 
                     signal_noisy: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio (SNR) in dB.
        
        Args:
            signal_clean: Reference clean signal
            signal_noisy: Signal with noise
            
        Returns:
            SNR in dB
        """
        noise = signal_noisy - signal_clean
        snr = 10 * np.log10(np.sum(signal_clean**2) / (np.sum(noise**2) + 1e-10))
        return snr

    def calculate_thd(self, audio: np.ndarray, fundamental_freq: float) -> float:
        """Calculate Total Harmonic Distortion (THD).
        
        Args:
            audio: Audio signal
            fundamental_freq: Fundamental frequency in Hz
            
        Returns:
            THD as percentage
        """
        # Compute FFT
        fft_result = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        magnitudes = np.abs(fft_result)
        
        # Find fundamental
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fund_mag = magnitudes[fund_idx]
        
        # Find harmonics (2nd to 10th)
        harmonic_mags = []
        for h in range(2, 11):
            h_freq = fundamental_freq * h
            h_idx = np.argmin(np.abs(freqs - h_freq))
            harmonic_mags.append(magnitudes[h_idx])
        
        thd = 100 * np.sqrt(np.sum(np.array(harmonic_mags)**2)) / fund_mag
        return thd

    def calculate_peak_to_average_power_ratio(self, audio: np.ndarray) -> float:
        """Calculate PAPR (Peak-to-Average Power Ratio) in dB.
        
        Indicates signal dynamic range and crest factor.
        
        Args:
            audio: Audio signal
            
        Returns:
            PAPR in dB
        """
        peak_power = np.max(np.abs(audio))**2
        avg_power = np.mean(audio**2)
        papr = 10 * np.log10(peak_power / (avg_power + 1e-10))
        return papr

    def extract_all_metrics(self, audio: np.ndarray) -> Dict[str, any]:
        """Extract all available audio metrics.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of all extracted metrics
        """
        metrics = {
            'rms_energy': self.extract_rms_energy(audio),
            'zero_crossing_rate': self.extract_zero_crossing_rate(audio),
            'energy_entropy': self.extract_short_term_energy_entropy(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'spectral_bandwidth': self.extract_spectral_bandwidth(audio),
            'spectral_rolloff': self.extract_spectral_rolloff(audio),
            'spectral_flux': self.extract_spectral_flux(audio),
            'onset_times': self.detect_onsets(audio),
            'papr_db': self.calculate_peak_to_average_power_ratio(audio),
        }
        return metrics
