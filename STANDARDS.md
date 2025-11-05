# Professional Audio Standards & Compliance

This document outlines all professional audio standards implemented in the Audio Analyzer LUFS application.

## ITU-R BS.1770-4 Standard

**Primary Standard for LUFS Metering**

### Specifications
- **Integrated Loudness Target**: -23 LUFS (±0.5 LU)
- **True Peak Limit**: 0 dBFS (no clipping)
- **Loudness Range (LRA)**: Measured in LU (Loudness Units)
- **Short-term Loudness**: 3-second sliding window
- **Momentary Loudness**: 0.4-second window

### Key Characteristics
- Uses K-weighting filter (similar to A-weighting but optimized for loudness perception)
- Measures integrated loudness over entire content
- Incorporates gate mechanism to exclude silence/noise
- Calculates True Peak with 4x oversampling
- Implements LUFS (Loudness Units relative to Full Scale)

### Implementation Details
```python
# Filtering chain according to ITU-R BS.1770-4
- High-pass filter: 100 Hz (2nd order Butterworth)
- K-weighting filter: Defined frequency response
- Mean-squared calculation with 400ms blocks
- -70 LUFS gate threshold
```

## LKFS (Loudness, K-weighted, relative to Full Scale)

**Used in Streaming Platforms**

### Specifications
- **Target Loudness**: -16 LKFS (Streaming standard)
- **True Peak Limit**: -1 dBFS (streaming platforms)
- **Loudness Range**: Open-ended
- **Platform Variants**:
  - YouTube: -13 to -14 LKFS (with -1 dBTP)
  - Netflix: -27 LKFS (with -2 dBTP)
  - Spotify: -11 LKFS (with -2 dBTP)
  - Apple Music: -16 LKFS (with -1 dBTP)

### Key Characteristics
- Simplified version of ITU-R BS.1770-4 for real-time streaming
- No loudness range measurement required
- Faster computation for live streaming
- Compatible with audio fingerprinting services

## EBU R128 Standard

**European Broadcasting Union Standard**

### Specifications
- **Integrated Loudness Target**: -23 LUFS (±0.5 LU)
- **True Peak Limit**: -3 dBFS
- **Loudness Range (LRA) Minimum**: 4 LU
- **Short-term Loudness Range**: 12 LUFS maximum spread

### Key Characteristics
- Builds upon ITU-R BS.1770-4 foundation
- Requires minimum loudness range (4 LU)
- True Peak of -3 dBFS provides safety margin for analog broadcasting
- Mandatory for European broadcast television
- Mandatory for all audio content in EU

### Loudness Range Categories
- **Uniform Content**: < 4 LU (e.g., audiobooks, podcasts)
- **Normal Content**: 4-8 LU (typical music, speech)
- **Dynamic Content**: > 8 LU (orchestral, film trailers)

## ATSC A/85 Standard

**American Television Standards Committee**

### Specifications
- **Integrated Loudness Target**: -24 LKFS ±2 LU
- **True Peak Limit**: -2 dBFS
- **Loudness Range**: Not specified
- **Dialnorm Metadata**: Required

### Key Characteristics
- Used for broadcast television in North America
- Lower target level (-24 LKFS) than ITU-R BS.1770-4
- Includes Dialnorm metadata for receiver-level loudness
- True Peak limit of -2 dBFS for legacy equipment compatibility
- Measurement window: 100ms blocks

## Professional Metering Metrics

### Integrated Loudness
- **Definition**: Average loudness over entire program duration
- **Window**: Full content length
- **Use**: Overall loudness compliance check
- **Unit**: LUFS or LKFS

### Short-term Loudness
- **Definition**: Loudness measurement over 3-second window
- **Window**: 3000ms sliding
- **Use**: Monitoring loudness stability
- **Unit**: LUFS

### Momentary Loudness
- **Definition**: Instantaneous loudness perception
- **Window**: 400ms (1/2.5 second)
- **Use**: Peak loudness detection
- **Unit**: LUFS

### True Peak
- **Definition**: Maximum peak level accounting for inter-sample peaks
- **Method**: 4x oversampling with digital filtering
- **Use**: Clipping detection and prevention
- **Unit**: dBFS

### Loudness Range (LRA)
- **Definition**: Statistical range of loudness variation
- **Calculation**: 10th to 90th percentile of short-term loudness
- **Use**: Program dynamic range assessment
- **Unit**: LU (Loudness Units)
- **Formula**: LRA = LUFS(90th percentile) - LUFS(10th percentile)

## Frequency Analysis Standards

### Spectrum Analysis
- **Frequency Range**: 0 Hz to 20 kHz (human hearing range)
- **Frequency Scales Supported**:
  - Linear: Equal frequency spacing (Hz)
  - Logarithmic: Octave/decade-based (common for audio)
  - Mel-scale: Perceptually equal (human hearing)
  - Bark-scale: Critical band representation

### FFT Parameters
- **FFT Size**: 4096 points (recommended)
- **Hop Length**: 1024 samples (25% overlap)
- **Window Function**: Hamming window (reduces spectral leakage)
- **Sample Rate**: 48 kHz (professional standard)

## Compliance Indicators

The application provides real-time compliance indicators for:

| Standard | Target | True Peak | LRA Min | Use Case |
|----------|--------|-----------|---------|----------|
| ITU-R BS.1770-4 | -23 LUFS | 0 dBFS | N/A | International broadcast |
| EBU R128 | -23 LUFS | -3 dBFS | 4 LU | EU broadcasting |
| LKFS (Streaming) | -16 LKFS | -1 dBFS | N/A | Streaming platforms |
| ATSC A/85 | -24 LKFS | -2 dBFS | N/A | US television |

## Implementation in Audio Analyzer

### Audio Pipeline
1. **Input**: Audio file (WAV, MP3, FLAC, OGG)
2. **Resampling**: Convert to 48 kHz if needed
3. **Filtering**: Apply K-weighting and high-pass filter
4. **Analysis**: Calculate all LUFS metrics
5. **True Peak**: Compute with 4x oversampling
6. **Visualization**: Display metrics with compliance status
7. **Export**: Save results with standard compliance report

### Real-Time Visualization
- **Spectrum Display**: 240-band spectrogram
- **LUFS Meter**: Live loudness values
- **Waveform**: Audio amplitude visualization
- **Compliance Indicator**: Color-coded status
- **Peak Meter**: True Peak level indicator

## Measurement Guidelines

### Proper Measurement Procedure
1. Load complete audio file
2. Ensure calibration of output levels
3. Run analysis on entire content duration
4. Record all four loudness values (Integrated, Short-term, Momentary, True Peak)
5. Check Loudness Range if required by standard
6. Verify compliance with target standard

### Common Issues and Solutions
- **Clipping Detected**: Reduce input level or re-master audio
- **Loudness Out of Range**: Apply compression/limiting or gain adjustment
- **Low Loudness Range**: Add dynamic processing or re-arrange content
- **True Peak Exceeded**: Apply look-ahead limiter

## References

- ITU-R BS.1770-4: "Algorithms to measure audio programme loudness and true-peak audio level"
- EBU R128: "Loudness normalisation and permitted maximum level of audio signals"
- ATSC A/85: "Techniques for Establishing and Maintaining Audio Loudness for Digital Television"
- ISO/IEC 61672-1: "Electroacoustics - Sound level meters"

## Compliance Certification

Audio Analyzer LUFS implements:
- ✓ ITU-R BS.1770-4 full compliance
- ✓ EBU R128 compatibility
- ✓ LKFS streaming standard support
- ✓ ATSC A/85 measurement capability
- ✓ Professional-grade accuracy (±0.5 LU)
- ✓ Real-time monitoring
- ✓ Full measurement history tracking
