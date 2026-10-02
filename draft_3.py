import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP A SYNTHETIC NOISY AUDIO SIGNAL
# ==========================================
sampling_rate = 8000  # 8 kHz sampling rate
duration = 2.0        # 2 seconds of audio
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Original signal: A clean 440 Hz tone (Musical note 'A')
clean_signal = np.sin(2 * np.pi * 440 * t)

# Add unwanted noise: A high-frequency hiss (3000 Hz) and a low-frequency hum (60 Hz)
noise_hiss = 0.5 * np.sin(2 * np.pi * 3000 * t)
noise_hum = 0.3 * np.sin(2 * np.pi * 60 * t)
noisy_signal = clean_signal + noise_hiss + noise_hum

# ==========================================
# 2. FORWARD FOURIER TRANSFORM (FFT)
# ==========================================
fft_spectrum = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(noisy_signal), 1 / sampling_rate)

# ==========================================
# 3. APPLY SPECTRAL FILTERING (MASKING)
# ==========================================
filtered_spectrum = fft_spectrum.copy()
cutoff_low = 200   # Hz
cutoff_high = 1000 # Hz
filtered_spectrum[(np.abs(frequencies) < cutoff_low) | (np.abs(frequencies) > cutoff_high)] = 0

# ==========================================
# 4. INVERSE FOURIER TRANSFORM (IFFT)
# ==========================================
cleaned_signal = np.real(np.fft.ifft(filtered_spectrum))

# ==========================================
# 5. VISUALIZE RESULTS WITH SPACED LAYOUT
# ==========================================
fig, axes = plt.subplots(6, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.6)  # Increases vertical space between subplots

# 1. Clean Signal
axes[0].plot(t[:400], clean_signal[:400], color='dodgerblue', lw=1.5)
axes[0].set_title('1. Original Clean Signal (440 Hz)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# 2. Hum Noise
axes[1].plot(t[:400], noise_hum[:400], color='orange', lw=1.5)
axes[1].set_title('2. Low-Frequency Hum Component (60 Hz)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

# 3. Hiss Noise
axes[2].plot(t[:400], noise_hiss[:400], color='crimson', lw=1.5)
axes[2].set_title('3. High-Frequency Hiss Component (3000 Hz)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

# 4. Noisy Signal
axes[3].plot(t[:400], noisy_signal[:400], color='purple', lw=1.5)
axes[3].set_title('4. Combined Noisy Audio Signal', fontsize=11, fontweight='bold')
axes[3].set_ylabel('Amplitude')
axes[3].grid(True, alpha=0.3)

# 5. Frequency Domain Spectrum
axes[4].plot(frequencies[:len(frequencies)//2], np.abs(fft_spectrum)[:len(fft_spectrum)//2], color='indigo', lw=1.5)
axes[4].set_title('5. Frequency Domain Spectrum (FFT Magnitude)', fontsize=11, fontweight='bold')
axes[4].set_xlabel('Frequency (Hz)')
axes[4].set_ylabel('Magnitude')
axes[4].grid(True, alpha=0.3)

# 6. Filtered Output
axes[5].plot(t[:400], cleaned_signal[:400], color='teal', lw=1.5)
axes[5].set_title('6. Cleaned Signal After Spectral Filtering (Post-IFFT)', fontsize=11, fontweight='bold')
axes[5].set_xlabel('Time (s)')
axes[5].set_ylabel('Amplitude')
axes[5].grid(True, alpha=0.3)

plt.show()