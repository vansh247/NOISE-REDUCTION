import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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

print("Synthetic audio signal generated successfully.")

# ==========================================
# 2. FORWARD FOURIER TRANSFORM (FFT)
# ==========================================
# Convert time-domain signal to the frequency domain
fft_spectrum = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(noisy_signal), 1 / sampling_rate)

# ==========================================
# 3. APPLY SPECTRAL FILTERING (MASKING)
# ==========================================
# We want to keep frequencies between 200 Hz and 1000 Hz (protecting our 440 Hz tone)
# and filter out everything else (removing the 60 Hz hum and 3000 Hz hiss).
filtered_spectrum = fft_spectrum.copy()

# Zero out frequencies outside our target band
cutoff_low = 200  # Hz
cutoff_high = 1000 # Hz

# Apply mask by zeroing out unwanted frequency bins
filtered_spectrum[(np.abs(frequencies) < cutoff_low) | (np.abs(frequencies) > cutoff_high)] = 0

# ==========================================
# 4. INVERSE FOURIER TRANSFORM (IFFT)
# ==========================================
# Convert the filtered frequency spectrum back to the time domain
# We use np.real because numerical inaccuracies can leave tiny imaginary residues
cleaned_signal = np.real(np.fft.ifft(filtered_spectrum))

# ==========================================
# 5. VISUALIZE RESULTS
# ==========================================
plt.figure(figsize=(12, 10))

# Plot Time Domain: Noisy vs Cleaned
plt.subplot(3, 1, 1)
plt.plot(t[:500], noisy_signal[:500], color='salmon', alpha=0.7, label='Noisy Audio Signal')
plt.plot(t[:500], clean_signal[:500], color='gray', linestyle='--', label='Original Clean Signal')
plt.title('Time Domain: Noisy Signal Sample')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot Frequency Domain Spectrum
plt.subplot(3, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_spectrum)[:len(fft_spectrum)//2], color='purple')
plt.title('Frequency Domain Spectrum (FFT Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot Time Domain: Filtered Output
plt.subplot(3, 1, 3)
plt.plot(t[:500], cleaned_signal[:500], color='teal', label='Filtered Audio Signal (Post-IFFT)')
plt.title('Time Domain: Cleaned Signal After Spectral Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: Save the filtered signal as a playable WAV file
# scaled = np.int16(cleaned_signal / np.max(np.abs(cleaned_signal)) * 32767)
# wavfile.write('filtered_output.wav', sampling_rate, scaled)