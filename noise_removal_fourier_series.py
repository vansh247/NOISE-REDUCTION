import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 1. SETUP A SYNTHETIC NOISY AUDIO SIGNAL
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
# 2. FORWARD FOURIER TRANSFORM (FFT)
fft_spectrum = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(noisy_signal), 1 / sampling_rate)
# 3. APPLY SPECTRAL FILTERING (MASKING)
filtered_spectrum = fft_spectrum.copy()
cutoff_low = 200   # Hz
cutoff_high = 1000 # Hz
filtered_spectrum[(np.abs(frequencies) < cutoff_low) | (np.abs(frequencies) > cutoff_high)] = 0
# 4. INVERSE FOURIER TRANSFORM (IFFT)
cleaned_signal = np.real(np.fft.ifft(filtered_spectrum))
# 5. VISUALIZATION: COMBINED STATIC + ANIMATED DASHBOARD
fig, axes = plt.subplots(6, 1, figsize=(12, 16))
plt.subplots_adjust(hspace=0.6)  # Generous vertical space to avoid crowding
# --- Panel 1: Clean Signal ---
axes[0].plot(t[:400], clean_signal[:400], color='dodgerblue', lw=1.5)
axes[0].set_title('1. Original Clean Signal (440 Hz)', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
# --- Panel 2: Hum Noise ---
axes[1].plot(t[:400], noise_hum[:400], color='orange', lw=1.5)
axes[1].set_title('2. Low-Frequency Hum Component (60 Hz)', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)
# --- Panel 3: Hiss Noise ---
axes[2].plot(t[:400], noise_hiss[:400], color='crimson', lw=1.5)
axes[2].set_title('3. High-Frequency Hiss Component (3000 Hz)', fontsize=10, fontweight='bold')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)
# --- Panel 4: Combined Noisy Signal ---
axes[3].plot(t[:400], noisy_signal[:400], color='purple', lw=1.5)
axes[3].set_title('4. Combined Noisy Audio Signal', fontsize=10, fontweight='bold')
axes[3].set_ylabel('Amplitude')
axes[3].grid(True, alpha=0.3)
# --- Panel 5: Frequency Domain Spectrum ---
axes[4].plot(frequencies[:len(frequencies)//2], np.abs(fft_spectrum)[:len(fft_spectrum)//2], color='indigo', lw=1.5)
axes[4].set_title('5. Frequency Domain Spectrum (FFT Magnitude)', fontsize=10, fontweight='bold')
axes[4].set_xlabel('Frequency (Hz)')
axes[4].set_ylabel('Magnitude')
axes[4].grid(True, alpha=0.3)
# --- Panel 6: Animated Scrolling Oscilloscope (Live Running Wave) ---
ax_anim = axes[5]
line_noisy, = ax_anim.plot([], [], color='purple', alpha=0.5, label='Noisy Signal')
line_clean, = ax_anim.plot([], [], color='teal', lw=2, label='Filtered Clean Signal')
ax_anim.set_xlim(0, 0.05)  # 50 ms view window
ax_anim.set_ylim(-1.8, 1.8)
ax_anim.set_title('6. ANIMATED: Running Wave Comparison (Noisy vs Filtered)', fontsize=10, fontweight='bold')
ax_anim.set_xlabel('Time (s)')
ax_anim.set_ylabel('Amplitude')
ax_anim.legend(loc='upper right', fontsize=9)
ax_anim.grid(True, alpha=0.3)
window_size = int(sampling_rate * 0.05) # 50 ms samples per frame
total_frames = len(t) - window_size
def update(frame):
    current_t = t[frame:frame + window_size]
    ax_anim.set_xlim(current_t[0], current_t[-1])
    line_noisy.set_data(current_t, noisy_signal[frame:frame + window_size])
    line_clean.set_data(current_t, cleaned_signal[frame:frame + window_size])
    return line_noisy, line_clean
# Run the animation across the bottom panel
ani = FuncAnimation(fig, update, frames=range(0, total_frames, 20), interval=20, blit=True)
plt.show()