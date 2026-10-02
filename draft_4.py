import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Setup Signal
sampling_rate = 8000
duration = 2.0
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
clean_signal = np.sin(2 * np.pi * 440 * t)
noise_hiss = 0.5 * np.sin(2 * np.pi * 3000 * t)
noise_hum = 0.3 * np.sin(2 * np.pi * 60 * t)
noisy_signal = clean_signal + noise_hiss + noise_hum

# 2. Filter Signal
fft_spectrum = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(noisy_signal), 1 / sampling_rate)
filtered_spectrum = fft_spectrum.copy()
filtered_spectrum[(np.abs(frequencies) < 200) | (np.abs(frequencies) > 1000)] = 0
cleaned_signal = np.real(np.fft.ifft(filtered_spectrum))

# 3. Setup Scrolling Animation Figure
fig, ax = plt.subplots(figsize=(10, 5))
line_noisy, = ax.plot([], [], color='purple', alpha=0.5, label='Noisy Signal')
line_clean, = ax.plot([], [], color='teal', lw=2, label='Filtered Clean Signal')

ax.set_xlim(0, 0.05)  # Window size of 50 ms viewable at a time
ax.set_ylim(-1.8, 1.8)
ax.set_title('Running Audio Waveform (Oscilloscope View)', fontsize=12, fontweight='bold')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

window_size = int(sampling_rate * 0.05) # 50 ms window samples
total_frames = len(t) - window_size

def update(frame):
    current_t = t[frame:frame + window_size]
    ax.set_xlim(current_t[0], current_t[-1])
    line_noisy.set_data(current_t, noisy_signal[frame:frame + window_size])
    line_clean.set_data(current_t, cleaned_signal[frame:frame + window_size])
    return line_noisy, line_clean

# Create the animation (runs frame by frame)
ani = FuncAnimation(fig, update, frames=range(0, total_frames, 20), interval=20, blit=True)

plt.show()