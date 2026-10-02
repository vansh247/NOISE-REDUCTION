import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import librosa  # Used for loading MP3, WAV, etc.
#1.LOAD MP3 (OR WAV) AUDIO FILE
audio_path = 'E:\\music\\chammak-challo_endMGL6Q.mp3'  #(Replace this path with the path to your audio file)
try:
    # sr=None preserves the original sampling rate; mono=True converts stereo to mono
    audio_data, sampling_rate = librosa.load(audio_path, sr=None, mono=True)
except Exception as e:
    print(f"Error loading file: {e}")
    print("Falling back to a synthetic tone so the script runs.")
    sampling_rate = 8000
    duration = 2.0
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sampling_rate * duration)))
duration = len(audio_data) / sampling_rate
t = np.linspace(0, duration, len(audio_data), endpoint=False)
#Librosa loads audio as float32 normalized between -1.0 and 1.0
noisy_signal = audio_data
print(f"Audio loaded successfully! Duration: {duration:.2f}s, Sampling Rate: {sampling_rate} Hz")
#2.FORWARD FOURIER TRANSFORM
fft_spectrum = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(noisy_signal), 1 / sampling_rate)
#3.APPLY SPECTRAL FILTERING
filtered_spectrum = fft_spectrum.copy()
cutoff_low = 200 # Hz (frequencies below this are muted) (u can adjust this value to remove low frequency noise) 
cutoff_high = 8000 # Hz (frequencies above this are muted)(u can adjust this value to remove high frequency noise) 
filtered_spectrum[(np.abs(frequencies) < cutoff_low) | (np.abs(frequencies) > cutoff_high)] = 0
#4.INVERSE FOURIER TRANSFORM
cleaned_signal = np.real(np.fft.ifft(filtered_spectrum))
#AUDIO PLAYBACK
print("Playing Original Audio...")
sd.play(noisy_signal, sampling_rate)
sd.wait()
print("Playing Filtered Audio...")
sd.play(cleaned_signal, sampling_rate)
sd.wait()
# 5.VISUALIZATION DASHBOARD
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)
# panel 1: original audio waveform
view_samples = min(int(sampling_rate * 0.5), len(t))
axes[0].plot(t[:view_samples], noisy_signal[:view_samples], color='purple', lw=1.5)
axes[0].set_title('1. Original Audio Waveform', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)
#Panel 2: frequency domain spectrum
axes[1].plot(frequencies[:len(frequencies)//2], np.abs(fft_spectrum)[:len(fft_spectrum)//2], color='indigo', lw=1.5)
axes[1].set_title('2. Frequency Domain Spectrum (FFT Magnitude)', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].grid(True, alpha=0.3)
#panel3:animated scrolling oscilloscope ---
ax_anim = axes[2]
line_noisy, = ax_anim.plot([], [], color='purple', alpha=0.5, label='Original Audio')
line_clean, = ax_anim.plot([], [], color='teal', lw=2, label='Filtered Audio')
ax_anim.set_xlim(0, 0.05)
ax_anim.set_ylim(-1.1, 1.1)
ax_anim.set_title('3. ANIMATED: Running Wave Comparison', fontsize=10, fontweight='bold')
ax_anim.set_xlabel('Time (s)')
ax_anim.set_ylabel('Amplitude')
ax_anim.legend(loc='upper right', fontsize=9)
ax_anim.grid(True, alpha=0.3)
window_size = int(sampling_rate * 0.05) # 50 ms window
total_frames = max(1, len(t) - window_size)
def update(frame):
    current_t = t[frame:frame + window_size]
    ax_anim.set_xlim(current_t[0], current_t[-1])
    line_noisy.set_data(current_t, noisy_signal[frame:frame + window_size])
    line_clean.set_data(current_t, cleaned_signal[frame:frame + window_size])
    return line_noisy, line_clean
ani = FuncAnimation(fig, update, frames=range(0, total_frames, 20), interval=20, blit=True)
plt.show()