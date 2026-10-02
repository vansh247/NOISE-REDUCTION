"""
FFT Noise Removal
=================
Removes noise from an audio signal using the Fourier Transform:

    noisy signal -> FFT -> remove unwanted frequencies -> inverse FFT -> clean signal

By default it builds a synthetic demo (440 Hz tone + 60 Hz hum + 3000 Hz hiss).
Set AUDIO_FILE to a .wav path to clean your own recording instead.
The cleaned audio is saved as a WAV file and played when you close the plots.

Requires: numpy, scipy, matplotlib   (optional: sounddevice, for playback)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile

# ======================================================
# SETTINGS - edit these
# ======================================================
AUDIO_FILE = None              # e.g. "my_recording.wav"  (None = synthetic demo)
OUTPUT_FILE = "cleaned_audio.wav"
PLAY_AFTER = True              # play the cleaned audio after the plots are closed

KEEP_LOW_HZ = 200              # frequencies below this are removed (hum)
KEEP_HIGH_HZ = 1000            # frequencies above this are removed (hiss)
# Tip: for speech try 80 -> 4000 Hz. The 200-1000 Hz band suits the demo tone.

DEMO_SAMPLING_RATE = 8000      # Hz
DEMO_DURATION = 2.0            # seconds


# ======================================================
# 1. GET A NOISY SIGNAL
# ======================================================
def make_demo_signal():
    """Clean 440 Hz tone + 60 Hz hum + 3000 Hz hiss."""
    sr = DEMO_SAMPLING_RATE
    t = np.linspace(0, DEMO_DURATION, int(sr * DEMO_DURATION), endpoint=False)

    clean = np.sin(2 * np.pi * 440 * t)           # musical note 'A'
    hum = 0.3 * np.sin(2 * np.pi * 60 * t)        # low-frequency hum
    hiss = 0.5 * np.sin(2 * np.pi * 3000 * t)     # high-frequency hiss
    noisy = clean + hum + hiss

    waves = [  # (title, data, colour) - one plot panel each
        ("Original Clean Signal (440 Hz)", clean, "dodgerblue"),
        ("Low-Frequency Hum Component (60 Hz)", hum, "orange"),
        ("High-Frequency Hiss Component (3000 Hz)", hiss, "crimson"),
        ("Combined Noisy Audio Signal", noisy, "purple"),
    ]
    return sr, t, noisy, waves


def load_audio_file(path):
    """Read a .wav file as a mono float signal in the range about -1 to 1."""
    sr, data = wavfile.read(path)
    if data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128) / 128
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)                   # stereo -> mono

    t = np.arange(len(data)) / sr
    waves = [("Original (Noisy) Signal", data, "purple")]
    return sr, t, data, waves


# ======================================================
# 2. FFT -> FILTER -> INVERSE FFT
# ======================================================
def remove_noise(signal, sr, low_hz, high_hz):
    """Zero every frequency outside [low_hz, high_hz] and rebuild the signal."""
    spectrum = np.fft.rfft(signal)                          # forward FFT
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)            # frequency of each bin

    filtered = spectrum.copy()
    filtered[(freqs < low_hz) | (freqs > high_hz)] = 0      # spectral mask

    cleaned = np.fft.irfft(filtered, n=len(signal))         # inverse FFT
    return freqs, spectrum, cleaned


# ======================================================
# 3. SAVE AND PLAY
# ======================================================
def normalize(audio):
    """Scale to 95% of full volume so it never clips."""
    peak = np.max(np.abs(audio))
    return audio / peak * 0.95 if peak > 0 else audio


def save_audio(path, audio, sr):
    wavfile.write(path, sr, (normalize(audio) * 32767).astype(np.int16))


def play_audio(audio, sr):
    try:
        import sounddevice as sd
    except ImportError:
        print("To play audio here: pip install sounddevice (or open the saved WAV file).")
        return
    print("Playing cleaned audio...")
    sd.play(normalize(audio), sr)
    sd.wait()


# ======================================================
# 4. VISUALIZATION: STATIC PANELS + ANIMATED WAVE
# ======================================================
def plot_dashboard(t, sr, waves, freqs, spectrum, noisy, cleaned):
    n_rows = len(waves) + 2                                  # waves + spectrum + animation
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.7 * n_rows), constrained_layout=True)
    fig.suptitle("FFT Noise Removal", fontsize=14, fontweight="bold")

    # --- Waveform panels (first 400 samples) ---
    view = slice(0, 400)
    for i, (ax, (title, data, color)) in enumerate(zip(axes, waves), start=1):
        ax.plot(t[view], data[view], color=color, lw=1.5)
        ax.set_title(f"{i}. {title}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    # --- Frequency spectrum with the band we keep ---
    ax_fft = axes[len(waves)]
    ax_fft.plot(freqs, np.abs(spectrum), color="indigo", lw=1.5)
    ax_fft.axvspan(KEEP_LOW_HZ, KEEP_HIGH_HZ, color="teal", alpha=0.15,
                   label=f"Kept band ({KEEP_LOW_HZ}-{KEEP_HIGH_HZ} Hz)")
    ax_fft.set_xlim(0, min(sr / 2, KEEP_HIGH_HZ * 4))
    ax_fft.set_title(f"{len(waves) + 1}. Frequency Spectrum (FFT Magnitude)",
                     fontsize=10, fontweight="bold")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.legend(loc="upper right", fontsize=9)
    ax_fft.grid(True, alpha=0.3)

    # --- Animated scrolling oscilloscope: noisy vs filtered ---
    ax = axes[-1]
    window = min(int(sr * 0.05), len(noisy) - 1)             # 50 ms visible at a time
    step = max(1, sr // 400)                                 # samples advanced per frame
    limit = 1.1 * np.max(np.abs(noisy))

    line_noisy, = ax.plot([], [], color="purple", alpha=0.5, label="Noisy Signal")
    line_clean, = ax.plot([], [], color="teal", lw=2, label="Filtered Clean Signal")
    ax.set_ylim(-limit, limit)
    ax.set_title(f"{len(waves) + 2}. ANIMATED: Running Wave Comparison (Noisy vs Filtered)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    def update(start):
        seg = slice(start, start + window)
        ax.set_xlim(t[start], t[start + window - 1])
        line_noisy.set_data(t[seg], noisy[seg])
        line_clean.set_data(t[seg], cleaned[seg])
        return line_noisy, line_clean

    # blit=False so the time axis labels scroll along with the wave
    return FuncAnimation(fig, update, frames=range(0, len(noisy) - window, step),
                         interval=30, blit=False, cache_frame_data=False)


# ======================================================
# MAIN
# ======================================================
def main():
    sr, t, noisy, waves = load_audio_file(AUDIO_FILE) if AUDIO_FILE else make_demo_signal()
    print(f"Signal ready: {len(noisy) / sr:.1f} s at {sr} Hz")

    freqs, spectrum, cleaned = remove_noise(noisy, sr, KEEP_LOW_HZ, KEEP_HIGH_HZ)
    save_audio(OUTPUT_FILE, cleaned, sr)
    print(f"Cleaned audio saved to: {OUTPUT_FILE}")

    ani = plot_dashboard(t, sr, waves, freqs, spectrum, noisy, cleaned)  # keep reference!
    plt.show()

    if PLAY_AFTER:
        play_audio(cleaned, sr)


if __name__ == "__main__":
    main()
