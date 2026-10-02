
import math
import random
import time

import numpy as np
import pygame

# ============================================================
# JARVIS AUDIO LAB | CINEMATIC SIGNAL PROCESSING
# Pygame + NumPy | Resizable scientific visualization
# ============================================================

# --------------------------- CONFIG --------------------------
FS = 8000
DURATION = 2.0
N = int(FS * DURATION)

LOW_CUTOFF = 200
HIGH_CUTOFF = 1000

FFT_SIZE = 4096
FFT_HOP = 160
WAVE_SECONDS = 0.05

FPS = 60

BG = (4, 10, 22)
PANEL = (8, 20, 37)
PANEL_ALT = (10, 27, 46)
CYAN = (43, 224, 255)
TEAL = (53, 255, 190)
BLUE = (56, 130, 255)
PURPLE = (155, 100, 255)
ORANGE = (255, 174, 75)
RED = (255, 79, 112)
WHITE = (226, 246, 255)
MUTED = (107, 151, 180)
GRID = (18, 53, 77)

# ---------------------- AUDIO GENERATION ---------------------
t = np.arange(N, dtype=np.float64) / FS

clean = np.sin(2 * np.pi * 440 * t)
hum = 0.3 * np.sin(2 * np.pi * 60 * t)
hiss = 0.5 * np.sin(2 * np.pi * 3000 * t)

noisy = clean + hum + hiss

# ---------------------- FFT FILTERING ------------------------
freq = np.fft.rfftfreq(N, d=1 / FS)
spectrum = np.fft.rfft(noisy)

mask = (freq >= LOW_CUTOFF) & (freq <= HIGH_CUTOFF)
filtered_spectrum = spectrum * mask
filtered = np.fft.irfft(filtered_spectrum, n=N)

# ---------------------- SIGNAL METRICS -----------------------
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

input_rms = rms(noisy)
output_rms = rms(filtered)

# The synthetic demo has known noise frequencies.
# Measure their sinusoidal amplitudes by direct projection.
def component_amplitude(signal, frequency):
    sin_ref = np.sin(2 * np.pi * frequency * t)
    cos_ref = np.cos(2 * np.pi * frequency * t)

    a = 2.0 * np.dot(signal, sin_ref) / N
    b = 2.0 * np.dot(signal, cos_ref) / N

    return float(np.hypot(a, b))

noise_before = np.hypot(
    component_amplitude(noisy, 60),
    component_amplitude(noisy, 3000)
)

noise_after = np.hypot(
    component_amplitude(filtered, 60),
    component_amplitude(filtered, 3000)
)

# Avoid displaying meaningless finite dB values when the
# remaining energy is at numerical precision.
if noise_after < noise_before * 1e-8:
    reduction_percent = 100.0
    attenuation_db = None
else:
    reduction_percent = 100 * (
        1 - noise_after / max(noise_before, 1e-15)
    )
    attenuation_db = 20 * np.log10(
        noise_before / max(noise_after, 1e-15)
    )

# ------------------- PRECOMPUTED FFT FRAMES ------------------
window = np.hanning(FFT_SIZE)
window_sum = window.sum()
fft_freq = np.fft.rfftfreq(FFT_SIZE, d=1 / FS)

starts = np.arange(0, N, FFT_HOP)
input_spectra = []
output_spectra = []

def get_frame(signal, start):
    segment = signal[start:start + FFT_SIZE]

    if len(segment) < FFT_SIZE:
        segment = np.pad(segment, (0, FFT_SIZE - len(segment)))

    return segment

def frame_spectrum(segment):
    values = np.abs(np.fft.rfft(segment * window))
    values *= 2.0 / window_sum
    values[0] *= 0.5
    return values

for start in starts:
    input_spectra.append(
        frame_spectrum(get_frame(noisy, start))
    )
    output_spectra.append(
        frame_spectrum(get_frame(filtered, start))
    )

input_spectra = np.asarray(input_spectra)
output_spectra = np.asarray(output_spectra)

# Focus on the frequency range used by the dashboard.
MAX_FREQ = 4000
visible = fft_freq <= MAX_FREQ

plot_freq = fft_freq[visible]
plot_in = input_spectra[:, visible]
plot_out = output_spectra[:, visible]

# Log-spaced frequency bar locations, plus the three target tones.
bar_freqs = np.unique(np.concatenate([
    np.geomspace(25, MAX_FREQ, 44),
    np.array([60, 440, 3000])
]))

bar_bins = np.array([
    np.argmin(np.abs(fft_freq - f)) for f in bar_freqs
])

# Normalize visualization levels using all precomputed frames.
visual_max = max(float(input_spectra.max()), 1e-6)

# -------------------------- PYGAME ---------------------------
pygame.init()
pygame.display.set_caption("JARVIS | Audio Intelligence Lab")

WIDTH, HEIGHT = 1440, 900
screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    pygame.RESIZABLE
)
clock = pygame.time.Clock()

FONT_CACHE = {}

def font(size, bold=False):
    key = (size, bold)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = pygame.font.SysFont(
            "consolas", size, bold=bold
        )
    return FONT_CACHE[key]

def text(surface, value, x, y, size=16, color=WHITE,
         bold=False, anchor="topleft"):
    image = font(size, bold).render(str(value), True, color)
    rect = image.get_rect()
    setattr(rect, anchor, (int(x), int(y)))
    surface.blit(image, rect)
    return rect

def panel(surface, rect, title, tag=None):
    x, y, w, h = rect

    pygame.draw.rect(
        surface, PANEL, rect, border_radius=12
    )
    pygame.draw.rect(
        surface, GRID, rect, width=1, border_radius=12
    )

    # Accent line beneath panel heading
    pygame.draw.line(
        surface, CYAN, (x + 16, y + 42),
        (x + 72, y + 42), 2
    )

    text(surface, title, x + 16, y + 13, 15, WHITE, True)

    if tag:
        text(
            surface, tag, x + w - 14, y + 16,
            10, MUTED, anchor="topright"
        )

def glow_line(surface, points, color, width=2):
    if len(points) < 2:
        return

    # Layered strokes create a lightweight glow illusion.
    for extra, factor in ((8, 0.12), (5, 0.22), (3, 0.40)):
        glow = tuple(int(c * factor) for c in color)
        pygame.draw.lines(
            surface, glow, False, points, width + extra
        )

    pygame.draw.aalines(surface, color, False, points)

def grid_panel(surface, rect, columns=8, rows=4):
    x, y, w, h = rect

    for i in range(1, columns):
        gx = x + int(w * i / columns)
        pygame.draw.line(
            surface, GRID, (gx, y), (gx, y + h), 1
        )

    for i in range(1, rows):
        gy = y + int(h * i / rows)
        pygame.draw.line(
            surface, GRID, (x, gy), (x + w, gy), 1
        )

def draw_wave(surface, rect, signal, current, color,
              label, gain=1.0):
    x, y, w, h = rect
    grid_panel(surface, rect)

    count = max(2, int(w))
    indices = (
        current + np.linspace(0, WAVE_SECONDS * FS, count)
    ).astype(int) % N

    values = signal[indices]
    mid = y + h // 2

    points = [
        (
            x + i,
            int(mid - float(v) * h * 0.38 * gain)
        )
        for i, v in enumerate(values)
    ]

    glow_line(surface, points, color, 2)
    text(surface, label, x + 8, y + 6, 11, color, True)

def draw_spectrum(surface, rect, frame):
    x, y, w, h = rect
    grid_panel(surface, rect, 10, 4)

    inside_x = x + 2
    inside_y = y + 8
    inside_w = w - 4
    inside_h = h - 20

    a = plot_in[frame]
    b = plot_out[frame]

    # Downsample only for drawing, not for signal processing.
    indices = np.linspace(
        0, len(plot_freq) - 1,
        min(360, len(plot_freq))
    ).astype(int)

    points_in = []
    points_out = []

    for j, idx in enumerate(indices):
        px = inside_x + int(
            j * inside_w / max(1, len(indices) - 1)
        )

        vin = min(float(a[idx]) / visual_max, 1.0)
        vout = min(float(b[idx]) / visual_max, 1.0)

        py_in = inside_y + inside_h - int(vin * inside_h)
        py_out = inside_y + inside_h - int(vout * inside_h)

        points_in.append((px, py_in))
        points_out.append((px, py_out))

    glow_line(surface, points_in, PURPLE, 1)
    glow_line(surface, points_out, CYAN, 2)

    # Highlight known frequency locations.
    for f, color, name in (
        (60, ORANGE, "60"),
        (440, TEAL, "440"),
        (3000, RED, "3k")
    ):
        px = x + int(f / MAX_FREQ * w)
        pygame.draw.line(
            surface, color, (px, y + 4), (px, y + h - 8), 1
        )
        text(
            surface, name, px + 4, y + 4, 10, color
        )

def draw_frequency_bars(surface, rect, frame):
    x, y, w, h = rect
    grid_panel(surface, rect, 8, 3)

    values_in = input_spectra[frame, bar_bins]
    values_out = output_spectra[frame, bar_bins]

    gap = 3
    bar_w = max(2, (w - gap * len(bar_freqs)) // len(bar_freqs))

    for i, f in enumerate(bar_freqs):
        bx = x + i * (bar_w + gap)

        vin = min(float(values_in[i]) / visual_max, 1.0)
        vout = min(float(values_out[i]) / visual_max, 1.0)

        hin = int(vin * (h - 8))
        hout = int(vout * (h - 8))

        pygame.draw.rect(
            surface, (67, 42, 105),
            (bx, y + h - hin, bar_w, hin)
        )
        pygame.draw.rect(
            surface, CYAN,
            (bx, y + h - hout, bar_w, hout)
        )

        # Mark the three frequencies of interest.
        if min(abs(f - target) for target in (60, 440, 3000)) < 0.1:
            color = (
                ORANGE if abs(f - 60) < 1
                else TEAL if abs(f - 440) < 1
                else RED
            )
            pygame.draw.line(
                surface, color,
                (bx + bar_w // 2, y + 1),
                (bx + bar_w // 2, y + h), 2
            )

    text(
        surface, "INPUT", x + 5, y + 5,
        9, PURPLE, True
    )
    text(
        surface, "FILTERED", x + 65, y + 5,
        9, CYAN, True
    )

def draw_gauge(surface, center, radius, progress):
    cx, cy = center

    # Background rings
    for r, color, width in (
        (radius + 12, GRID, 1),
        (radius + 5, (20, 67, 89), 2),
        (radius - 4, GRID, 1)
    ):
        pygame.draw.circle(surface, color, center, r, width)

    gauge_rect = pygame.Rect(
        cx - radius, cy - radius,
        radius * 2, radius * 2
    )

    pygame.draw.arc(
        surface, (25, 55, 74), gauge_rect,
        math.radians(135), math.radians(405), 8
    )

    # Pygame arcs start at the right and proceed counterclockwise.
    # Convert the desired clockwise sweep into Pygame's angles.
    start_angle = math.radians(225)
    end_angle = start_angle - math.radians(270 * progress)

    if progress > 0.001:
        pygame.draw.arc(
            surface, CYAN, gauge_rect,
            end_angle, start_angle, 8
        )

        # Bright tip at the moving progress endpoint.
        angle = math.radians(225 - 270 * progress)
        tx = int(cx + radius * math.cos(angle))
        ty = int(cy - radius * math.sin(angle))
        pygame.draw.circle(surface, TEAL, (tx, ty), 5)
        pygame.draw.circle(surface, WHITE, (tx, ty), 2)

    text(
        surface, f"{progress * 100:05.1f}%",
        cx, cy - 8, 31, TEAL, True, "center"
    )
    text(
        surface, "NOISE REDUCTION",
        cx, cy + 24, 10, MUTED, True, "center"
    )

def draw_particles(surface, particles, dt):
    for p in particles:
        p[0] += p[2] * dt
        p[1] += p[3] * dt

        if p[1] < 0 or p[1] > surface.get_height():
            p[1] = random.uniform(0, surface.get_height())
            p[0] = random.uniform(0, surface.get_width())

        radius = int(p[4])
        pygame.draw.circle(
            surface, (22, 92, 119),
            (int(p[0]), int(p[1])), radius
        )

# -------------------------- MAIN LOOP ------------------------
particles = [
    [
        random.uniform(0, WIDTH),
        random.uniform(0, HEIGHT),
        random.uniform(-5, 5),
        random.uniform(-15, -3),
        random.choice([1, 1, 2])
    ]
    for _ in range(85)
]

running = True
elapsed = 0.0
gauge_display = 0.0
frame_index = 0

while running:
    dt = min(clock.tick(FPS) / 1000.0, 0.05)
    elapsed += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.VIDEORESIZE:
            WIDTH = max(900, event.w)
            HEIGHT = max(650, event.h)
            screen = pygame.display.set_mode(
                (WIDTH, HEIGHT), pygame.RESIZABLE
            )

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    W, H = screen.get_size()

    # Smoothly cycle through precomputed audio frames.
    frame_index = int(elapsed * FS / FFT_HOP) % len(starts)
    current_sample = int(elapsed * FS) % N

    # Smooth gauge animation.
    gauge_target = reduction_percent / 100.0
    gauge_display += (
        gauge_target - gauge_display
    ) * min(1.0, dt * 2.5)

    screen.fill(BG)

    # Background particle field
    draw_particles(screen, particles, dt)

    # Decorative scanning line
    scan_y = int((elapsed * 65) % H)
    pygame.draw.line(
        screen, (8, 32, 48),
        (0, scan_y), (W, scan_y), 1
    )

    # ----------------------- HEADER --------------------------
    text(
        screen, "J.A.R.V.I.S.",
        30, 22, 28, CYAN, True
    )
    text(
        screen, "AUDIO INTELLIGENCE SYSTEM",
        32, 58, 11, WHITE, True
    )
    text(
        screen, "FFT / SPECTRAL FILTER / IFFT",
        32, 80, 10, MUTED
    )

    # Animated system indicator
    pulse = int(120 + 100 * (0.5 + 0.5 * math.sin(elapsed * 3)))
    status_color = (0, min(255, pulse), 170)

    pygame.draw.circle(screen, status_color, (W - 230, 38), 5)
    text(
        screen, "SYSTEM ONLINE",
        W - 215, 29, 12, TEAL, True
    )
    text(
        screen, f"FPS {clock.get_fps():.0f}",
        W - 30, 58, 11, MUTED, anchor="topright"
    )

    pygame.draw.line(
        screen, GRID, (28, 110), (W - 28, 110), 1
    )

    # ----------------------- LAYOUT --------------------------
    margin = 28
    gap = 18
    header = 128
    footer = 44

    content_h = H - header - footer
    left_w = int((W - 2 * margin - gap) * 0.56)
    right_w = W - 2 * margin - gap - left_w

    left_x = margin
    right_x = margin + left_w + gap
    top_y = header
    top_h = int((content_h - gap) * 0.48)
    bottom_y = top_y + top_h + gap
    bottom_h = H - footer - bottom_y

    wave_rect = (
        left_x, top_y, left_w, top_h
    )
    fft_rect = (
        right_x, top_y, right_w, top_h
    )
    bars_rect = (
        left_x, bottom_y, left_w, bottom_h
    )
    gauge_rect_panel = (
        right_x, bottom_y, right_w, bottom_h
    )

    panel(screen, wave_rect, "LIVE WAVEFORM", "INPUT / OUTPUT")
    panel(screen, fft_rect, "FREQUENCY ANALYZER", "0-4 kHz")
    panel(screen, bars_rect, "FREQUENCY COMPONENTS", "FFT BINS")
    panel(
        screen, gauge_rect_panel,
        "SIGNAL RESTORATION", "QUALITY MONITOR"
    )

    # Waveforms
    inner_wave_h = max(35, (top_h - 65) // 2)
    wave_x = left_x + 16
    wave_w = left_w - 32

    draw_wave(
        screen,
        (wave_x, top_y + 52, wave_w, inner_wave_h),
        noisy, current_sample, PURPLE,
        "NOISY INPUT"
    )

    draw_wave(
        screen,
        (
            wave_x, top_y + 60 + inner_wave_h,
            wave_w, inner_wave_h
        ),
        filtered, current_sample, TEAL,
        "FILTERED OUTPUT"
    )

    # Spectrum
    draw_spectrum(
        screen,
        (
            right_x + 16, top_y + 55,
            right_w - 32, top_h - 75
        ),
        frame_index
    )

    # Frequency bars
    draw_frequency_bars(
        screen,
        (
            left_x + 16, bottom_y + 55,
            left_w - 32, max(40, bottom_h - 88)
        ),
        frame_index
    )

    # Gauge panel
    gauge_cx = right_x + right_w // 2
    gauge_cy = bottom_y + int(bottom_h * 0.43)
    gauge_radius = max(
        45, min(95, int(bottom_h * 0.25), int(right_w * 0.22))
    )

    draw_gauge(
        screen, (gauge_cx, gauge_cy),
        gauge_radius, gauge_display
    )

    # Metrics under gauge
    metric_y = bottom_y + int(bottom_h * 0.78)
    text(
        screen, f"INPUT RMS      {input_rms:.4f}",
        right_x + 22, metric_y, 11, PURPLE, True
    )
    text(
        screen, f"OUTPUT RMS     {output_rms:.4f}",
        right_x + 22, metric_y + 20, 11, TEAL, True
    )

    if attenuation_db is None:
        attenuation_label = "KNOWN-NOISE ATTENUATION  > 160 dB*"
    else:
        attenuation_label = (
            f"KNOWN-NOISE ATTENUATION  {attenuation_db:.1f} dB"
        )

    text(
        screen, attenuation_label,
        right_x + 22, metric_y + 40, 10, CYAN
    )

    # Footer
    pygame.draw.line(
        screen, GRID, (28, H - 30), (W - 28, H - 30), 1
    )
    text(
        screen,
        f"BANDPASS {LOW_CUTOFF}-{HIGH_CUTOFF} Hz"
        "   |   WINDOWED FFT   |   SYNTHETIC SIGNAL",
        30, H - 22, 10, MUTED
    )
    text(
        screen, "ESC  /  EXIT",
        W - 30, H - 22, 10, CYAN, anchor="topright"
    )

    pygame.display.flip()

pygame.quit()