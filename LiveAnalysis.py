import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# -----------------------------
# Settings (tweak these)
# -----------------------------
SAMPLE_RATE = 48000        # Hz
BLOCK_SIZE = 2048          # samples per audio block (higher = smoother, slower)
CHANNELS = 1               # mono mic input
WINDOW = np.hanning(BLOCK_SIZE)

# Optional: smooth the spectrum a bit (moving average over recent frames)
SMOOTH_FRAMES = 3
spectrum_history = deque(maxlen=SMOOTH_FRAMES)

# -----------------------------
# Plot setup
# -----------------------------
plt.style.use("default")
fig, ax = plt.subplots()
freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)

# We’ll plot only up to some max frequency for readability (e.g. 10 kHz)
MAX_FREQ = 10000
max_bin = np.searchsorted(freqs, MAX_FREQ)

line, = ax.plot(freqs[:max_bin], np.zeros(max_bin), lw=1)

ax.set_title("Live Frequency Spectrum (Mic Input)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, freqs[max_bin - 1])
ax.set_ylim(0, 1)  # will auto-adjust dynamically below

# -----------------------------
# Audio callback buffer
# -----------------------------
latest_block = np.zeros(BLOCK_SIZE, dtype=np.float32)

def audio_callback(indata, frames, time, status):
    global latest_block
    if status:
        # This prints underruns/overruns; you can comment it out if noisy
        print(status)

    # indata shape: (frames, channels)
    block = indata[:, 0].astype(np.float32)

    # If frames != BLOCK_SIZE (rare), pad/truncate safely
    if len(block) < BLOCK_SIZE:
        block = np.pad(block, (0, BLOCK_SIZE - len(block)))
    else:
        block = block[:BLOCK_SIZE]

    latest_block = block

# -----------------------------
# Animation update
# -----------------------------
def update(_frame):
    # Window the signal to reduce spectral leakage
    x = latest_block * WINDOW

    # FFT -> magnitude spectrum
    X = np.fft.rfft(x)
    mag = np.abs(X)

    # Convert to something more visually stable:
    # Normalize by block size; optionally log-scale
    mag = mag / (BLOCK_SIZE / 2)

    # Keep only up to MAX_FREQ
    mag = mag[:max_bin]

    # Smooth across last few frames
    spectrum_history.append(mag)
    mag_smooth = np.mean(spectrum_history, axis=0)

    line.set_ydata(mag_smooth)

    # Dynamic y-limits so it doesn't look flat
    peak = float(np.max(mag_smooth) + 1e-9)
    ax.set_ylim(0, peak * 1.2)

    return (line,)

# -----------------------------
# Run stream + animation
# -----------------------------
def main():
    print("Starting mic stream... Close the plot window to stop.")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=audio_callback,
    ):
        ani = FuncAnimation(fig, update, interval=30, blit=True)
        plt.show()

if __name__ == "__main__":
    main()
