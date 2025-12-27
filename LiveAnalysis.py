import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, lfilter
from collections import deque

# -----------------------------
# Settings (tweak these)
# -----------------------------
SAMPLE_RATE = 44100        # Hz
BLOCK_SIZE = 2000         # samples per audio block (higher = smoother, slower)
CHANNELS = 1               # mono mic input
VOICE_LOW = 1
VOICE_HIGH = 2500
WINDOW = np.hanning(BLOCK_SIZE)

# Noise estimation
NOISE_FRAMES = 20
noise_history = deque(maxlen=NOISE_FRAMES)

latest_block = np.zeros(BLOCK_SIZE, dtype=np.float32)

# Band-pass filter
def bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a

b_bp, a_bp = bandpass(VOICE_LOW, VOICE_HIGH, SAMPLE_RATE)



# Optional: smooth the spectrum a bit (moving average over recent frames)
SMOOTH_FRAMES = 3
spectrum_history = deque(maxlen=SMOOTH_FRAMES)


# Plot setup
plt.style.use("default")
fig, ax = plt.subplots()
freqs = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)
voice_bins = np.where((freqs >= VOICE_LOW) & (freqs <= VOICE_HIGH))[0]

# We’ll plot only up to some max frequency for readability (e.g. 10 kHz)
#MAX_FREQ = 10000
#max_bin = np.searchsorted(freqs, MAX_FREQ)

line, = ax.plot(freqs[voice_bins], np.zeros(len(voice_bins)))

ax.set_title("Live Frequency Spectrum (Mic Input, Noise-Reduced)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_xlim(VOICE_LOW, VOICE_HIGH)
ax.set_ylim(0, 0.02)  # will auto-adjust dynamically below


# Audio callback

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
    latest_block = block[:BLOCK_SIZE]

# -----------------------------
# Animation update
# -----------------------------
def update(_):
    # Band-pass filter
    filtered = lfilter(b_bp, a_bp, latest_block)

    # FFT
    spectrum = np.abs(np.fft.rfft(filtered * WINDOW))
    spectrum = spectrum / (BLOCK_SIZE / 2)

    voice_spec = spectrum[voice_bins]

    # Noise floor learning
    noise_history.append(voice_spec)
    noise_floor = np.mean(noise_history, axis=0)

    # Noise subtraction
    clean = np.maximum(voice_spec - 1.2 * noise_floor, 0)

    line.set_ydata(clean)

    peak = np.max(clean) + 1e-6
    ax.set_ylim(0, peak * 3.0)

    return (line,)


# Run stream + animation
def main():
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=audio_callback,
    ):
        ani = FuncAnimation(fig, update, interval=20, blit=True)
        plt.show()

if __name__ == "__main__":
    main()