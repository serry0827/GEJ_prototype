#dsp = digital signal processing
import numpy as np
from scipy.signal import butter, lfilter

def highpass(x: np.ndarray, sr: int, cutoff_hz: float = 100.0, order: int = 4) -> np.ndarray:
    """Light high-pass to remove rumble / handling noise."""
    nyq = 0.5 * sr
    b, a = butter(order, cutoff_hz / nyq, btype="highpass")
    return lfilter(b, a, x).astype(np.float32)

def normalize_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-window RMS normalization (helps with distance variation)."""
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / (rms + eps)).astype(np.float32)