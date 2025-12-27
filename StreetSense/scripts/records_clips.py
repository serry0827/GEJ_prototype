import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
DURATION = 3.0

def record_clip() -> np.ndarray:
    print(f"Recording {DURATION:.1f}s...")
    audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]

def main():
    label = input("Label to record (bike/escooter/other): ").strip()
    out_dir = os.path.join("data", label)
    os.makedirs(out_dir, exist_ok=True)

    print("Press Enter to record a clip. Type 'q' then Enter to quit.")
    i = 0
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        wav = record_clip()
        ts = int(time.time() * 1000)
        path = os.path.join(out_dir, f"{label}_{ts}.wav")
        sf.write(path, wav, SR)
        i += 1
        print(f"Saved: {path} (#{i})")

if __name__ == "__main__":
    main()
