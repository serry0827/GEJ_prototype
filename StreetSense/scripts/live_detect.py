import time
import numpy as np
import sounddevice as sd

from street_sense.dsp import highpass, normalize_window
from street_sense.yamnet_embed import YAMNetEmbedder
from street_sense.classifier import TopClassifier
from street_sense.events import EventSmoother

SR = 16000
WIN_S = 1.0
HOP_S = 0.25

WIN_N = int(SR * WIN_S)
HOP_N = int(SR * HOP_S)

def main():
    embedder = YAMNetEmbedder()
    top = TopClassifier("models/top_clf.joblib")
    smoother = EventSmoother(target_labels=("bike", "escooter"), start_th=0.70, end_th=0.40, min_duration_s=0.8)

    ring = np.zeros(WIN_N * 8, dtype=np.float32)  # a few seconds buffer
    wptr = 0
    total_samples = 0

    def callback(indata, frames, time_info, status):
        nonlocal ring, wptr, total_samples
        x = indata[:, 0].astype(np.float32)
        n = len(x)
        # write into ring buffer
        end = wptr + n
        if end < len(ring):
            ring[wptr:end] = x
        else:
            first = len(ring) - wptr
            ring[wptr:] = x[:first]
            ring[: end % len(ring)] = x[first:]
        wptr = end % len(ring)
        total_samples += n

    print("Listening... (Ctrl+C to stop)")
    t0 = time.time()
    last_proc = 0

    with sd.InputStream(samplerate=SR, channels=1, callback=callback, blocksize=HOP_N):
        while True:
            time.sleep(HOP_S / 2)

            # process every hop
            if total_samples - last_proc < HOP_N:
                continue
            last_proc = total_samples

            # read the latest WIN_N samples from ring
            rptr = (wptr - WIN_N) % len(ring)
            if rptr + WIN_N <= len(ring):
                wav = ring[rptr:rptr + WIN_N].copy()
            else:
                part1 = ring[rptr:].copy()
                part2 = ring[: (rptr + WIN_N) % len(ring)].copy()
                wav = np.concatenate([part1, part2], axis=0)

            # DSP
            wav = highpass(wav, SR, cutoff_hz=100.0)
            wav = normalize_window(wav)

            # embed + classify
            emb = embedder.embed(wav)
            pred = top.predict(emb)

            # target probability = sum of bike + escooter
            p_target = 0.0
            for k in ("bike", "escooter"):
                p_target += pred.probs.get(k, 0.0)

            t_mid = time.time() - t0
            evt = smoother.update(t_mid=t_mid, label=pred.label, conf=pred.confidence, p_target=p_target)

            # Print a lightweight live meter
            print(f"\r{pred.label:8s} conf={pred.confidence:0.2f} p_target={p_target:0.2f}", end="")

            if evt is not None:
                print("\nEVENT:", evt)

if __name__ == "__main__":
    main()
