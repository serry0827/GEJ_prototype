import os
import glob
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from street_sense.dsp import highpass, normalize_window
from street_sense.yamnet_embed import YAMNetEmbedder

SR = 16000
LABELS = ["bike", "escooter", "other"]

def load_wavs():
    X, y = [], []
    for li, lab in enumerate(LABELS):
        paths = glob.glob(os.path.join("data", lab, "*.wav"))
        for p in paths:
            wav, sr = sf.read(p, dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]
            if sr != SR:
                # Keep it simple: skip mismatched SR for now
                continue
            X.append(wav)
            y.append(li)
    return X, np.array(y, dtype=np.int64)

def main():
    embedder = YAMNetEmbedder()

    wavs, y = load_wavs()
    if len(wavs) < 10:
        raise RuntimeError("Not enough data. Record more clips into data/bike, data/escooter, data/other.")

    feats = []
    for wav in wavs:
        wav = highpass(wav, SR, cutoff_hz=100.0)
        wav = normalize_window(wav)
        emb = embedder.embed(wav)
        feats.append(emb)

    X = np.stack(feats, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=LABELS))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"clf": clf, "label_names": LABELS}, "models/top_clf.joblib")
    print("Saved: models/top_clf.joblib")

if __name__ == "__main__":
    main()
