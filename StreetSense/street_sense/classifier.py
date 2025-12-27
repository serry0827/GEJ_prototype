from dataclasses import dataclass
import numpy as np
import joblib

@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict

class TopClassifier:
    def __init__(self, model_path: str):
        payload = joblib.load(model_path)
        self.clf = payload["clf"]
        self.label_names = payload["label_names"]

    def predict(self, embedding: np.ndarray) -> Prediction:
        X = embedding.reshape(1, -1)
        probs = self.clf.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        label = self.label_names[idx]
        conf = float(probs[idx])
        return Prediction(label=label, confidence=conf, probs={self.label_names[i]: float(probs[i]) for i in range(len(probs))})
