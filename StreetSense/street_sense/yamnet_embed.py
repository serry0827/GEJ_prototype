import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# YAMNet expects mono waveform at 16 kHz in [-1, 1]
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

class YAMNetEmbedder:
    def __init__(self):
        self.model = hub.load(YAMNET_HANDLE)

    @tf.function
    def _infer(self, wav_16k: tf.Tensor):
        scores, embeddings, spectrogram = self.model(wav_16k)
        return scores, embeddings  # embeddings: [num_frames, 1024]

    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        """
        Returns a single 1024-d embedding by averaging YAMNet frame embeddings.
        """
        wav_16k = wav_16k.astype(np.float32)
        # Ensure tensor shape: [N]
        scores, emb = self._infer(tf.convert_to_tensor(wav_16k))
        emb_np = emb.numpy()
        if emb_np.ndim != 2 or emb_np.shape[0] == 0:
            return np.zeros((1024,), dtype=np.float32)
        return emb_np.mean(axis=0).astype(np.float32)