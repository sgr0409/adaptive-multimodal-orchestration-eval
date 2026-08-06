"""Text-modality scorer: embeds a maintenance ticket with a local MiniLM
sentence encoder, then a small logistic-regression classifier (trained on
the train split) predicts severity from the embedding. Confidence is the
classifier's own max predicted probability (a real prediction-margin
signal, not a proxy) -- deliberately uninformative "degraded" tickets
should get low confidence here because the classifier has genuinely never
seen vocabulary like theirs paired with a clear class during training."""
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["normal", "warning", "critical"]


class TextScorer:
    def __init__(self, device="cpu"):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        self._fitted = False

    def embed(self, texts):
        return self.encoder.encode(texts, show_progress_bar=False)

    def fit(self, texts, labels):
        X = self.embed(texts)
        self.clf.fit(X, labels)
        self._fitted = True

    def predict(self, text):
        assert self._fitted, "TextScorer.fit() must be called before predict()"
        x = self.embed([text])
        probs = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        best_idx = probs.argmax()
        return {
            "modality": "text",
            "label": classes[best_idx],
            "confidence": float(probs[best_idx]),
            "probs": {c: float(p) for c, p in zip(classes, probs)},
        }
