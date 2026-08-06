"""Telemetry-modality scorer: extracts simple summary features (mean, std,
min, max, and linear trend) per channel from the time-series telemetry,
then a logistic-regression classifier trained on those features predicts
severity. Confidence is the classifier's own max predicted probability.

Channel names are inferred from the telemetry dict's own keys (sorted, for
a deterministic feature order) rather than hardcoded to the first domain's
temperature/vibration/pressure channels, the same way framework/fusion.py
infers its label set from the data -- so this works unchanged for a second
task domain's differently-named channels (e.g. IT-incident triage's
cpu_pct/latency_ms/error_rate_pct)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Historical default for the equipment-maintenance-triage domain; kept for
# backward-compat imports but no longer used internally.
CHANNELS = ["temperature", "vibration", "pressure"]


def extract_features(telemetry):
    feats = []
    for ch in sorted(telemetry.keys()):
        series = np.asarray(telemetry[ch], dtype=float)
        t = np.arange(len(series))
        trend = np.polyfit(t, series, 1)[0] if len(series) > 1 else 0.0
        feats.extend([series.mean(), series.std(), series.min(), series.max(), trend])
    return feats


class TelemetryScorer:
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        self._fitted = False

    def fit(self, telemetry_list, labels):
        X = [extract_features(t) for t in telemetry_list]
        self.clf.fit(X, labels)
        self._fitted = True

    def predict(self, telemetry):
        assert self._fitted, "TelemetryScorer.fit() must be called before predict()"
        x = [extract_features(telemetry)]
        probs = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        best_idx = probs.argmax()
        return {
            "modality": "telemetry",
            "label": classes[best_idx],
            "confidence": float(probs[best_idx]),
            "probs": {c: float(p) for c, p in zip(classes, probs)},
        }
