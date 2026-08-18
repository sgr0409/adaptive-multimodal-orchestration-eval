"""Distribution-aware confidence recalibration, a new fusion-weighting
input designed to directly target this paper's diagnosed failure mode: on
real data (CrisisMMD), a classifier's own softmax confidence is not just
weakly informative, it is *anti*-correlated with reliability (degraded text
scores higher confidence than clean text). Every fusion mechanism tested so
far -- hand-designed power-law weighting, stacking, learned gating --
trusts that softmax confidence as its input signal and inherits the same
failure regardless of how cleverly it combines modalities. This targets the
signal itself, not the combination rule.

The mechanism: for each modality, fit a class-conditional Gaussian model
over the same frozen embeddings the per-modality scorer already uses
(Lee et al.'s Mahalanobis-distance out-of-distribution
detector~\\cite{lee2018simple}, applied here to a new problem -- recalibrating
fusion-time confidence under real-world distribution shift -- rather than
its original use, flagging OOD or adversarial inputs outright). At predict
time, compute each input's Mahalanobis distance to the nearest class
centroid under a shared covariance; convert that distance to a trust score
via an exponential decay scaled by the training set's own median distance
(data-driven, not an asserted constant); and multiply the classifier's raw
confidence by that trust score before it ever reaches fusion. A classifier
can be confidently wrong on an input unlike anything in its training set;
softmax confidence alone does not know that, but embedding-space distance
does.

This does not replace confidence-weighted fusion's combination rule
(Eq. w_i = c_i^p / sum_j c_j^p) -- it replaces what gets plugged into c_i,
isolating exactly one question: does recalibrating the confidence signal
fix the transfer failure, holding the fusion mechanism itself fixed.
"""
import numpy as np


class OODConfidenceRecalibrator:
    """Fits one class-conditional Gaussian model (shared covariance) per
    modality from training embeddings, then recalibrates a raw confidence
    score at predict time using each input's Mahalanobis distance to its
    nearest class centroid."""

    def __init__(self, shrinkage=1e-3):
        self.shrinkage = shrinkage
        self.class_means_ = None
        self.precision_ = None
        self.labels_ = None
        self.median_distance_ = None
        self._fitted = False

    def fit(self, embeddings, labels):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        self.labels_ = sorted(set(labels))
        labels_arr = np.asarray(labels)

        self.class_means_ = {}
        centered = []
        for label in self.labels_:
            mask = labels_arr == label
            mu = embeddings[mask].mean(axis=0)
            self.class_means_[label] = mu
            centered.append(embeddings[mask] - mu)
        centered = np.concatenate(centered, axis=0)

        d = embeddings.shape[1]
        cov = (centered.T @ centered) / max(len(centered) - 1, 1)
        cov += self.shrinkage * np.eye(d)
        self.precision_ = np.linalg.inv(cov)

        train_distances = self._distances(embeddings)
        self.median_distance_ = float(np.median(train_distances))
        self._fitted = True

    def _distances(self, embeddings):
        dists = []
        for x in embeddings:
            best = min(
                float((x - mu) @ self.precision_ @ (x - mu))
                for mu in self.class_means_.values()
            )
            dists.append(np.sqrt(max(best, 0.0)))
        return np.array(dists)

    def trust_scores(self, embeddings):
        assert self._fitted, "OODConfidenceRecalibrator.fit() must be called before use"
        scale = self.median_distance_ if self.median_distance_ > 0 else 1.0
        distances = self._distances(np.asarray(embeddings, dtype=np.float64))
        return np.exp(-distances / scale)

    def recalibrate(self, embeddings, raw_confidences):
        trust = self.trust_scores(embeddings)
        return np.asarray(raw_confidences) * trust
