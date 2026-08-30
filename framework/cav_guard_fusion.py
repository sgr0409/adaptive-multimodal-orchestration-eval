"""Reference-preserving fusion with global and hierarchical CAV guards.

``CAVGuardFusion`` evaluates two fixed late-fusion paths on out-of-fold
training predictions: equal-weight fusion is the reference path and
confidence-weighted fusion is the candidate intervention.  The guard counts
candidate benefits (candidate correct, reference wrong) and harms (reference
correct, candidate wrong) on decisive switches.  With a uniform Beta prior,
the posterior probability that a decisive candidate switch is beneficial is

    Pr(theta > 1/2 | B, H),  theta ~ Beta(B + 1, H + 1).

The candidate path is enabled only when that probability reaches a
pre-specified credibility threshold.  Otherwise inference preserves the
reference path.  This differs from modality attention or generic stacking:
the learned object is the value of a decision-changing fusion intervention
relative to an explicit reference, not a new set of modality weights.

``HierarchicalCAVGuardFusion`` adds a local benefit model over a fixed
19-dimensional, label-free feature representation.  An intervention region
must pass a multiplicity-corrected one-sided Clopper--Pearson lower bound.
Independent calibration supplies the conditional finite-sample mode;
cross-fitted calibration is explicitly marked descriptive.

Only out-of-fold predictions, training labels, and an optional independent
calibration sample enter ``fit``.  Test labels, degradation annotations, and
OOD flags are neither accepted nor required at prediction time.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from framework.fusion import confidence_weighted_fusion, equal_weight_fusion


@dataclass(frozen=True)
class CAVGuardStats:
    n_examples: int
    switches: int
    benefits: int
    harms: int
    neutral_switches: int
    posterior_probability_positive: float
    credibility_threshold: float
    candidate_enabled: bool

    @property
    def switch_value(self):
        return ((self.benefits - self.harms) / self.switches
                if self.switches else 0.0)


class CAVGuardFusion:
    """Select a confidence-weighted candidate only with credible positive
    out-of-fold switch value; otherwise preserve equal weighting."""

    def __init__(self, power=3, credibility_threshold=0.95):
        if not 0.5 < credibility_threshold < 1.0:
            raise ValueError("credibility_threshold must lie strictly between 0.5 and 1")
        self.power = power
        self.credibility_threshold = credibility_threshold
        self.stats_ = None

    def fit(self, oof_modality_results_list: Sequence, y: Sequence):
        if len(oof_modality_results_list) != len(y):
            raise ValueError("Out-of-fold modality results and labels must have identical lengths")
        benefits = harms = neutral = switches = 0
        for modality_results, true_label in zip(oof_modality_results_list, y):
            reference = equal_weight_fusion(modality_results)["label"]
            candidate = confidence_weighted_fusion(
                modality_results, power=self.power
            )["label"]
            if reference == candidate:
                continue
            switches += 1
            if candidate == true_label and reference != true_label:
                benefits += 1
            elif reference == true_label and candidate != true_label:
                harms += 1
            else:
                neutral += 1

        probability_positive = float(
            1.0 - beta.cdf(0.5, benefits + 1, harms + 1)
        )
        self.stats_ = CAVGuardStats(
            n_examples=len(y),
            switches=switches,
            benefits=benefits,
            harms=harms,
            neutral_switches=neutral,
            posterior_probability_positive=probability_positive,
            credibility_threshold=self.credibility_threshold,
            candidate_enabled=probability_positive >= self.credibility_threshold,
        )
        return self

    def predict(self, modality_results):
        if self.stats_ is None:
            raise RuntimeError("CAVGuardFusion.fit() must be called before predict()")
        reference = equal_weight_fusion(modality_results)
        candidate = confidence_weighted_fusion(
            modality_results, power=self.power
        )
        selected = candidate if self.stats_.candidate_enabled else reference
        result = dict(selected)
        result.update({
            "modality": "cav_guard_fusion",
            "selected_path": (
                "confidence_weighted" if self.stats_.candidate_enabled
                else "equal_weight"
            ),
            "posterior_probability_positive": (
                self.stats_.posterior_probability_positive
            ),
        })
        return result


def _normalized_entropy(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.sum(values * np.log(values)) / np.log(len(values)))


def _margin(probabilities):
    ordered = np.sort(np.asarray(probabilities, dtype=float))
    return float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0


def _jensen_shannon(left, right):
    left = np.clip(np.asarray(left, dtype=float), 1e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=float), 1e-12, 1.0)
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(left * np.log(left / midpoint))
    divergence += 0.5 * np.sum(right * np.log(right / midpoint))
    return float(divergence / np.log(2.0))


def cav_switch_features(modality_results, power=3):
    """Label-free features describing a candidate departure from reference.

    The vector is invariant to modality order and has the same dimensionality
    for two- and three-modality domains.  It combines modality conflict,
    confidence dispersion, probability entropy/divergence, and the geometry
    of the reference/candidate fused distributions.  True labels, corruption
    indicators, and OOD flags are intentionally absent.
    """
    labels = list(modality_results[0]["probs"].keys())
    modality_probabilities = np.asarray([
        [result["probs"][label] for label in labels]
        for result in modality_results
    ], dtype=float)
    confidences = np.asarray([
        result["confidence"] for result in modality_results
    ], dtype=float)
    entropies = np.asarray([
        _normalized_entropy(probabilities)
        for probabilities in modality_probabilities
    ])
    pairwise_js = np.asarray([
        _jensen_shannon(modality_probabilities[left],
                        modality_probabilities[right])
        for left, right in combinations(range(len(modality_results)), 2)
    ])
    reference = equal_weight_fusion(modality_results)
    candidate = confidence_weighted_fusion(modality_results, power=power)
    reference_probs = np.asarray([
        reference["probs"][label] for label in labels
    ])
    candidate_probs = np.asarray([
        candidate["probs"][label] for label in labels
    ])
    modality_labels = [result["label"] for result in modality_results]
    n_modalities = len(modality_results)
    ordered_confidences = np.sort(confidences)
    top_confidence_gap = (
        ordered_confidences[-1] - ordered_confidences[-2]
        if n_modalities > 1 else ordered_confidences[-1]
    )
    candidate_support = modality_labels.count(candidate["label"]) / n_modalities
    reference_support = modality_labels.count(reference["label"]) / n_modalities
    top_modality_label = modality_labels[int(np.argmax(confidences))]
    return np.asarray([
        float(n_modalities),
        len(set(modality_labels)) / n_modalities,
        float(confidences.mean()),
        float(confidences.std()),
        float(confidences.max() - confidences.min()),
        float(top_confidence_gap),
        float(entropies.mean()),
        float(entropies.std()),
        float(pairwise_js.mean()) if len(pairwise_js) else 0.0,
        float(pairwise_js.max()) if len(pairwise_js) else 0.0,
        _margin(reference_probs),
        _margin(candidate_probs),
        _margin(candidate_probs) - _margin(reference_probs),
        float(0.5 * np.abs(candidate_probs - reference_probs).sum()),
        _jensen_shannon(reference_probs, candidate_probs),
        float(candidate_probs.max() - reference_probs.max()),
        float(candidate_support - reference_support),
        float(top_modality_label == candidate["label"]),
        float(top_modality_label == reference["label"]),
    ], dtype=float)


@dataclass(frozen=True)
class HierarchicalCAVGuardStats:
    global_guard: dict
    local_model_fitted: bool
    calibration_mode: str
    calibration_thresholds_tested: int
    familywise_error_rate: float
    local_score_threshold: Optional[float]
    calibrated_benefits: int
    calibrated_harms: int
    calibrated_decisive_switches: int
    clopper_pearson_lower_benefit_probability: float
    estimated_accuracy_gain_lower_bound: float


class HierarchicalCAVGuardFusion:
    """Hierarchical, instance-selective extension of :class:`CAVGuardFusion`.

    Stage one is the global CAV guard.  Only when it permits the candidate do
    we learn a local score for beneficial reference-to-candidate switches.
    Stage two calibrates a score threshold by maximizing a conservative lower
    bound on accuracy gain.  Candidate thresholds must pass a one-sided exact
    Clopper-Pearson benefit-rate bound with a Bonferroni correction across all
    thresholds searched.

    With an independent ``calibration_*`` sample, the threshold test has its
    usual finite-sample familywise interpretation conditional on the fitted
    score model and exchangeability.  Without one, the implementation uses
    cross-fitted scores for data efficiency and labels the result descriptive.
    The paper reports both the descriptive profile and a disjoint-split strict
    profile that fixes its threshold grid before calibration labels are read.
    """

    def __init__(self, power=3, credibility_threshold=0.95,
                 familywise_error_rate=0.05, min_local_decisive=5, seed=42,
                 local_score_thresholds=None,
                 require_local_certification=False):
        if not 0.0 < familywise_error_rate < 0.5:
            raise ValueError("familywise_error_rate must lie between 0 and 0.5")
        self.power = power
        self.credibility_threshold = credibility_threshold
        self.familywise_error_rate = familywise_error_rate
        self.min_local_decisive = min_local_decisive
        self.seed = seed
        if local_score_thresholds is not None:
            thresholds = tuple(sorted(set(
                float(value) for value in local_score_thresholds
            )))
            if not thresholds or thresholds[0] < 0.0 or thresholds[-1] > 1.0:
                raise ValueError("local_score_thresholds must lie in [0, 1]")
            self.local_score_thresholds = thresholds
        else:
            self.local_score_thresholds = None
        self.require_local_certification = bool(require_local_certification)
        self.global_guard_ = None
        self.local_model_ = None
        self.stats_ = None

    def _new_local_model(self):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.seed,
            ),
        )

    def _decisive_data(self, modality_results_list, y):
        features = []
        outcomes = []
        for modality_results, true_label in zip(modality_results_list, y):
            reference = equal_weight_fusion(modality_results)
            candidate = confidence_weighted_fusion(
                modality_results, power=self.power
            )
            if reference["label"] == candidate["label"]:
                continue
            if candidate["label"] == true_label and reference["label"] != true_label:
                outcome = 1
            elif reference["label"] == true_label and candidate["label"] != true_label:
                outcome = 0
            else:
                continue
            features.append(cav_switch_features(modality_results, self.power))
            outcomes.append(outcome)
        return np.asarray(features, dtype=float), np.asarray(outcomes, dtype=int)

    @staticmethod
    def _cp_lower(successes, failures, alpha):
        if successes == 0:
            return 0.0
        return float(beta.ppf(alpha, successes, failures + 1))

    def _calibrate_threshold(self, scores, outcomes, n_examples):
        thresholds = (
            np.asarray(self.local_score_thresholds, dtype=float)
            if self.local_score_thresholds is not None
            else np.unique(np.asarray(scores, dtype=float))
        )
        if not len(thresholds):
            return None
        alpha_each = self.familywise_error_rate / len(thresholds)
        candidates = []
        for threshold in thresholds:
            selected = scores >= threshold
            benefits = int(np.sum(outcomes[selected] == 1))
            harms = int(np.sum(outcomes[selected] == 0))
            decisive = benefits + harms
            if decisive < self.min_local_decisive:
                continue
            lower = self._cp_lower(benefits, harms, alpha_each)
            if lower <= 0.5:
                continue
            lower_gain = decisive / n_examples * (2.0 * lower - 1.0)
            candidates.append((
                lower_gain,
                decisive,
                -float(threshold),
                float(threshold),
                benefits,
                harms,
                lower,
            ))
        if not candidates:
            return None
        return max(candidates)

    def fit(self, oof_modality_results_list: Sequence, y: Sequence,
            calibration_modality_results_list=None, calibration_y=None):
        if len(oof_modality_results_list) != len(y):
            raise ValueError("Out-of-fold modality results and labels must have identical lengths")
        if (calibration_modality_results_list is not None
                and (calibration_y is None
                     or len(calibration_modality_results_list) != len(calibration_y))):
            raise ValueError("Independent calibration results and labels must have identical lengths")
        self.global_guard_ = CAVGuardFusion(
            power=self.power,
            credibility_threshold=self.credibility_threshold,
        ).fit(oof_modality_results_list, y)

        default_stats = dict(
            global_guard=asdict(self.global_guard_.stats_),
            local_model_fitted=False,
            calibration_mode="global_fallback",
            calibration_thresholds_tested=0,
            familywise_error_rate=self.familywise_error_rate,
            local_score_threshold=None,
            calibrated_benefits=0,
            calibrated_harms=0,
            calibrated_decisive_switches=0,
            clopper_pearson_lower_benefit_probability=0.0,
            estimated_accuracy_gain_lower_bound=0.0,
        )
        if not self.global_guard_.stats_.candidate_enabled:
            self.stats_ = HierarchicalCAVGuardStats(**default_stats)
            return self

        x_train, y_train = self._decisive_data(oof_modality_results_list, y)
        if (len(y_train) < self.min_local_decisive
                or len(np.unique(y_train)) < 2
                or np.bincount(y_train, minlength=2).min() < 2):
            # Global evidence is positive but local evidence cannot support a
            # stable score model.  The strict profile preserves the reference;
            # the default profile preserves the validated global candidate.
            default_stats.update(
                calibration_mode=(
                    "global_candidate_insufficient_local_data_reference_fallback"
                    if self.require_local_certification
                    else "global_candidate_insufficient_local_data"
                ),
                local_score_threshold=None,
            )
            self.stats_ = HierarchicalCAVGuardStats(**default_stats)
            return self

        if calibration_modality_results_list is not None:
            self.local_model_ = self._new_local_model().fit(x_train, y_train)
            x_cal, y_cal = self._decisive_data(
                calibration_modality_results_list, calibration_y
            )
            if not len(y_cal):
                default_stats.update(calibration_mode=(
                    "independent_no_decisive_calibration_reference_fallback"
                    if self.require_local_certification
                    else "independent_no_decisive_calibration"
                ))
                self.local_model_ = None
                self.stats_ = HierarchicalCAVGuardStats(**default_stats)
                return self
            calibration_scores = self.local_model_.predict_proba(x_cal)[:, 1]
            calibration_outcomes = y_cal
            calibration_mode = "independent_exact"
            calibration_n = len(calibration_modality_results_list)
        else:
            n_splits = min(5, int(np.bincount(y_train, minlength=2).min()))
            cross_fitted_scores = np.zeros(len(y_train), dtype=float)
            folds = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.seed
            )
            for fit_indices, score_indices in folds.split(x_train, y_train):
                fold_model = self._new_local_model().fit(
                    x_train[fit_indices], y_train[fit_indices]
                )
                cross_fitted_scores[score_indices] = (
                    fold_model.predict_proba(x_train[score_indices])[:, 1]
                )
            self.local_model_ = self._new_local_model().fit(x_train, y_train)
            calibration_scores = cross_fitted_scores
            calibration_outcomes = y_train
            calibration_mode = "cross_fitted_descriptive"
            calibration_n = len(oof_modality_results_list)

        calibrated = self._calibrate_threshold(
            calibration_scores, calibration_outcomes, calibration_n
        )
        if calibrated is None:
            # The local layer declines to intervene when no
            # multiplicity-corrected subset is evidenced.  The default
            # profile may still use the globally supported candidate; the
            # strict profile preserves the reference.
            self.local_model_ = None
            default_stats.update(
                calibration_mode=(
                    calibration_mode + "_no_local_subset_reference_fallback"
                    if self.require_local_certification
                    else calibration_mode + "_no_local_subset"
                ),
                local_score_threshold=None,
                calibration_thresholds_tested=(
                    len(self.local_score_thresholds)
                    if self.local_score_thresholds is not None
                    else len(np.unique(calibration_scores))
                ),
            )
            self.stats_ = HierarchicalCAVGuardStats(**default_stats)
            return self

        (lower_gain, decisive, _negative_threshold, threshold,
         benefits, harms, lower) = calibrated
        self.stats_ = HierarchicalCAVGuardStats(
            global_guard=asdict(self.global_guard_.stats_),
            local_model_fitted=True,
            calibration_mode=calibration_mode,
            calibration_thresholds_tested=(
                len(self.local_score_thresholds)
                if self.local_score_thresholds is not None
                else len(np.unique(calibration_scores))
            ),
            familywise_error_rate=self.familywise_error_rate,
            local_score_threshold=threshold,
            calibrated_benefits=benefits,
            calibrated_harms=harms,
            calibrated_decisive_switches=decisive,
            clopper_pearson_lower_benefit_probability=lower,
            estimated_accuracy_gain_lower_bound=lower_gain,
        )
        return self

    def predict(self, modality_results):
        if self.stats_ is None:
            raise RuntimeError("HierarchicalCAVGuardFusion.fit() must be called before predict()")
        reference = equal_weight_fusion(modality_results)
        candidate = confidence_weighted_fusion(
            modality_results, power=self.power
        )
        score = None
        candidate_selected = False
        if self.global_guard_.stats_.candidate_enabled:
            if reference["label"] == candidate["label"]:
                selected = reference
            elif self.local_model_ is None:
                candidate_selected = not self.require_local_certification
                selected = candidate if candidate_selected else reference
            else:
                features = cav_switch_features(
                    modality_results, self.power
                ).reshape(1, -1)
                score = float(self.local_model_.predict_proba(features)[0, 1])
                candidate_selected = score >= self.stats_.local_score_threshold
                selected = candidate if candidate_selected else reference
        else:
            selected = reference
        result = dict(selected)
        result.update({
            "modality": "hierarchical_cav_guard_fusion",
            "selected_path": "confidence_weighted" if candidate_selected else "equal_weight",
            "global_candidate_enabled": self.global_guard_.stats_.candidate_enabled,
            "local_benefit_score": score,
            "local_score_threshold": self.stats_.local_score_threshold,
        })
        return result
