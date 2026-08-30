"""Reference-preserving selection from a fixed fusion-policy portfolio.

The selector compares each frozen candidate with a named reference on paired
predictions.  For each candidate, paired gain is factored into the probability of a decisive
comparison and the conditional probability that the candidate wins that
comparison.  Separate multiplicity-corrected one-sided exact bounds yield a
simultaneous lower confidence bound on population accuracy gain.  The selector
maximizes that certified gain and otherwise returns the reference.

The finite-sample interpretation requires candidates fixed before labels from
an independent i.i.d. calibration set are read, Bernoulli decisive indicators,
and Bernoulli benefits conditional on the decisive count. Cross-fitted training
predictions remain useful for descriptive model selection, but do not satisfy
that independent-calibration premise.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from scipy.stats import beta


@dataclass(frozen=True)
class PortfolioCandidateEvidence:
    name: str
    n_examples: int
    switches: int
    benefits: int
    harms: int
    neutral_switches: int
    decisive_switches: int
    corrected_alpha_per_component: float
    lower_decisive_probability: float
    lower_benefit_probability: float
    lower_accuracy_gain: float
    certified_total_variation_radius: float
    admissible: bool


@dataclass(frozen=True)
class CAVPortfolioGuardStats:
    reference_name: str
    candidate_names: tuple
    familywise_error_rate: float
    minimum_decisive_switches: int
    calibration_mode: str
    selected_path: str
    selected_lower_accuracy_gain: float
    selected_total_variation_radius: float
    candidate_evidence: dict


class CAVPortfolioGuardFusion:
    """Choose a supported fusion path or preserve the reference.

    Inputs to :meth:`fit` are already-fused probability records.  This keeps
    the selector independent of the candidate implementations: fixed rules,
    stacking, learned gates, or other late-fusion policies can share the same
    paired evidence calculation.
    """

    def __init__(self, reference_name="equal_weight",
                 familywise_error_rate=0.05,
                 minimum_decisive_switches=5):
        if not 0.0 < familywise_error_rate < 0.5:
            raise ValueError("familywise_error_rate must lie between 0 and 0.5")
        if minimum_decisive_switches < 1:
            raise ValueError("minimum_decisive_switches must be positive")
        self.reference_name = str(reference_name)
        self.familywise_error_rate = float(familywise_error_rate)
        self.minimum_decisive_switches = int(minimum_decisive_switches)
        self.stats_ = None

    @staticmethod
    def _label(result):
        if not isinstance(result, Mapping) or "label" not in result:
            raise ValueError("every fused result must be a mapping with a label")
        return result["label"]

    @staticmethod
    def _lower(successes, failures, alpha):
        if successes == 0:
            return 0.0
        return float(beta.ppf(alpha, successes, failures + 1))

    def _candidate_evidence(self, name, reference, candidate, labels,
                            corrected_alpha):
        switches = benefits = harms = neutral = 0
        for reference_result, candidate_result, true_label in zip(
                reference, candidate, labels):
            reference_label = self._label(reference_result)
            candidate_label = self._label(candidate_result)
            if reference_label == candidate_label:
                continue
            switches += 1
            if candidate_label == true_label and reference_label != true_label:
                benefits += 1
            elif reference_label == true_label and candidate_label != true_label:
                harms += 1
            else:
                neutral += 1
        decisive = benefits + harms
        lower_decisive = self._lower(
            decisive, len(labels) - decisive, corrected_alpha
        )
        lower_benefit = self._lower(benefits, harms, corrected_alpha)
        lower_gain = lower_decisive * (2.0 * lower_benefit - 1.0)
        return PortfolioCandidateEvidence(
            name=name,
            n_examples=len(labels),
            switches=switches,
            benefits=benefits,
            harms=harms,
            neutral_switches=neutral,
            decisive_switches=decisive,
            corrected_alpha_per_component=corrected_alpha,
            lower_decisive_probability=lower_decisive,
            lower_benefit_probability=lower_benefit,
            lower_accuracy_gain=lower_gain,
            certified_total_variation_radius=max(0.0, lower_gain / 2.0),
            admissible=(
                decisive >= self.minimum_decisive_switches
                and lower_gain > 0.0
            ),
        )

    def fit(self, reference_results: Sequence,
            candidate_results: Mapping[str, Sequence], labels: Sequence,
            calibration_mode="independent_exact"):
        labels = list(labels)
        reference_results = list(reference_results)
        if len(reference_results) != len(labels):
            raise ValueError("reference results and labels must have equal length")
        if not candidate_results:
            raise ValueError("candidate_results must contain at least one path")
        # Preserve the caller's predeclared order for deterministic,
        # complexity-aware tie breaking.  The experiments declare confidence
        # weighting, stacking, then learned gating, so an exact certificate
        # tie favors the simpler earlier path without consulting test data.
        candidate_names = tuple(candidate_results)
        if self.reference_name in candidate_names:
            raise ValueError("reference_name cannot also be a candidate name")
        for name in candidate_names:
            if len(candidate_results[name]) != len(labels):
                raise ValueError(
                    f"candidate {name!r} and labels must have equal length"
                )

        # There are two simultaneous binomial bounds per candidate: decisive
        # coverage and conditional candidate benefit.  Bonferroni correction
        # therefore covers 2J component statements for J fixed candidates.
        corrected_alpha = (
            self.familywise_error_rate / (2 * len(candidate_names))
        )
        evidence = {
            name: self._candidate_evidence(
                name,
                reference_results,
                candidate_results[name],
                labels,
                corrected_alpha,
            )
            for name in candidate_names
        }
        admitted = [name for name in candidate_names if evidence[name].admissible]
        selected = (
            max(
                admitted,
                key=lambda name: (
                    evidence[name].lower_accuracy_gain,
                    -candidate_names.index(name),
                ),
            )
            if admitted else self.reference_name
        )
        selected_lower_gain = (
            evidence[selected].lower_accuracy_gain
            if selected != self.reference_name else 0.0
        )
        selected_tv_radius = (
            evidence[selected].certified_total_variation_radius
            if selected != self.reference_name else 0.0
        )
        self.stats_ = CAVPortfolioGuardStats(
            reference_name=self.reference_name,
            candidate_names=candidate_names,
            familywise_error_rate=self.familywise_error_rate,
            minimum_decisive_switches=self.minimum_decisive_switches,
            calibration_mode=str(calibration_mode),
            selected_path=selected,
            selected_lower_accuracy_gain=selected_lower_gain,
            selected_total_variation_radius=selected_tv_radius,
            candidate_evidence={
                name: asdict(candidate) for name, candidate in evidence.items()
            },
        )
        return self

    def predict(self, reference_result, candidate_results: Mapping[str, Mapping]):
        if self.stats_ is None:
            raise RuntimeError("CAVPortfolioGuardFusion.fit() must be called first")
        if self.stats_.selected_path == self.reference_name:
            selected = reference_result
        else:
            if self.stats_.selected_path not in candidate_results:
                raise ValueError("selected candidate is missing at prediction time")
            selected = candidate_results[self.stats_.selected_path]
        result = dict(selected)
        result.update({
            "modality": "cav_portfolio_guard",
            "selected_path": self.stats_.selected_path,
            "calibration_mode": self.stats_.calibration_mode,
        })
        return result
