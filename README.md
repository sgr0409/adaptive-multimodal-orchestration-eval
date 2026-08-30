# Adaptive Multimodal Orchestration Evaluation

Evaluation code and released artifacts for the IEEE Access manuscript
"CAV-Portfolio: Certified Reference-Preserving Selection of Multimodal
Fusion Policies Under Reliability Shift."

The repository compares equal-weight, confidence-weighted, a naive accuracy
selector, stacking, learned gating, CAV-Portfolio, and the earlier local
H-CAV ablation across two generated
text/image/telemetry domains and two naturally sourced image--text datasets
with injected corruption (CrisisMMD and MM-IMDb). Random train/test
partitions are overlapping sensitivity analyses, not independent
replications.

## CAV-Portfolio architecture

`framework/cav_portfolio_guard.py` implements candidate-agnostic,
reference-preserving selection from a fixed fusion-policy portfolio. Equal
weighting is the protected reference; the evaluated candidates are
confidence weighting, stacking, and learned gating. For every candidate, the
guard counts paired benefits and harms relative to the reference. For each
candidate, paired accuracy gain is written as `q * (2 * theta - 1)`, where
`q` is decisive-comparison probability and `theta` is candidate benefit
probability conditional on a decisive comparison. Separate one-sided
Clopper--Pearson lower bounds for `q` and `theta`, Bonferroni-corrected across
both components and all fixed candidates, produce a simultaneous lower
confidence bound on population accuracy gain. A candidate is admissible only
when it has at least five decisive comparisons and this gain bound is
positive. The selected path maximizes certified gain; if none is admitted,
inference reproduces equal weighting.

Because the paired accuracy contrast lies in `[-1, 1]`, a selected lower gain
bound `G_L` also certifies non-degradation for deployment paired-outcome
distributions within total-variation radius `G_L / 2` of calibration. This is
a quantified local shift statement, not protection under arbitrary shift.

The primary 50/50 sensitivity study uses two leakage barriers. Base scorers
first produce five-fold out-of-fold predictions; stacking and learned gating
then receive a second stratified cross-fit, so portfolio evidence never scores
a meta-learner on an example used to fit that meta-learner. These results are
labelled descriptive because repeated folds share training data. The strict
study uses disjoint 40/30/30 training/calibration/test splits. Candidate
models are fixed before calibration labels are read, and the final test labels
are opened only after the portfolio path is frozen.

Run the selector tests and both evaluations with:

```bash
python -m unittest tests/test_cav_portfolio_guard.py
python experiments/cav_portfolio_multidomain.py
python experiments/cav_portfolio_independent_calibration.py
python experiments/cav_portfolio_evidence_sensitivity.py
python experiments/cav_portfolio_certificate_audit.py
```

Complete partition-level evidence, selected paths, split indices, comparisons
against every constituent candidate, and the unattainable test-oracle ceiling
are stored in:

- `experiments/results/cav_portfolio_multidomain.json`
- `experiments/results/cav_portfolio_independent_calibration.json`
- `experiments/results/cav_portfolio_evidence_sensitivity.json`
- `experiments/results/cav_portfolio_certificate_audit.json`

## Earlier H-CAV local ablation

`framework/cav_guard_fusion.py` implements both the global ablation and the
hierarchical architecture. Stage I enables confidence weighting only when
five-fold OOF benefit/harm evidence satisfies
`Pr(theta > 0.5 | B, H) >= 0.95` under a uniform Beta prior. Stage II learns a
logistic benefit score on decisive switches from 19 label-free,
modality-order-invariant features: conflict, confidence dispersion, entropy,
pairwise Jensen--Shannon divergence, and reference/candidate probability
geometry. A score threshold is admitted only when it contains at least five
decisive calibration examples and its Bonferroni-corrected, one-sided exact
Clopper--Pearson lower benefit bound exceeds `0.5`.

Passing an independent calibration sample to
`HierarchicalCAVGuardFusion.fit` activates the finite-sample calibration mode
under the assumptions documented in the class. The primary 50/50 experiment
uses inner cross-fitted scores for data efficiency and labels those results
`cross_fitted_descriptive`; they are not presented as a formal guarantee.
The independent-calibration validation uses domain-stratified 40/30/30
training/calibration/test splits, a fixed predeclared threshold grid
`{0.5, 0.6, 0.7, 0.8, 0.9}`, and mandatory equal-weight fallback when no
local region is certified. One shared guard is trained and calibrated on the
pooled four-domain mixture while each domain retains its own base scorers.
Test labels, degradation annotations, and OOD flags are not accepted at
prediction time. Rejected instances reproduce equal weighting.

Run its unit tests and the retained ablation evaluations with:

```bash
python -m unittest tests/test_cav_guard_fusion.py \
  tests/test_cav_guard_independent_calibration.py \
  tests/test_cav_guard_leave_one_domain_out.py
python experiments/cav_guard_multidomain.py
python experiments/cav_guard_pooled_independent_calibration.py
python experiments/cav_guard_leave_one_domain_out.py
```

The complete partition-level artifacts are
`experiments/results/cav_guard_multidomain.json` and
`experiments/results/cav_guard_pooled_independent_calibration.json`. The
latter records every split index; the 30 calibration/test pairs are disjoint
within partitions, although repeated partitions overlap and remain a
sensitivity analysis.

## Leave-one-domain-out transfer

`experiments/cav_guard_leave_one_domain_out.py` tests transfer of the fusion
control policy rather than zero-shot transfer of each underlying task. For
each target domain, its task-specific base scorers are fit on its 40% training
split, but the global gate, local benefit model, and fixed-grid calibration
threshold use only the other three domains. No target-domain example enters
guard training or calibration; its reserved calibration split is unused.

The complete audit artifact is
`experiments/results/cav_guard_leave_one_domain_out.json`. It stores every
split index, source-domain list, explicit zero target-domain guard sample
counts, 120 held-out evaluations, and the fixed seed-42 audit.

## Exact CAV diagnostic

For equal-weight (EW) and confidence-weighted (CW) fusion, the released
diagnostic computes

```text
Acc(CW) - Acc(EW) = C * A * V
```

where `C` is modality-prediction conflict rate, `A` is the probability that
CW changes EW's decision given conflict, and `V` is the signed correctness
advantage on changed decisions. It also reports confidence-ranking lift,
10,000 paired bootstrap resamples, and a 5,000-permutation
reliability-alignment comparator.

```bash
python experiments/reliability_complementarity_decomposition.py \
  --root /path/to/adaptive-multimodal-orchestration-eval \
  --resamples 10000 \
  --permutations 5000 \
  --output experiments/results/reliability_complementarity_decomposition.json
```

`framework/fusion_diagnostics.py` computes the same exact identity inside
each partition in the four `robustness_multiseed.py` experiments.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset preparation and domain-specific experiments live under `data*/` and
`experiments/`. The final JSON results committed under each domain's
`results/` directory are the source of truth for manuscript numbers.

The companion paper source is in the `adaptive-multimodal-orchestration/`
module of the `ai-spm-guardrail-paper` repository.
