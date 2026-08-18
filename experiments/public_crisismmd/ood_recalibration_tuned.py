"""Refinement of ood_recalibration_experiment.py, motivated directly by what
that experiment found: applying OOD-Mahalanobis confidence recalibration
uniformly to both modalities makes things WORSE (30-seed Wilcoxon p=0.0005
favoring equal-weight fusion over recalibrated confidence-weighted fusion),
not better. The diagnosis was mechanistic, not noise: text's raw confidence
is the one that's actually inverted (degraded 0.815 > clean 0.804,
see run_benchmark.py's modality_confidence_reliability_check), so recalibrating
it helps; but image's raw confidence was already correctly-directioned
(degraded 0.826 < clean 0.921), and blind recalibration corrupted it
(recalibrated trust ends up backwards for image: degraded 0.392 > clean
0.266). Applying the same fixed-strength correction to both modalities does
more harm (breaking image) than good (fixing text).

This script tests whether a per-modality blend weight, selected via k-fold
cross-validation on the TRAIN split only (never touching test), fixes that:
recalibrated_confidence = raw_confidence * (trust_score ** alpha), alpha in
[0, 1] per modality, alpha=0 meaning "ignore recalibration, use raw
confidence" and alpha=1 the original (failed) strategy. alpha is selected
per modality by a joint grid search over (alpha_text, alpha_image) that
maximizes hard-subset fusion accuracy on out-of-fold predictions from the
train split -- the same out-of-fold discipline framework/learned_fusion.py's
StackingFusion/LearnedGatingFusion already use for their meta-learners, and
the same "sweep, don't assert" discipline experiments/sweep_fusion_power.py
uses for the fusion power hyperparameter. The selection never sees test
data; only the resulting fixed alpha pair is applied to a fresh test split.

Reports the honest result, same as everything else in this paper.
"""
import json
import random
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.image_scorer import ImageScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion
from framework.ood_calibration import OODConfidenceRecalibrator

SEED = 42
DATA_DIR = ROOT / "data_public_crisismmd"
RESULTS_DIR = ROOT / "experiments" / "public_crisismmd" / "results"
N_SEEDS = 30
N_BOOTSTRAP = 2000
N_INNER_FOLDS = 5
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def fit_predict(X_train, y_train, X_test, modality_name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    out = []
    for p in probs:
        best = int(np.argmax(p))
        out.append({"modality": modality_name, "label": classes[best], "confidence": float(p[best]),
                    "probs": {c: float(v) for c, v in zip(classes, p)}})
    return out


def apply_alpha(modality_results, trust, alpha):
    if alpha == 0.0:
        return modality_results
    return [{**r, "confidence": float(r["confidence"] * (t ** alpha))} for r, t in zip(modality_results, trust)]


def mcnemar(correct_a, correct_b):
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    p = scipy_stats.binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"b": b, "c": c, "p_value": round(float(p), 6)}


def bootstrap_ci(correct, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    correct = np.asarray(correct)
    n = len(correct)
    accs = [correct[rng.randint(0, n, size=n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return {"lo95": round(float(lo), 4), "hi95": round(float(hi), 4)}


def select_alphas(train_idx, y_train, text_emb, image_emb, scenarios, seed, n_folds=N_INNER_FOLDS):
    """5-fold OOF grid search over (alpha_text, alpha_image), train-split only.
    Selection criterion: hard-subset (exactly one modality degraded) fused
    accuracy on OOF predictions -- the same diagnostic subset the rest of
    the paper reports on, so the selected alpha is optimized for the
    condition that actually matters, not overall accuracy where the effect
    is diluted."""
    n_tr = len(train_idx)
    train_idx_arr = np.array(train_idx)
    oof_text = [None] * n_tr
    oof_image = [None] * n_tr
    oof_text_trust = [None] * n_tr
    oof_image_trust = [None] * n_tr

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_i, hold_i in kf.split(np.arange(n_tr)):
        fit_global = train_idx_arr[fit_i]
        hold_global = train_idx_arr[hold_i]
        y_fit = [y_train[i] for i in fit_i]

        t_preds = fit_predict(text_emb[fit_global], y_fit, text_emb[hold_global], "text")
        i_preds = fit_predict(image_emb[fit_global], y_fit, image_emb[hold_global], "image")

        t_ood = OODConfidenceRecalibrator()
        t_ood.fit(text_emb[fit_global], y_fit)
        i_ood = OODConfidenceRecalibrator()
        i_ood.fit(image_emb[fit_global], y_fit)
        t_trust = t_ood.trust_scores(text_emb[hold_global])
        i_trust = i_ood.trust_scores(image_emb[hold_global])

        for local_j, global_j in enumerate(hold_i):
            oof_text[global_j] = t_preds[local_j]
            oof_image[global_j] = i_preds[local_j]
            oof_text_trust[global_j] = t_trust[local_j]
            oof_image_trust[global_j] = i_trust[local_j]

    y_oof = [y_train[i] for i in range(n_tr)]
    deg_oof = [len(scenarios[train_idx_arr[i]]["degraded_modalities"]) for i in range(n_tr)]
    hard_oof = [j for j, d in enumerate(deg_oof) if d == 1]

    best = {"alpha_text": 0.0, "alpha_image": 0.0, "hard_acc": -1.0}
    grid_results = []
    for a_text, a_image in product(ALPHA_GRID, ALPHA_GRID):
        t_recal = apply_alpha(oof_text, oof_text_trust, a_text)
        i_recal = apply_alpha(oof_image, oof_image_trust, a_image)
        preds = [confidence_weighted_fusion([t_recal[j], i_recal[j]])["label"] for j in range(n_tr)]
        hard_correct = [preds[j] == y_oof[j] for j in hard_oof]
        hard_acc = sum(hard_correct) / len(hard_correct)
        grid_results.append({"alpha_text": a_text, "alpha_image": a_image, "oof_hard_acc": round(hard_acc, 4)})
        if hard_acc > best["hard_acc"]:
            best = {"alpha_text": a_text, "alpha_image": a_image, "hard_acc": hard_acc}

    return best["alpha_text"], best["alpha_image"], grid_results


def main():
    print("Loading scenarios and computing embeddings once for the full pool...", flush=True)
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    labels_all = [r["label"] for r in scenarios]

    text_encoder = TextScorer().encoder
    text_emb = text_encoder.encode([r["text"] for r in scenarios], show_progress_bar=False)
    image_embedder = ImageScorer()
    image_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in scenarios])
    n = len(scenarios)
    print(f"Embeddings ready for {n} scenarios.", flush=True)

    # --- Single split (seed=42) ---
    print("\nSelecting (alpha_text, alpha_image) via train-only 5-fold OOF grid search...", flush=True)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = n // 2
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    y_train = [labels_all[i] for i in train_idx]
    y_test = [labels_all[i] for i in test_idx]

    alpha_text, alpha_image, grid = select_alphas(train_idx, y_train, text_emb, image_emb, scenarios, SEED)
    print(f"  Selected alpha_text={alpha_text}, alpha_image={alpha_image} "
          f"(OOF hard-subset acc {max(g['oof_hard_acc'] for g in grid):.4f})", flush=True)
    print(f"  alpha_text=0 selected {sum(1 for g in grid if g['alpha_text']==0 and g['oof_hard_acc']>=max(x['oof_hard_acc'] for x in grid)-1e-9)>0}", flush=True)

    text_test = fit_predict(text_emb[train_idx], y_train, text_emb[test_idx], "text")
    image_test = fit_predict(image_emb[train_idx], y_train, image_emb[test_idx], "image")

    text_ood = OODConfidenceRecalibrator()
    text_ood.fit(text_emb[train_idx], y_train)
    image_ood = OODConfidenceRecalibrator()
    image_ood.fit(image_emb[train_idx], y_train)
    text_trust = text_ood.trust_scores(text_emb[test_idx])
    image_trust = image_ood.trust_scores(image_emb[test_idx])

    text_test_tuned = apply_alpha(text_test, text_trust, alpha_text)
    image_test_tuned = apply_alpha(image_test, image_trust, alpha_image)

    ew_pred, cw_pred, tuned_pred = [], [], []
    for j in range(len(test_idx)):
        mods = [text_test[j], image_test[j]]
        mods_tuned = [text_test_tuned[j], image_test_tuned[j]]
        ew_pred.append(equal_weight_fusion(mods)["label"])
        cw_pred.append(confidence_weighted_fusion(mods)["label"])
        tuned_pred.append(confidence_weighted_fusion(mods_tuned)["label"])

    ew_correct = [t == p for t, p in zip(y_test, ew_pred)]
    cw_correct = [t == p for t, p in zip(y_test, cw_pred)]
    tuned_correct = [t == p for t, p in zip(y_test, tuned_pred)]
    deg_test = [len(scenarios[i]["degraded_modalities"]) for i in test_idx]
    hard = [j for j, d in enumerate(deg_test) if d == 1]

    def acc(correct, idxs=None):
        idxs = idxs if idxs is not None else range(len(correct))
        return round(sum(correct[j] for j in idxs) / len(idxs), 4)

    single_split = {
        "selected_alpha_text": alpha_text, "selected_alpha_image": alpha_image,
        "n_test": len(test_idx), "n_hard_subset": len(hard),
        "overall_accuracy": {"equal_weight": acc(ew_correct), "confidence_weighted": acc(cw_correct),
                              "confidence_weighted_tuned_recal": acc(tuned_correct)},
        "hard_subset_accuracy": {"equal_weight": acc(ew_correct, hard), "confidence_weighted": acc(cw_correct, hard),
                                  "confidence_weighted_tuned_recal": acc(tuned_correct, hard)},
        "hard_subset_mcnemar_tuned_vs_ew": mcnemar([tuned_correct[j] for j in hard], [ew_correct[j] for j in hard]),
        "hard_subset_mcnemar_tuned_vs_raw_cw": mcnemar([tuned_correct[j] for j in hard], [cw_correct[j] for j in hard]),
        "overall_bootstrap_ci": {"equal_weight": bootstrap_ci(ew_correct), "confidence_weighted": bootstrap_ci(cw_correct),
                                  "confidence_weighted_tuned_recal": bootstrap_ci(tuned_correct)},
    }
    print(json.dumps(single_split, indent=2), flush=True)

    # --- 30-seed robustness check; alpha re-selected inside each outer seed
    # from that seed's own train split only, so test data never leaks into
    # selection in any seed. ---
    print(f"\nRunning {N_SEEDS}-seed robustness check (alpha re-selected per seed)...", flush=True)
    per_seed = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        tr_idx, te_idx = idx[:n // 2], idx[n // 2:]
        y_tr = [labels_all[i] for i in tr_idx]
        y_te = [labels_all[i] for i in te_idx]

        a_text, a_image, _ = select_alphas(tr_idx, y_tr, text_emb, image_emb, scenarios, seed)

        t_te = fit_predict(text_emb[tr_idx], y_tr, text_emb[te_idx], "text")
        i_te = fit_predict(image_emb[tr_idx], y_tr, image_emb[te_idx], "image")
        t_ood = OODConfidenceRecalibrator()
        t_ood.fit(text_emb[tr_idx], y_tr)
        i_ood = OODConfidenceRecalibrator()
        i_ood.fit(image_emb[tr_idx], y_tr)
        t_trust = t_ood.trust_scores(text_emb[te_idx])
        i_trust = i_ood.trust_scores(image_emb[te_idx])
        t_te_tuned = apply_alpha(t_te, t_trust, a_text)
        i_te_tuned = apply_alpha(i_te, i_trust, a_image)

        ew_c, cw_c, tuned_c = [], [], []
        deg_te = [len(scenarios[i]["degraded_modalities"]) for i in te_idx]
        for j in range(len(te_idx)):
            mods = [t_te[j], i_te[j]]
            mods_tuned = [t_te_tuned[j], i_te_tuned[j]]
            ew_c.append(equal_weight_fusion(mods)["label"] == y_te[j])
            cw_c.append(confidence_weighted_fusion(mods)["label"] == y_te[j])
            tuned_c.append(confidence_weighted_fusion(mods_tuned)["label"] == y_te[j])
        hard_j = [j for j, d in enumerate(deg_te) if d == 1]
        ew_hard = [ew_c[j] for j in hard_j]
        cw_hard = [cw_c[j] for j in hard_j]
        tuned_hard = [tuned_c[j] for j in hard_j]
        per_seed.append({
            "seed": seed, "alpha_text": a_text, "alpha_image": a_image, "n_hard": len(hard_j),
            "ew_acc_hard": round(sum(ew_hard) / len(ew_hard), 4),
            "cw_acc_hard": round(sum(cw_hard) / len(cw_hard), 4),
            "tuned_acc_hard": round(sum(tuned_hard) / len(tuned_hard), 4),
        })
        print(f"  seed={seed:2d} alpha_text={a_text} alpha_image={a_image} "
              f"ew={per_seed[-1]['ew_acc_hard']:.4f} cw={per_seed[-1]['cw_acc_hard']:.4f} "
              f"tuned={per_seed[-1]['tuned_acc_hard']:.4f} "
              f"gap(tuned-ew)={per_seed[-1]['tuned_acc_hard']-per_seed[-1]['ew_acc_hard']:+.4f} "
              f"gap(tuned-cw)={per_seed[-1]['tuned_acc_hard']-per_seed[-1]['cw_acc_hard']:+.4f}", flush=True)

    gaps_vs_ew = [r["tuned_acc_hard"] - r["ew_acc_hard"] for r in per_seed]
    gaps_vs_cw = [r["tuned_acc_hard"] - r["cw_acc_hard"] for r in per_seed]
    n_favor_tuned_vs_ew = sum(1 for g in gaps_vs_ew if g > 0)
    n_favor_ew = sum(1 for g in gaps_vs_ew if g < 0)
    n_favor_tuned_vs_cw = sum(1 for g in gaps_vs_cw if g > 0)
    n_favor_raw_cw = sum(1 for g in gaps_vs_cw if g < 0)

    try:
        w_ew_stat, w_ew_p = scipy_stats.wilcoxon(gaps_vs_ew)
    except ValueError:
        w_ew_stat, w_ew_p = None, None
    try:
        w_cw_stat, w_cw_p = scipy_stats.wilcoxon(gaps_vs_cw)
    except ValueError:
        w_cw_stat, w_cw_p = None, None

    alpha_text_dist = {a: sum(1 for r in per_seed if r["alpha_text"] == a) for a in ALPHA_GRID}
    alpha_image_dist = {a: sum(1 for r in per_seed if r["alpha_image"] == a) for a in ALPHA_GRID}

    robustness = {
        "n_seeds": N_SEEDS,
        "per_seed": per_seed,
        "alpha_text_selection_distribution": alpha_text_dist,
        "alpha_image_selection_distribution": alpha_image_dist,
        "mean_gap_tuned_minus_ew": round(float(np.mean(gaps_vs_ew)), 4),
        "mean_gap_tuned_minus_raw_cw": round(float(np.mean(gaps_vs_cw)), 4),
        "n_seeds_tuned_beats_ew": n_favor_tuned_vs_ew,
        "n_seeds_ew_beats_tuned": n_favor_ew,
        "n_seeds_tuned_beats_raw_cw": n_favor_tuned_vs_cw,
        "n_seeds_raw_cw_beats_tuned": n_favor_raw_cw,
        "wilcoxon_tuned_vs_ew": {"statistic": float(w_ew_stat) if w_ew_stat is not None else None,
                                  "p_value": round(float(w_ew_p), 6) if w_ew_p is not None else None},
        "wilcoxon_tuned_vs_raw_cw": {"statistic": float(w_cw_stat) if w_cw_stat is not None else None,
                                      "p_value": round(float(w_cw_p), 6) if w_cw_p is not None else None},
    }

    out = {
        "meta": {
            "purpose": "Refines ood_recalibration_experiment.py's failed naive "
                       "(alpha=1 always) recalibration by selecting a per-modality "
                       "blend weight alpha via train-only 5-fold OOF cross-validation, "
                       "so each modality can independently decide how much (if any) "
                       "OOD-distance recalibration to trust.",
            "n_scenarios": n, "seed": SEED, "n_seeds_robustness": N_SEEDS,
            "alpha_grid": ALPHA_GRID, "n_inner_folds": N_INNER_FOLDS,
        },
        "single_split": single_split,
        "single_split_alpha_grid": grid,
        "robustness_multiseed": robustness,
    }
    with open(RESULTS_DIR / "ood_recalibration_tuned.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Single split (hard subset): EW={single_split['hard_subset_accuracy']['equal_weight']} "
          f"raw-CW={single_split['hard_subset_accuracy']['confidence_weighted']} "
          f"tuned-recal-CW={single_split['hard_subset_accuracy']['confidence_weighted_tuned_recal']} "
          f"(alpha_text={alpha_text}, alpha_image={alpha_image})")
    print(f"30-seed: tuned beats EW in {n_favor_tuned_vs_ew}/{N_SEEDS}, EW beats tuned in {n_favor_ew}/{N_SEEDS}, "
          f"mean gap {robustness['mean_gap_tuned_minus_ew']:+.4f}, Wilcoxon p={robustness['wilcoxon_tuned_vs_ew']['p_value']}")
    print(f"30-seed: tuned beats raw-CW in {n_favor_tuned_vs_cw}/{N_SEEDS}, raw-CW beats tuned in {n_favor_raw_cw}/{N_SEEDS}, "
          f"mean gap {robustness['mean_gap_tuned_minus_raw_cw']:+.4f}, Wilcoxon p={robustness['wilcoxon_tuned_vs_raw_cw']['p_value']}")
    print(f"alpha_text selection distribution across seeds: {alpha_text_dist}")
    print(f"alpha_image selection distribution across seeds: {alpha_image_dist}")
    print(f"\nWrote {RESULTS_DIR / 'ood_recalibration_tuned.json'}")


if __name__ == "__main__":
    main()
