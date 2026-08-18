"""Tests RichGatingFusion (framework/learned_fusion.py) -- a learned gate
over [confidence, OOD-distance trust, entropy, cross-modality disagreement]
per modality -- against equal-weight fusion, raw confidence-weighted
fusion, and the two existing learned-fusion baselines (stacking, plain
confidence-gating) on the public-CrisisMMD domain.

Motivation: two hand-designed OOD-recalibration mechanisms
(ood_recalibration_experiment.py, ood_recalibration_tuned.py) both failed --
neither raw confidence nor OOD-distance trust is individually a reliable
per-example reliability signal in this domain (verified directly: when text
is degraded, raw confidence already favors image correctly but OOD trust
flips it backwards; when image is degraded, both signals agree on the wrong
answer). But a real, exploitable reliability gap exists (15.6-point accuracy
gap between modalities when text specifically is degraded), so the signal
is there, just not extractable via a hand-written formula over one feature.
This tests whether a trained gate, given several imperfect signals at once,
can do what no single hand-designed formula could.

Same methodology as run_benchmark.py / robustness_multiseed.py: out-of-fold
predictions (5-fold) on the train split train the meta-learners, base
scorers deployed at test time are fit on the full train split, single-split
(seed=42) plus a 30-seed robustness check. Reports the honest result.
"""
import json
import random
import sys
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
from framework.learned_fusion import StackingFusion, LearnedGatingFusion, RichGatingFusion
from framework.ood_calibration import OODConfidenceRecalibrator

SEED = 42
DATA_DIR = ROOT / "data_public_crisismmd"
RESULTS_DIR = ROOT / "experiments" / "public_crisismmd" / "results"
N_SEEDS = 30
N_FOLDS = 5
N_BOOTSTRAP = 2000
METHODS = ["equal_weight", "confidence_weighted", "stacking", "gating", "rich_gating"]


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


def attach_trust(modality_results, embeddings, recalibrator):
    trust = recalibrator.trust_scores(embeddings)
    return [{**r, "ood_trust": float(t)} for r, t in zip(modality_results, trust)]


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


def build_oof(train_idx, y_train, text_emb, image_emb, seed, n_folds=N_FOLDS):
    """Out-of-fold text/image predictions + ood_trust on the train split,
    for fitting the meta-learners without label leakage."""
    n_tr = len(train_idx)
    train_idx_arr = np.array(train_idx)
    oof_text, oof_image = [None] * n_tr, [None] * n_tr

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_i, hold_i in kf.split(np.arange(n_tr)):
        fit_global = train_idx_arr[fit_i]
        hold_global = train_idx_arr[hold_i]
        y_fit = [y_train[i] for i in fit_i]

        t_preds = fit_predict(text_emb[fit_global], y_fit, text_emb[hold_global], "text")
        i_preds = fit_predict(image_emb[fit_global], y_fit, image_emb[hold_global], "image")

        t_ood = OODConfidenceRecalibrator(); t_ood.fit(text_emb[fit_global], y_fit)
        i_ood = OODConfidenceRecalibrator(); i_ood.fit(image_emb[fit_global], y_fit)
        t_preds = attach_trust(t_preds, text_emb[hold_global], t_ood)
        i_preds = attach_trust(i_preds, image_emb[hold_global], i_ood)

        for local_j, global_j in enumerate(hold_i):
            oof_text[global_j] = t_preds[local_j]
            oof_image[global_j] = i_preds[local_j]

    return [[oof_text[i], oof_image[i]] for i in range(n_tr)]


def run_split(train_idx, test_idx, labels_all, scenarios, text_emb, image_emb, seed):
    y_train = [labels_all[i] for i in train_idx]
    y_test = [labels_all[i] for i in test_idx]

    text_test = fit_predict(text_emb[train_idx], y_train, text_emb[test_idx], "text")
    image_test = fit_predict(image_emb[train_idx], y_train, image_emb[test_idx], "image")
    text_ood = OODConfidenceRecalibrator(); text_ood.fit(text_emb[train_idx], y_train)
    image_ood = OODConfidenceRecalibrator(); image_ood.fit(image_emb[train_idx], y_train)
    text_test = attach_trust(text_test, text_emb[test_idx], text_ood)
    image_test = attach_trust(image_test, image_emb[test_idx], image_ood)

    oof_mods = build_oof(train_idx, y_train, text_emb, image_emb, seed)
    stacking = StackingFusion(); stacking.fit(oof_mods, y_train)
    gating = LearnedGatingFusion(seed=seed); gating.fit(oof_mods, y_train)
    rich = RichGatingFusion(seed=seed); rich.fit(oof_mods, y_train)

    correct = {m: [] for m in METHODS}
    for j in range(len(test_idx)):
        mods = [text_test[j], image_test[j]]
        preds = {
            "equal_weight": equal_weight_fusion(mods)["label"],
            "confidence_weighted": confidence_weighted_fusion(mods)["label"],
            "stacking": stacking.predict(mods)["label"],
            "gating": gating.predict(mods)["label"],
            "rich_gating": rich.predict(mods)["label"],
        }
        for m in METHODS:
            correct[m].append(preds[m] == y_test[j])

    deg_test = [len(scenarios[i]["degraded_modalities"]) for i in test_idx]
    hard = [j for j, d in enumerate(deg_test) if d == 1]
    return correct, hard


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
    print("\nRunning main single-split comparison...", flush=True)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = n // 2
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    correct, hard = run_split(np.array(train_idx), np.array(test_idx), labels_all, scenarios, text_emb, image_emb, SEED)

    def acc(key, idxs=None):
        c = correct[key]
        idxs = idxs if idxs is not None else range(len(c))
        return round(sum(c[j] for j in idxs) / len(idxs), 4)

    single_split = {
        "n_test": len(test_idx), "n_hard_subset": len(hard),
        "overall_accuracy": {m: acc(m) for m in METHODS},
        "hard_subset_accuracy": {m: acc(m, hard) for m in METHODS},
        "hard_subset_mcnemar_rich_gating_vs_equal_weight": mcnemar(
            [correct["rich_gating"][j] for j in hard], [correct["equal_weight"][j] for j in hard]),
        "hard_subset_mcnemar_rich_gating_vs_confidence_weighted": mcnemar(
            [correct["rich_gating"][j] for j in hard], [correct["confidence_weighted"][j] for j in hard]),
        "hard_subset_mcnemar_rich_gating_vs_gating": mcnemar(
            [correct["rich_gating"][j] for j in hard], [correct["gating"][j] for j in hard]),
        "hard_subset_mcnemar_rich_gating_vs_stacking": mcnemar(
            [correct["rich_gating"][j] for j in hard], [correct["stacking"][j] for j in hard]),
        "overall_bootstrap_ci": {m: bootstrap_ci(correct[m]) for m in METHODS},
    }
    print(json.dumps(single_split, indent=2), flush=True)

    # --- 30-seed robustness check ---
    print(f"\nRunning {N_SEEDS}-seed robustness check...", flush=True)
    per_seed = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        tr_idx, te_idx = idx[:n // 2], idx[n // 2:]
        c, h = run_split(np.array(tr_idx), np.array(te_idx), labels_all, scenarios, text_emb, image_emb, seed)
        row = {"seed": seed, "n_hard": len(h)}
        for m in METHODS:
            row[f"{m}_acc_hard"] = round(sum(c[m][j] for j in h) / len(h), 4)
        per_seed.append(row)
        print(f"  seed={seed:2d} " + " ".join(f"{m}={row[f'{m}_acc_hard']:.4f}" for m in METHODS) +
              f"  gap(rich-ew)={row['rich_gating_acc_hard']-row['equal_weight_acc_hard']:+.4f}"
              f"  gap(rich-cw)={row['rich_gating_acc_hard']-row['confidence_weighted_acc_hard']:+.4f}"
              f"  gap(rich-gating)={row['rich_gating_acc_hard']-row['gating_acc_hard']:+.4f}", flush=True)

    def gaps(vs):
        return [r["rich_gating_acc_hard"] - r[f"{vs}_acc_hard"] for r in per_seed]

    robustness = {"n_seeds": N_SEEDS, "per_seed": per_seed}
    for vs in ["equal_weight", "confidence_weighted", "stacking", "gating"]:
        g = gaps(vs)
        try:
            wstat, wp = scipy_stats.wilcoxon(g)
        except ValueError:
            wstat, wp = None, None
        robustness[f"vs_{vs}"] = {
            "mean_gap": round(float(np.mean(g)), 4),
            "n_seeds_rich_gating_wins": sum(1 for x in g if x > 0),
            "n_seeds_loses": sum(1 for x in g if x < 0),
            "n_seeds_tied": sum(1 for x in g if x == 0),
            "wilcoxon_p": round(float(wp), 6) if wp is not None else None,
        }

    out = {
        "meta": {
            "purpose": "Tests RichGatingFusion (learned gate over confidence, "
                       "OOD-distance trust, entropy, cross-modality disagreement) "
                       "against equal-weight, raw confidence-weighted, and existing "
                       "learned-fusion baselines on real (CrisisMMD) data, after two "
                       "hand-designed OOD-recalibration formulas both failed.",
            "n_scenarios": n, "seed": SEED, "n_seeds_robustness": N_SEEDS, "methods": METHODS,
        },
        "single_split": single_split,
        "robustness_multiseed": robustness,
    }
    with open(RESULTS_DIR / "rich_gating_experiment.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Single split (hard subset): " + " ".join(f"{m}={single_split['hard_subset_accuracy'][m]}" for m in METHODS))
    for vs in ["equal_weight", "confidence_weighted", "stacking", "gating"]:
        r = robustness[f"vs_{vs}"]
        print(f"30-seed rich_gating vs {vs}: wins {r['n_seeds_rich_gating_wins']}/{N_SEEDS}, "
              f"loses {r['n_seeds_loses']}/{N_SEEDS}, mean gap {r['mean_gap']:+.4f}, Wilcoxon p={r['wilcoxon_p']}")
    print(f"\nWrote {RESULTS_DIR / 'rich_gating_experiment.json'}")


if __name__ == "__main__":
    main()
