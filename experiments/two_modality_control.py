"""Control experiment answering a specific question raised about the
public-CrisisMMD validation domain (experiments/public_crisismmd/): is the
equal-weight-beats-confidence-weighted reversal there caused by real data
(distribution shift), or is it partly an artifact of CrisisMMD having only
two modalities (text, image) instead of the synthetic domains' three
(text, image, telemetry)?

This isolates that confound directly rather than arguing about it: it
re-runs the identical confidence-weighted-vs-equal-weight-fusion
comparison on domain one's own SYNTHETIC data, restricted to exactly the
same two modalities (text, image; telemetry dropped) and the same
degradation structure CrisisMMD uses (at most one of the two modalities
degraded per scenario, mirroring CrisisMMD's "one of two degraded, one
clean" design rather than the three-modality domains' "two of three
degraded" condition). Everything else -- the data-generation process, the
text/image scorers, the fusion mechanism -- is unchanged and still
synthetic.

If confidence-weighted fusion still beats equal-weight fusion under this
two-modality restriction on synthetic data, that is direct evidence the
CrisisMMD reversal is attributable to real vs. synthetic data, not to the
modality-count change. If it does not, that is equally informative: it
would mean modality count is itself a contributing factor, not just
distribution shift alone.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.image_scorer import ImageScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion

SEED = 42
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
N_SEEDS = 30
N_BOOTSTRAP = 2000


def fit_predict(X_train, y_train, X_test, modality_name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    out = []
    for p in probs:
        best = int(np.argmax(p))
        out.append({"modality": modality_name, "label": classes[best], "confidence": float(p[best]),
                    "probs": {c: float(v) for c, v in zip(classes, p)}})
    return out


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


def main():
    print("Loading domain-one scenarios and filtering to the text/image, "
          "at-most-one-degraded subset...", flush=True)
    all_scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    scenarios = [r for r in all_scenarios
                 if set(r["degraded_modalities"]) <= {"text", "image"} and len(r["degraded_modalities"]) <= 1]
    print(f"  {len(scenarios)}/{len(all_scenarios)} scenarios kept "
          f"(0 or 1 of {{text, image}} degraded, telemetry-degraded scenarios excluded)", flush=True)

    print("Embedding the full filtered pool once...", flush=True)
    text_encoder = TextScorer().encoder
    text_emb = text_encoder.encode([r["text"] for r in scenarios], show_progress_bar=False)
    image_embedder = ImageScorer()
    image_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in scenarios])
    labels_all = [r["label"] for r in scenarios]
    n = len(scenarios)

    # --- Single split (seed=42), matching the paper's main-table methodology ---
    print("Running main single-split benchmark...", flush=True)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = n // 2
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    y_train = [labels_all[i] for i in train_idx]
    y_test = [labels_all[i] for i in test_idx]

    text_test = fit_predict(text_emb[train_idx], y_train, text_emb[test_idx], "text")
    image_test = fit_predict(image_emb[train_idx], y_train, image_emb[test_idx], "image")

    ew_pred, cw_pred = [], []
    for j in range(len(test_idx)):
        mods = [text_test[j], image_test[j]]
        ew_pred.append(equal_weight_fusion(mods)["label"])
        cw_pred.append(confidence_weighted_fusion(mods)["label"])

    ew_correct = [t == p for t, p in zip(y_test, ew_pred)]
    cw_correct = [t == p for t, p in zip(y_test, cw_pred)]
    deg_test = [len(scenarios[i]["degraded_modalities"]) for i in test_idx]
    hard = [j for j, d in enumerate(deg_test) if d == 1]

    single_split = {
        "n_test": len(test_idx),
        "n_hard_subset": len(hard),
        "overall_accuracy": {
            "equal_weight": round(sum(ew_correct) / len(ew_correct), 4),
            "confidence_weighted": round(sum(cw_correct) / len(cw_correct), 4),
        },
        "hard_subset_accuracy": {
            "equal_weight": round(sum(ew_correct[j] for j in hard) / len(hard), 4),
            "confidence_weighted": round(sum(cw_correct[j] for j in hard) / len(hard), 4),
        },
        "overall_mcnemar": mcnemar(cw_correct, ew_correct),
        "hard_subset_mcnemar": mcnemar([cw_correct[j] for j in hard], [ew_correct[j] for j in hard]),
        "overall_bootstrap_ci": {"equal_weight": bootstrap_ci(ew_correct), "confidence_weighted": bootstrap_ci(cw_correct)},
    }
    print(json.dumps(single_split, indent=2), flush=True)

    # --- 30-seed robustness check, same methodology as the rest of the paper ---
    print(f"\nRunning {N_SEEDS}-seed robustness check...", flush=True)
    per_seed = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_tr = n // 2
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
        y_tr = [labels_all[i] for i in tr_idx]
        y_te = [labels_all[i] for i in te_idx]

        t_te = fit_predict(text_emb[tr_idx], y_tr, text_emb[te_idx], "text")
        i_te = fit_predict(image_emb[tr_idx], y_tr, image_emb[te_idx], "image")

        ew_c, cw_c = [], []
        deg_te = [len(scenarios[i]["degraded_modalities"]) for i in te_idx]
        for j in range(len(te_idx)):
            mods = [t_te[j], i_te[j]]
            ew_c.append(equal_weight_fusion(mods)["label"] == y_te[j])
            cw_c.append(confidence_weighted_fusion(mods)["label"] == y_te[j])
        hard_j = [j for j, d in enumerate(deg_te) if d == 1]
        ew_hard = [ew_c[j] for j in hard_j]
        cw_hard = [cw_c[j] for j in hard_j]
        m = mcnemar(cw_hard, ew_hard)
        per_seed.append({
            "seed": seed, "n_hard": len(hard_j),
            "ew_acc_hard": round(sum(ew_hard) / len(ew_hard), 4),
            "cw_acc_hard": round(sum(cw_hard) / len(cw_hard), 4),
            "mcnemar_b": m["b"], "mcnemar_c": m["c"], "mcnemar_p": m["p_value"],
        })
        print(f"  seed={seed:2d} ew={per_seed[-1]['ew_acc_hard']:.4f} "
              f"cw={per_seed[-1]['cw_acc_hard']:.4f} "
              f"gap={per_seed[-1]['cw_acc_hard']-per_seed[-1]['ew_acc_hard']:+.4f} "
              f"p={m['p_value']:.4f}", flush=True)

    gaps = [r["cw_acc_hard"] - r["ew_acc_hard"] for r in per_seed]
    n_favor_cw = sum(1 for g in gaps if g > 0)
    n_favor_ew = sum(1 for g in gaps if g < 0)
    n_tied = sum(1 for g in gaps if g == 0)
    total_b = sum(r["mcnemar_b"] for r in per_seed)
    total_c = sum(r["mcnemar_c"] for r in per_seed)
    pooled_n = total_b + total_c
    pooled_p = scipy_stats.binomtest(min(total_b, total_c), pooled_n, 0.5).pvalue if pooled_n > 0 else 1.0
    try:
        wstat, wp = scipy_stats.wilcoxon(gaps)
    except ValueError:
        wstat, wp = None, None

    robustness = {
        "n_seeds": N_SEEDS,
        "per_seed": per_seed,
        "mean_gap_hard_subset": round(float(np.mean(gaps)), 4),
        "std_gap_hard_subset": round(float(np.std(gaps)), 4),
        "n_seeds_favoring_confidence_weighted": n_favor_cw,
        "n_seeds_favoring_equal_weight": n_favor_ew,
        "n_seeds_tied": n_tied,
        "pooled_mcnemar": {"b": total_b, "c": total_c, "p_value": round(float(pooled_p), 8)},
        "wilcoxon": {"statistic": float(wstat) if wstat is not None else None,
                     "p_value": round(float(wp), 6) if wp is not None else None},
    }

    out = {
        "meta": {
            "purpose": "Isolates whether the CrisisMMD reversal is caused by real-vs-synthetic "
                       "data or by the modality-count reduction (3->2), by re-running the "
                       "confidence-weighted-vs-equal-weight comparison on domain one's own "
                       "SYNTHETIC data restricted to text+image only, with CrisisMMD's "
                       "at-most-one-of-two-degraded structure.",
            "n_scenarios_kept": n, "n_scenarios_total_domain_one": len(all_scenarios),
            "seed": SEED, "n_seeds_robustness": N_SEEDS,
        },
        "single_split": single_split,
        "robustness_multiseed": robustness,
    }
    with open(RESULTS_DIR / "two_modality_control.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Single split: EW={single_split['hard_subset_accuracy']['equal_weight']} "
          f"CW={single_split['hard_subset_accuracy']['confidence_weighted']} "
          f"McNemar={single_split['hard_subset_mcnemar']}")
    print(f"30-seed: CW favored in {n_favor_cw}/{N_SEEDS}, EW in {n_favor_ew}/{N_SEEDS}, "
          f"mean gap {robustness['mean_gap_hard_subset']:+.4f}, "
          f"Wilcoxon p={robustness['wilcoxon']['p_value']}")
    print(f"\nWrote {RESULTS_DIR / 'two_modality_control.json'}")


if __name__ == "__main__":
    main()
