"""Multi-seed robustness check for the core confidence-weighted vs.
equal-weight fusion comparison.

The main benchmark (run_benchmark.py) reports this comparison on a single
fixed train/test split (seed=42): p=0.0525 on the two-degraded-modality
subset, just above the conventional 0.05 threshold. That is not enough
evidence on its own that the effect is real rather than a property of one
particular split -- this script re-runs the same comparison across many
independent random splits of the same underlying 1500 scenarios and
reports the distribution of outcomes, not just a second point estimate.

Expensive model forward passes (MiniLM text embeddings, CLIP image
embeddings) are computed once for the full dataset up front, since they
don't depend on the train/test split; only the lightweight logistic-
regression classifiers are refit per seed. This is what makes running
dozens of seeds tractable.
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
from framework.telemetry_scorer import extract_features as telemetry_features
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion, LABELS

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
N_SEEDS = 30
FUSION_POWER = 3


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
    return b, c, float(p)


def main():
    print("Loading scenarios and computing embeddings once for the full dataset...", flush=True)
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    labels_all = [r["label"] for r in scenarios]

    text_encoder = TextScorer().encoder
    text_emb = text_encoder.encode([r["text"] for r in scenarios], show_progress_bar=False)

    image_scorer_for_embed = ImageScorer()
    image_emb = image_scorer_for_embed.embed([str(DATA_DIR / r["image_path"]) for r in scenarios])

    telemetry_feat = np.array([telemetry_features(r["telemetry"]) for r in scenarios])

    n = len(scenarios)
    print(f"Embeddings ready for {n} scenarios. Running {N_SEEDS} independent splits...", flush=True)

    per_seed_results = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_train = n // 2
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        text_res = fit_predict(text_emb[train_idx], [labels_all[i] for i in train_idx],
                                text_emb[test_idx], "text")
        image_res = fit_predict(image_emb[train_idx], [labels_all[i] for i in train_idx],
                                 image_emb[test_idx], "image")
        telemetry_res = fit_predict(telemetry_feat[train_idx], [labels_all[i] for i in train_idx],
                                     telemetry_feat[test_idx], "telemetry")

        y_test = [labels_all[i] for i in test_idx]
        deg_test = [len(scenarios[i]["degraded_modalities"]) for i in test_idx]

        cw_correct_hard, ew_correct_hard = [], []
        cw_correct_all, ew_correct_all = [], []
        for j in range(len(test_idx)):
            mods = [text_res[j], image_res[j], telemetry_res[j]]
            cw = confidence_weighted_fusion(mods, power=FUSION_POWER)["label"]
            ew = equal_weight_fusion(mods)["label"]
            cw_ok, ew_ok = cw == y_test[j], ew == y_test[j]
            cw_correct_all.append(cw_ok)
            ew_correct_all.append(ew_ok)
            if deg_test[j] == 2:
                cw_correct_hard.append(cw_ok)
                ew_correct_hard.append(ew_ok)

        b, c, p = mcnemar(cw_correct_hard, ew_correct_hard)
        per_seed_results.append({
            "seed": seed,
            "n_hard": len(cw_correct_hard),
            "cw_acc_hard": round(sum(cw_correct_hard) / len(cw_correct_hard), 4),
            "ew_acc_hard": round(sum(ew_correct_hard) / len(ew_correct_hard), 4),
            "cw_acc_overall": round(sum(cw_correct_all) / len(cw_correct_all), 4),
            "ew_acc_overall": round(sum(ew_correct_all) / len(ew_correct_all), 4),
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p, 6),
        })
        print(f"seed={seed:2d}  cw_hard={per_seed_results[-1]['cw_acc_hard']:.4f}  "
              f"ew_hard={per_seed_results[-1]['ew_acc_hard']:.4f}  "
              f"gap={per_seed_results[-1]['cw_acc_hard']-per_seed_results[-1]['ew_acc_hard']:+.4f}  "
              f"p={p:.4f}", flush=True)

    gaps = [r["cw_acc_hard"] - r["ew_acc_hard"] for r in per_seed_results]
    n_favor_cw = sum(1 for g in gaps if g > 0)
    n_favor_ew = sum(1 for g in gaps if g < 0)
    n_tied = sum(1 for g in gaps if g == 0)
    n_significant = sum(1 for r in per_seed_results if r["mcnemar_p"] < 0.05)

    # Pooled McNemar across all seeds' discordant pairs -- a higher-powered
    # single test on the combined evidence, not just counting how many
    # per-seed tests individually crossed 0.05.
    total_b = sum(r["mcnemar_b"] for r in per_seed_results)
    total_c = sum(r["mcnemar_c"] for r in per_seed_results)
    pooled_n = total_b + total_c
    pooled_p = scipy_stats.binomtest(min(total_b, total_c), pooled_n, 0.5).pvalue if pooled_n > 0 else 1.0

    # Wilcoxon signed-rank test on the per-seed accuracy gap itself:
    # treats each seed's gap as one paired observation, independent of the
    # per-seed McNemar significance.
    try:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(gaps)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = None, None

    summary = {
        "n_seeds": N_SEEDS,
        "fusion_power": FUSION_POWER,
        "per_seed": per_seed_results,
        "mean_gap_hard_subset": round(float(np.mean(gaps)), 4),
        "std_gap_hard_subset": round(float(np.std(gaps)), 4),
        "n_seeds_favoring_confidence_weighted": n_favor_cw,
        "n_seeds_favoring_equal_weight": n_favor_ew,
        "n_seeds_tied": n_tied,
        "n_seeds_individually_significant_p_lt_05": n_significant,
        "pooled_mcnemar_across_all_seeds": {
            "total_b_cw_right_ew_wrong": total_b,
            "total_c_cw_wrong_ew_right": total_c,
            "p_value": round(float(pooled_p), 8),
        },
        "wilcoxon_signed_rank_on_per_seed_gap": {
            "statistic": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
            "p_value": round(float(wilcoxon_p), 6) if wilcoxon_p is not None else None,
        },
    }

    with open(RESULTS_DIR / "robustness_multiseed.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary across {N_SEEDS} seeds ===")
    print(f"Mean accuracy gap (CW - EW) on hard subset: {summary['mean_gap_hard_subset']:+.4f} "
          f"(std {summary['std_gap_hard_subset']:.4f})")
    print(f"Seeds favoring confidence-weighted: {n_favor_cw}/{N_SEEDS}, "
          f"equal-weight: {n_favor_ew}/{N_SEEDS}, tied: {n_tied}/{N_SEEDS}")
    print(f"Seeds individually significant (p<0.05): {n_significant}/{N_SEEDS}")
    print(f"Pooled McNemar across all seeds' discordant pairs: "
          f"b={total_b} c={total_c} p={pooled_p:.8f}")
    print(f"Wilcoxon signed-rank on per-seed gap: p={summary['wilcoxon_signed_rank_on_per_seed_gap']['p_value']}")
    print(f"\nWrote {RESULTS_DIR / 'robustness_multiseed.json'}")


if __name__ == "__main__":
    main()
