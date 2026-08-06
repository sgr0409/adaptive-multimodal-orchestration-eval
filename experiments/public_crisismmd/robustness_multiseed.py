"""Multi-seed robustness check for the public-CrisisMMD domain, mirroring
experiments/robustness_multiseed.py's methodology (see that file's
docstring). Needed here specifically because the single main-benchmark
split showed confidence-weighted fusion beating equal-weight by a small,
non-significant margin (p=0.26-0.29) -- this domain's own analogue of
domain two's single-split near-null result, and exactly the situation this
paper's own Practical Deployment Guidance warns not to take at face value
without a multi-seed check.

Embeddings for the full pool are computed once; only the lightweight
per-modality logistic-regression heads and the two fusion meta-learners are
refit per seed.
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
from framework.learned_fusion import StackingFusion, LearnedGatingFusion

DATA_DIR = ROOT / "data_public_crisismmd"
RESULTS_DIR = ROOT / "experiments" / "public_crisismmd" / "results"
N_SEEDS = 30
N_FOLDS = 5


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


def mcnemar(correct_a, correct_b):
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    p = scipy_stats.binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return b, c, float(p)


def main():
    print("Loading scenarios and computing embeddings once for the full pool...", flush=True)
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    labels_all = [r["label"] for r in scenarios]

    text_encoder = TextScorer().encoder
    text_emb = text_encoder.encode([r["text"] for r in scenarios], show_progress_bar=False)
    image_embedder = ImageScorer()
    image_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in scenarios])

    n = len(scenarios)
    print(f"Embeddings ready for {n} scenarios. Running {N_SEEDS} independent splits...", flush=True)

    per_seed_results = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_train = n // 2
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        y_train = [labels_all[i] for i in train_idx]
        y_test = [labels_all[i] for i in test_idx]

        text_test = fit_predict(text_emb[train_idx], y_train, text_emb[test_idx], "text")
        image_test = fit_predict(image_emb[train_idx], y_train, image_emb[test_idx], "image")

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        n_tr = len(train_idx)
        oof_text, oof_image = [None] * n_tr, [None] * n_tr
        train_idx_arr = np.array(train_idx)
        for fit_i, hold_i in kf.split(np.arange(n_tr)):
            y_fit = [labels_all[train_idx_arr[i]] for i in fit_i]
            t_preds = fit_predict(text_emb[train_idx_arr[fit_i]], y_fit, text_emb[train_idx_arr[hold_i]], "text")
            i_preds = fit_predict(image_emb[train_idx_arr[fit_i]], y_fit, image_emb[train_idx_arr[hold_i]], "image")
            for local_j, global_j in enumerate(hold_i):
                oof_text[global_j] = t_preds[local_j]
                oof_image[global_j] = i_preds[local_j]
        oof_mods = [[oof_text[i], oof_image[i]] for i in range(n_tr)]

        stacking = StackingFusion()
        stacking.fit(oof_mods, y_train)
        gating = LearnedGatingFusion(seed=seed)
        gating.fit(oof_mods, y_train)

        ew_correct, cw_correct, st_correct, gt_correct = [], [], [], []
        for j in range(len(test_idx)):
            mods = [text_test[j], image_test[j]]
            ew = equal_weight_fusion(mods)["label"]
            cw = confidence_weighted_fusion(mods)["label"]
            st = stacking.predict(mods)["label"]
            gt = gating.predict(mods)["label"]
            ew_correct.append(ew == y_test[j])
            cw_correct.append(cw == y_test[j])
            st_correct.append(st == y_test[j])
            gt_correct.append(gt == y_test[j])

        b, c, p = mcnemar(cw_correct, ew_correct)
        per_seed_results.append({
            "seed": seed,
            "ew_acc": round(sum(ew_correct) / len(ew_correct), 4),
            "cw_acc": round(sum(cw_correct) / len(cw_correct), 4),
            "stacking_acc": round(sum(st_correct) / len(st_correct), 4),
            "gating_acc": round(sum(gt_correct) / len(gt_correct), 4),
            "cw_vs_ew_mcnemar_b": b, "cw_vs_ew_mcnemar_c": c, "cw_vs_ew_mcnemar_p": round(p, 6),
        })
        print(f"seed={seed:2d}  ew={per_seed_results[-1]['ew_acc']:.4f}  cw={per_seed_results[-1]['cw_acc']:.4f}  "
              f"stacking={per_seed_results[-1]['stacking_acc']:.4f}  gating={per_seed_results[-1]['gating_acc']:.4f}  "
              f"gap(cw-ew)={per_seed_results[-1]['cw_acc']-per_seed_results[-1]['ew_acc']:+.4f}  p={p:.4f}", flush=True)

    gaps = [r["cw_acc"] - r["ew_acc"] for r in per_seed_results]
    n_favor_cw = sum(1 for g in gaps if g > 0)
    n_favor_ew = sum(1 for g in gaps if g < 0)
    n_tied = sum(1 for g in gaps if g == 0)
    n_significant = sum(1 for r in per_seed_results if r["cw_vs_ew_mcnemar_p"] < 0.05)

    total_b = sum(r["cw_vs_ew_mcnemar_b"] for r in per_seed_results)
    total_c = sum(r["cw_vs_ew_mcnemar_c"] for r in per_seed_results)
    pooled_n = total_b + total_c
    pooled_p = scipy_stats.binomtest(min(total_b, total_c), pooled_n, 0.5).pvalue if pooled_n > 0 else 1.0

    try:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(gaps)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = None, None

    stacking_gaps = [r["stacking_acc"] - r["cw_acc"] for r in per_seed_results]
    gating_gaps = [r["gating_acc"] - r["cw_acc"] for r in per_seed_results]

    summary = {
        "n_seeds": N_SEEDS,
        "per_seed": per_seed_results,
        "mean_gap_cw_minus_ew": round(float(np.mean(gaps)), 4),
        "std_gap_cw_minus_ew": round(float(np.std(gaps)), 4),
        "n_seeds_favoring_confidence_weighted": n_favor_cw,
        "n_seeds_favoring_equal_weight": n_favor_ew,
        "n_seeds_tied": n_tied,
        "n_seeds_individually_significant_p_lt_05": n_significant,
        "pooled_mcnemar_across_all_seeds": {"total_b": total_b, "total_c": total_c, "p_value": round(float(pooled_p), 8)},
        "wilcoxon_signed_rank_on_per_seed_gap": {
            "statistic": float(wilcoxon_stat) if wilcoxon_stat is not None else None,
            "p_value": round(float(wilcoxon_p), 6) if wilcoxon_p is not None else None,
        },
        "mean_gap_stacking_minus_cw": round(float(np.mean(stacking_gaps)), 4),
        "mean_gap_gating_minus_cw": round(float(np.mean(gating_gaps)), 4),
        "n_seeds_stacking_beats_cw": sum(1 for g in stacking_gaps if g > 0),
        "n_seeds_gating_beats_cw": sum(1 for g in gating_gaps if g > 0),
    }

    with open(RESULTS_DIR / "robustness_multiseed.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary across {N_SEEDS} seeds ===")
    print(f"Mean accuracy gap (CW - EW): {summary['mean_gap_cw_minus_ew']:+.4f} (std {summary['std_gap_cw_minus_ew']:.4f})")
    print(f"Seeds favoring CW: {n_favor_cw}/{N_SEEDS}, EW: {n_favor_ew}/{N_SEEDS}, tied: {n_tied}/{N_SEEDS}")
    print(f"Seeds individually significant (p<0.05): {n_significant}/{N_SEEDS}")
    print(f"Pooled McNemar: b={total_b} c={total_c} p={pooled_p:.8f}")
    print(f"Wilcoxon signed-rank on per-seed gap: p={summary['wilcoxon_signed_rank_on_per_seed_gap']['p_value']}")
    print(f"Stacking beats CW in {summary['n_seeds_stacking_beats_cw']}/{N_SEEDS} seeds "
          f"(mean gap {summary['mean_gap_stacking_minus_cw']:+.4f})")
    print(f"Gating beats CW in {summary['n_seeds_gating_beats_cw']}/{N_SEEDS} seeds "
          f"(mean gap {summary['mean_gap_gating_minus_cw']:+.4f})")
    print(f"\nWrote {RESULTS_DIR / 'robustness_multiseed.json'}")


if __name__ == "__main__":
    main()
