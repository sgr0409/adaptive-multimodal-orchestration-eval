"""Oracle upper-bound check for degradation-aware fusion on CrisisMMD (real
data), the counterpart to experiments/oracle_ceiling_check.py's synthetic-
domain version. See that script's docstring for full motivation: the oracle
is given ground-truth knowledge of which modality is degraded on each
example (impossible at real deployment time) and hard-selects the other,
clean modality's own prediction. This establishes the true ceiling for the
entire class of degradation-aware reweighting mechanisms -- including all
three recalibration/gating attempts in ood_recalibration_experiment.py,
ood_recalibration_tuned.py, and rich_gating_experiment.py -- independent of
how well any of them estimates degradation.

Single split (seed=42) plus the standard 30-seed robustness check, on the
hard (exactly-one-modality-degraded) subset used throughout this domain's
results.
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

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.image_scorer import ImageScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion

SEED = 42
DATA_DIR = ROOT / "data_public_crisismmd"
RESULTS_DIR = ROOT / "experiments" / "public_crisismmd" / "results"
N_SEEDS = 30


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


def oracle_predict(text_res, image_res, degraded_modalities):
    if degraded_modalities == ["text"]:
        return image_res["label"]
    if degraded_modalities == ["image"]:
        return text_res["label"]
    return confidence_weighted_fusion([text_res, image_res])["label"]


def wilcoxon_safe(g):
    try:
        w = scipy_stats.wilcoxon(g)
        return round(float(w.statistic), 4), float(w.pvalue)
    except ValueError:
        return None, None


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

    per_seed = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        tr_idx, te_idx = idx[:n // 2], idx[n // 2:]
        y_tr = [labels_all[i] for i in tr_idx]
        y_te = [labels_all[i] for i in te_idx]

        t_te = fit_predict(text_emb[tr_idx], y_tr, text_emb[te_idx], "text")
        i_te = fit_predict(image_emb[tr_idx], y_tr, image_emb[te_idx], "image")
        deg_te = [scenarios[i]["degraded_modalities"] for i in te_idx]
        hard = [j for j, d in enumerate(deg_te) if len(d) == 1]

        ew_c = [equal_weight_fusion([t_te[j], i_te[j]])["label"] == y_te[j] for j in hard]
        cw_c = [confidence_weighted_fusion([t_te[j], i_te[j]])["label"] == y_te[j] for j in hard]
        or_c = [oracle_predict(t_te[j], i_te[j], deg_te[j]) == y_te[j] for j in hard]

        ew_h = sum(ew_c) / len(hard)
        cw_h = sum(cw_c) / len(hard)
        or_h = sum(or_c) / len(hard)
        per_seed.append({
            "seed": seed, "n_hard": len(hard),
            "ew_acc_hard": round(ew_h, 4), "cw_acc_hard": round(cw_h, 4), "oracle_acc_hard": round(or_h, 4),
        })
        print(f"  seed={seed:2d} n_hard={len(hard):4d} ew={ew_h:.4f} cw={cw_h:.4f} oracle={or_h:.4f} "
              f"gap(oracle-ew)={or_h-ew_h:+.4f} gap(oracle-cw)={or_h-cw_h:+.4f}", flush=True)

    gaps_ew = [r["oracle_acc_hard"] - r["ew_acc_hard"] for r in per_seed]
    gaps_cw = [r["oracle_acc_hard"] - r["cw_acc_hard"] for r in per_seed]
    w_ew_stat, w_ew_p = wilcoxon_safe(gaps_ew)
    w_cw_stat, w_cw_p = wilcoxon_safe(gaps_cw)

    summary = {
        "domain": "public_crisismmd", "n_seeds": N_SEEDS,
        "per_seed": per_seed,
        "mean_gap_oracle_minus_ew": round(float(np.mean(gaps_ew)), 4),
        "std_gap_oracle_minus_ew": round(float(np.std(gaps_ew)), 4),
        "mean_gap_oracle_minus_cw": round(float(np.mean(gaps_cw)), 4),
        "std_gap_oracle_minus_cw": round(float(np.std(gaps_cw)), 4),
        "n_seeds_oracle_beats_ew": sum(1 for g in gaps_ew if g > 0),
        "n_seeds_oracle_beats_cw": sum(1 for g in gaps_cw if g > 0),
        "wilcoxon_oracle_vs_ew": {"statistic": w_ew_stat, "p_value": w_ew_p},
        "wilcoxon_oracle_vs_cw": {"statistic": w_cw_stat, "p_value": w_cw_p},
    }
    with open(RESULTS_DIR / "oracle_ceiling_check.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Mean gap oracle-EW: {summary['mean_gap_oracle_minus_ew']:+.4f} "
          f"(oracle beats EW in {summary['n_seeds_oracle_beats_ew']}/{N_SEEDS} seeds, "
          f"Wilcoxon stat={w_ew_stat} p={w_ew_p})")
    print(f"Mean gap oracle-CW: {summary['mean_gap_oracle_minus_cw']:+.4f} "
          f"(oracle beats CW in {summary['n_seeds_oracle_beats_cw']}/{N_SEEDS} seeds, "
          f"Wilcoxon stat={w_cw_stat} p={w_cw_p})")
    print(f"\nWrote {RESULTS_DIR / 'oracle_ceiling_check.json'}")


if __name__ == "__main__":
    main()
