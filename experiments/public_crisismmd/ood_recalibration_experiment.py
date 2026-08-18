"""Tests whether OOD-aware confidence recalibration (framework/ood_calibration.py)
fixes the core negative finding of this paper's real-data domain: on CrisisMMD,
confidence-weighted fusion loses to equal-weight fusion because raw softmax
confidence is not reliability-correlated (degraded text scores HIGHER
confidence than clean text; see modality_confidence_reliability_check in
run_benchmark.py's output).

Methodology, mirroring run_benchmark.py and robustness_multiseed.py exactly
so results are directly comparable:

  1. Fit each modality's classifier on the train split (unchanged).
  2. ALSO fit an OODConfidenceRecalibrator per modality on the same train
     embeddings and labels (no new data, no new model).
  3. At test time, recalibrate each modality's raw confidence by its
     Mahalanobis-distance trust score, holding the predicted label and
     probability vector fixed -- only the scalar fed into confidence-weighted
     fusion's c_i^p / sum(c_j^p) rule changes.
  4. Compare equal-weight, raw confidence-weighted, and recalibrated
     confidence-weighted fusion on the same test split, single-split (seed=42)
     and a 30-seed robustness check.

Reports the honest outcome. If recalibration does not close the gap to
equal-weight fusion, that is reported as directly as the original reversal
was.
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
from framework.ood_calibration import OODConfidenceRecalibrator

SEED = 42
DATA_DIR = ROOT / "data_public_crisismmd"
RESULTS_DIR = ROOT / "experiments" / "public_crisismmd" / "results"
N_SEEDS = 30
N_BOOTSTRAP = 2000


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


def recalibrate(modality_results, embeddings, recalibrator):
    trust = recalibrator.trust_scores(embeddings)
    return [{**r, "confidence": float(r["confidence"] * t)} for r, t in zip(modality_results, trust)], trust


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
    print("Loading scenarios and computing embeddings once for the full pool...", flush=True)
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    labels_all = [r["label"] for r in scenarios]

    text_encoder = TextScorer().encoder
    text_emb = text_encoder.encode([r["text"] for r in scenarios], show_progress_bar=False)
    image_embedder = ImageScorer()
    image_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in scenarios])
    n = len(scenarios)
    print(f"Embeddings ready for {n} scenarios.", flush=True)

    # --- Single split (seed=42), matching run_benchmark.py's main table ---
    print("\nRunning main single-split comparison...", flush=True)
    rng = random.Random(SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = n // 2
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    y_train = [labels_all[i] for i in train_idx]
    y_test = [labels_all[i] for i in test_idx]

    text_test = fit_predict(text_emb[train_idx], y_train, text_emb[test_idx], "text")
    image_test = fit_predict(image_emb[train_idx], y_train, image_emb[test_idx], "image")

    text_ood = OODConfidenceRecalibrator()
    text_ood.fit(text_emb[train_idx], y_train)
    image_ood = OODConfidenceRecalibrator()
    image_ood.fit(image_emb[train_idx], y_train)

    text_test_recal, text_trust = recalibrate(text_test, text_emb[test_idx], text_ood)
    image_test_recal, image_trust = recalibrate(image_test, image_emb[test_idx], image_ood)

    ew_pred, cw_pred, cwr_pred = [], [], []
    for j in range(len(test_idx)):
        mods = [text_test[j], image_test[j]]
        mods_recal = [text_test_recal[j], image_test_recal[j]]
        ew_pred.append(equal_weight_fusion(mods)["label"])
        cw_pred.append(confidence_weighted_fusion(mods)["label"])
        cwr_pred.append(confidence_weighted_fusion(mods_recal)["label"])

    ew_correct = [t == p for t, p in zip(y_test, ew_pred)]
    cw_correct = [t == p for t, p in zip(y_test, cw_pred)]
    cwr_correct = [t == p for t, p in zip(y_test, cwr_pred)]
    deg_test = [len(scenarios[i]["degraded_modalities"]) for i in test_idx]
    hard = [j for j, d in enumerate(deg_test) if d == 1]

    def acc(correct, idxs=None):
        idxs = idxs if idxs is not None else range(len(correct))
        return round(sum(correct[j] for j in idxs) / len(idxs), 4)

    single_split = {
        "n_test": len(test_idx),
        "n_hard_subset": len(hard),
        "overall_accuracy": {"equal_weight": acc(ew_correct), "confidence_weighted": acc(cw_correct),
                              "confidence_weighted_recalibrated": acc(cwr_correct)},
        "hard_subset_accuracy": {"equal_weight": acc(ew_correct, hard), "confidence_weighted": acc(cw_correct, hard),
                                  "confidence_weighted_recalibrated": acc(cwr_correct, hard)},
        "hard_subset_mcnemar_recal_vs_ew": mcnemar([cwr_correct[j] for j in hard], [ew_correct[j] for j in hard]),
        "hard_subset_mcnemar_recal_vs_raw_cw": mcnemar([cwr_correct[j] for j in hard], [cw_correct[j] for j in hard]),
        "overall_bootstrap_ci": {"equal_weight": bootstrap_ci(ew_correct), "confidence_weighted": bootstrap_ci(cw_correct),
                                  "confidence_weighted_recalibrated": bootstrap_ci(cwr_correct)},
    }

    # Diagnostic: does the trust score actually separate degraded from clean
    # better than raw confidence did? This is the mechanism check, not just
    # the outcome check.
    trust_check = {}
    for mod, trust in [("text", text_trust), ("image", image_trust)]:
        deg = [trust[j] for j, i in enumerate(test_idx) if mod in scenarios[i]["degraded_modalities"]]
        clean = [trust[j] for j, i in enumerate(test_idx) if mod not in scenarios[i]["degraded_modalities"]]
        trust_check[mod] = {
            "mean_trust_when_degraded": round(float(np.mean(deg)), 4) if deg else None,
            "mean_trust_when_not_degraded": round(float(np.mean(clean)), 4) if clean else None,
            "n_degraded": len(deg),
        }
    single_split["trust_score_reliability_check"] = trust_check

    print(json.dumps(single_split, indent=2), flush=True)

    # --- 30-seed robustness check, same methodology as robustness_multiseed.py ---
    print(f"\nRunning {N_SEEDS}-seed robustness check...", flush=True)
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

        t_ood = OODConfidenceRecalibrator()
        t_ood.fit(text_emb[tr_idx], y_tr)
        i_ood = OODConfidenceRecalibrator()
        i_ood.fit(image_emb[tr_idx], y_tr)
        t_te_recal, _ = recalibrate(t_te, text_emb[te_idx], t_ood)
        i_te_recal, _ = recalibrate(i_te, image_emb[te_idx], i_ood)

        ew_c, cw_c, cwr_c = [], [], []
        deg_te = [len(scenarios[i]["degraded_modalities"]) for i in te_idx]
        for j in range(len(te_idx)):
            mods = [t_te[j], i_te[j]]
            mods_recal = [t_te_recal[j], i_te_recal[j]]
            ew_c.append(equal_weight_fusion(mods)["label"] == y_te[j])
            cw_c.append(confidence_weighted_fusion(mods)["label"] == y_te[j])
            cwr_c.append(confidence_weighted_fusion(mods_recal)["label"] == y_te[j])
        hard_j = [j for j, d in enumerate(deg_te) if d == 1]
        ew_hard = [ew_c[j] for j in hard_j]
        cw_hard = [cw_c[j] for j in hard_j]
        cwr_hard = [cwr_c[j] for j in hard_j]
        m_ew = mcnemar(cwr_hard, ew_hard)
        m_cw = mcnemar(cwr_hard, cw_hard)
        per_seed.append({
            "seed": seed, "n_hard": len(hard_j),
            "ew_acc_hard": round(sum(ew_hard) / len(ew_hard), 4),
            "cw_acc_hard": round(sum(cw_hard) / len(cw_hard), 4),
            "cwr_acc_hard": round(sum(cwr_hard) / len(cwr_hard), 4),
            "recal_vs_ew_mcnemar_p": m_ew["p_value"],
            "recal_vs_cw_mcnemar_p": m_cw["p_value"],
        })
        print(f"  seed={seed:2d} ew={per_seed[-1]['ew_acc_hard']:.4f} "
              f"cw={per_seed[-1]['cw_acc_hard']:.4f} cwr={per_seed[-1]['cwr_acc_hard']:.4f} "
              f"gap(cwr-ew)={per_seed[-1]['cwr_acc_hard']-per_seed[-1]['ew_acc_hard']:+.4f} "
              f"gap(cwr-cw)={per_seed[-1]['cwr_acc_hard']-per_seed[-1]['cw_acc_hard']:+.4f}", flush=True)

    gaps_vs_ew = [r["cwr_acc_hard"] - r["ew_acc_hard"] for r in per_seed]
    gaps_vs_cw = [r["cwr_acc_hard"] - r["cw_acc_hard"] for r in per_seed]
    n_favor_recal_vs_ew = sum(1 for g in gaps_vs_ew if g > 0)
    n_favor_ew = sum(1 for g in gaps_vs_ew if g < 0)
    n_favor_recal_vs_cw = sum(1 for g in gaps_vs_cw if g > 0)
    n_favor_raw_cw = sum(1 for g in gaps_vs_cw if g < 0)

    try:
        w_ew_stat, w_ew_p = scipy_stats.wilcoxon(gaps_vs_ew)
    except ValueError:
        w_ew_stat, w_ew_p = None, None
    try:
        w_cw_stat, w_cw_p = scipy_stats.wilcoxon(gaps_vs_cw)
    except ValueError:
        w_cw_stat, w_cw_p = None, None

    robustness = {
        "n_seeds": N_SEEDS,
        "per_seed": per_seed,
        "mean_gap_recal_minus_ew": round(float(np.mean(gaps_vs_ew)), 4),
        "mean_gap_recal_minus_raw_cw": round(float(np.mean(gaps_vs_cw)), 4),
        "n_seeds_recal_beats_ew": n_favor_recal_vs_ew,
        "n_seeds_ew_beats_recal": n_favor_ew,
        "n_seeds_recal_beats_raw_cw": n_favor_recal_vs_cw,
        "n_seeds_raw_cw_beats_recal": n_favor_raw_cw,
        "wilcoxon_recal_vs_ew": {"statistic": float(w_ew_stat) if w_ew_stat is not None else None,
                                  "p_value": round(float(w_ew_p), 6) if w_ew_p is not None else None},
        "wilcoxon_recal_vs_raw_cw": {"statistic": float(w_cw_stat) if w_cw_stat is not None else None,
                                      "p_value": round(float(w_cw_p), 6) if w_cw_p is not None else None},
    }

    out = {
        "meta": {
            "purpose": "Tests whether OOD-Mahalanobis confidence recalibration "
                       "(framework/ood_calibration.py) closes the gap between "
                       "confidence-weighted and equal-weight fusion on real "
                       "(CrisisMMD) data, targeting the diagnosed root cause "
                       "(miscalibrated raw softmax confidence) rather than the "
                       "combination rule.",
            "n_scenarios": n, "seed": SEED, "n_seeds_robustness": N_SEEDS,
        },
        "single_split": single_split,
        "robustness_multiseed": robustness,
    }
    with open(RESULTS_DIR / "ood_recalibration_experiment.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Single split (hard subset): EW={single_split['hard_subset_accuracy']['equal_weight']} "
          f"raw-CW={single_split['hard_subset_accuracy']['confidence_weighted']} "
          f"recal-CW={single_split['hard_subset_accuracy']['confidence_weighted_recalibrated']}")
    print(f"Trust score reliability check: {json.dumps(trust_check)}")
    print(f"30-seed: recal-CW beats EW in {n_favor_recal_vs_ew}/{N_SEEDS}, EW beats recal-CW in {n_favor_ew}/{N_SEEDS}, "
          f"mean gap {robustness['mean_gap_recal_minus_ew']:+.4f}, Wilcoxon p={robustness['wilcoxon_recal_vs_ew']['p_value']}")
    print(f"30-seed: recal-CW beats raw-CW in {n_favor_recal_vs_cw}/{N_SEEDS}, raw-CW beats recal-CW in {n_favor_raw_cw}/{N_SEEDS}, "
          f"mean gap {robustness['mean_gap_recal_minus_raw_cw']:+.4f}, Wilcoxon p={robustness['wilcoxon_recal_vs_raw_cw']['p_value']}")
    print(f"\nWrote {RESULTS_DIR / 'ood_recalibration_experiment.json'}")


if __name__ == "__main__":
    main()
