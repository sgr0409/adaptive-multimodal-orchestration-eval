"""Ground-truth degradation-exclusion diagnostic on both generated domains.

The policy discards every marked modality. It diagnoses mandatory hard
exclusion, not soft reweighting or arbitrary learned fusion. A negative
gap on naturally sourced data therefore shows that corruption status is
not equivalent to correctness; it does not prove that all adaptive fusion
mechanisms must fail.

This script asks the natural follow-up: does the same exclusion pattern hold
on the synthetic domains, where confidence-weighted fusion's advantage over
equal-weight is real and well-established (30-seed robustness check,
sec:results)? If the oracle clearly beats both baselines on synthetic data
but not on real data, that is a clean, mechanistic explanation for exactly
why the synthetic-domain finding fails to transfer: on synthetic data,
"degraded" was constructed to reliably mean "less predictive," so
degradation-awareness has real headroom to exploit; on real data it
doesn't, because real degradation does not cleanly imply incorrectness.

Runs identically on data/ (domain one, 3 modalities: text/image/telemetry)
and data_it_incidents/ (domain two, same 3 modalities, different label set
and telemetry channel names -- both already handled generically by
framework/fusion.py and framework/telemetry_scorer.py, see their
docstrings). Oracle weight rule: zero weight to every degraded modality,
equal weight split across the remaining clean modalities (equal weight
across all three if all three are degraded, since there is no oracle
information left to use in that case).
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
from framework.telemetry_scorer import TelemetryScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion, _weighted_sum

N_SEEDS = 30
MODALITIES = ["text", "image", "telemetry"]


def oracle_fusion(modality_results, degraded_modalities):
    n = len(modality_results)
    clean_idx = [k for k in range(n) if MODALITIES[k] not in degraded_modalities]
    if not clean_idx:
        weights = [1 / n] * n
    else:
        weights = [1 / len(clean_idx) if k in clean_idx else 0.0 for k in range(n)]
    return _weighted_sum(modality_results, weights)


def run_domain(domain_name, data_dir):
    print(f"\n{'='*70}\nDomain: {domain_name} ({data_dir})\n{'='*70}", flush=True)
    scenarios = [json.loads(l) for l in open(data_dir / "scenarios.jsonl")]
    labels_all = [r["label"] for r in scenarios]
    n = len(scenarios)

    print("Computing embeddings/features once for the full pool...", flush=True)
    text_encoder = TextScorer().encoder
    text_emb = text_encoder.encode([r["text"] for r in scenarios], show_progress_bar=False)
    image_embedder = ImageScorer()
    image_emb = image_embedder.embed([str(data_dir / r["image_path"]) for r in scenarios])
    from framework.telemetry_scorer import extract_features
    telemetry_feat = np.array([extract_features(r["telemetry"]) for r in scenarios])
    print(f"Ready: {n} scenarios.", flush=True)

    def fit_predict_generic(X_train, y_train, X_test, name):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        classes = list(clf.classes_)
        out = []
        for p in probs:
            best = int(np.argmax(p))
            out.append({"modality": name, "label": classes[best], "confidence": float(p[best]),
                        "probs": {c: float(v) for c, v in zip(classes, p)}})
        return out

    per_seed = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_train = n // 2
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        y_train = [labels_all[i] for i in train_idx]
        y_test = [labels_all[i] for i in test_idx]

        text_test = fit_predict_generic(text_emb[train_idx], y_train, text_emb[test_idx], "text")
        image_test = fit_predict_generic(image_emb[train_idx], y_train, image_emb[test_idx], "image")
        telem_test = fit_predict_generic(telemetry_feat[train_idx], y_train, telemetry_feat[test_idx], "telemetry")

        deg_test = [scenarios[i]["degraded_modalities"] for i in test_idx]
        hard = [j for j, d in enumerate(deg_test) if len(d) == 2]  # this domain's established hard case

        ew_c, cw_c, or_c = [], [], []
        for j in range(len(test_idx)):
            mods = [text_test[j], image_test[j], telem_test[j]]
            ew_c.append(equal_weight_fusion(mods)["label"] == y_test[j])
            cw_c.append(confidence_weighted_fusion(mods)["label"] == y_test[j])
            or_c.append(oracle_fusion(mods, deg_test[j])["label"] == y_test[j])

        if hard:
            ew_h = sum(ew_c[j] for j in hard) / len(hard)
            cw_h = sum(cw_c[j] for j in hard) / len(hard)
            or_h = sum(or_c[j] for j in hard) / len(hard)
        else:
            ew_h = cw_h = or_h = None

        row = {"seed": seed, "n_hard": len(hard),
               "ew_acc_hard": round(ew_h, 4) if ew_h is not None else None,
               "cw_acc_hard": round(cw_h, 4) if cw_h is not None else None,
               "oracle_acc_hard": round(or_h, 4) if or_h is not None else None}
        per_seed.append(row)
        if hard:
            print(f"  seed={seed:2d} n_hard={len(hard):3d} ew={ew_h:.4f} cw={cw_h:.4f} oracle={or_h:.4f} "
                  f"gap(oracle-ew)={or_h-ew_h:+.4f} gap(oracle-cw)={or_h-cw_h:+.4f}", flush=True)
        else:
            print(f"  seed={seed:2d} n_hard=0 (skipped)", flush=True)

    valid = [r for r in per_seed if r["oracle_acc_hard"] is not None]
    gaps_oracle_ew = [r["oracle_acc_hard"] - r["ew_acc_hard"] for r in valid]
    gaps_oracle_cw = [r["oracle_acc_hard"] - r["cw_acc_hard"] for r in valid]

    def wilcoxon_safe(g):
        try:
            w = scipy_stats.wilcoxon(g)
            return round(float(w.statistic), 4), round(float(w.pvalue), 8)
        except ValueError:
            return None, None

    w_ew_stat, w_ew_p = wilcoxon_safe(gaps_oracle_ew)
    w_cw_stat, w_cw_p = wilcoxon_safe(gaps_oracle_cw)

    summary = {
        "domain": domain_name, "n_seeds": len(valid),
        "per_seed": per_seed,
        "mean_gap_oracle_minus_ew": round(float(np.mean(gaps_oracle_ew)), 4),
        "mean_gap_oracle_minus_cw": round(float(np.mean(gaps_oracle_cw)), 4),
        "n_seeds_oracle_beats_ew": sum(1 for g in gaps_oracle_ew if g > 0),
        "n_seeds_oracle_beats_cw": sum(1 for g in gaps_oracle_cw if g > 0),
        "wilcoxon_oracle_vs_ew": {"statistic": w_ew_stat, "p_value": w_ew_p},
        "wilcoxon_oracle_vs_cw": {"statistic": w_cw_stat, "p_value": w_cw_p},
    }
    print(f"\n--- {domain_name} summary ---")
    print(f"Mean gap oracle-EW: {summary['mean_gap_oracle_minus_ew']:+.4f}  "
          f"(oracle beats EW in {summary['n_seeds_oracle_beats_ew']}/{len(valid)} seeds, "
          f"Wilcoxon p={w_ew_p})")
    print(f"Mean gap oracle-CW: {summary['mean_gap_oracle_minus_cw']:+.4f}  "
          f"(oracle beats CW in {summary['n_seeds_oracle_beats_cw']}/{len(valid)} seeds, "
          f"Wilcoxon p={w_cw_p})")
    return summary


def main():
    results = {}
    results["domain_one_maintenance"] = run_domain("domain_one_maintenance", ROOT / "data")
    results["domain_two_it_incidents"] = run_domain("domain_two_it_incidents", ROOT / "data_it_incidents")

    out_path = ROOT / "experiments" / "oracle_ceiling_check_synthetic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
