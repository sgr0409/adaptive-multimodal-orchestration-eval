"""Fourth validation domain: real movie posters + real IMDb plot summaries
(MM-IMDb, data_public_mmimdb/), added to test whether the synthetic-to-real
transfer-failure finding (experiments/public_crisismmd/) replicates on a
second, independent real dataset from a completely different task and
content domain (movie genre, not disaster response), per the same external
reviewer concern that motivated CrisisMMD: one real dataset's finding could
be a property of that dataset, not a general phenomenon.

Only two real modalities exist here (no third, telemetry-like signal),
identical to CrisisMMD; see data_public_mmimdb/generate_dataset.py for the
degradation mechanism (byte-identical to CrisisMMD's) and genre selection.

Reports equal-weight fusion, confidence-weighted fusion (power=3), and the
two learned fusion baselines (framework/learned_fusion.py) side by side, on
the same held-out test split -- the identical methodology used for
CrisisMMD, so the two real domains are directly comparable.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.image_scorer import ImageScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion
from framework.learned_fusion import StackingFusion, LearnedGatingFusion

SEED = 42
DATA_DIR = ROOT / "data_public_mmimdb"
RESULTS_DIR = ROOT / "experiments" / "public_mmimdb" / "results"
N_BOOTSTRAP = 2000
N_FOLDS = 5
LABELS = ["Comedy", "Documentary", "Horror"]


def _fit_predict_probs(X_train, y_train, X_eval, modality_name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_eval)
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
    print("Loading and splitting scenarios...", flush=True)
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    rng = random.Random(SEED)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    n_train = len(shuffled) // 2
    train, test = shuffled[:n_train], shuffled[n_train:]
    y_train = [r["label"] for r in train]
    y_test = [r["label"] for r in test]

    print(f"Embedding {len(train)} train + {len(test)} test examples...", flush=True)
    text_encoder = TextScorer().encoder
    text_train_emb = text_encoder.encode([r["text"] for r in train], show_progress_bar=False)
    text_test_emb = text_encoder.encode([r["text"] for r in test], show_progress_bar=False)

    image_embedder = ImageScorer()
    image_train_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in train])
    image_test_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in test])

    print("Training per-modality scorers on the full train split...", flush=True)
    text_test_preds = _fit_predict_probs(text_train_emb, y_train, text_test_emb, "text")
    image_test_preds = _fit_predict_probs(image_train_emb, y_train, image_test_emb, "image")

    print(f"Generating {N_FOLDS}-fold out-of-fold meta-features for learned fusion baselines...", flush=True)
    n = len(train)
    oof_text, oof_image = [None] * n, [None] * n
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fit_idx, hold_idx in kf.split(np.arange(n)):
        y_fit = [y_train[i] for i in fit_idx]
        t_preds = _fit_predict_probs(text_train_emb[fit_idx], y_fit, text_train_emb[hold_idx], "text")
        i_preds = _fit_predict_probs(image_train_emb[fit_idx], y_fit, image_train_emb[hold_idx], "image")
        for local_i, global_i in enumerate(hold_idx):
            oof_text[global_i] = t_preds[local_i]
            oof_image[global_i] = i_preds[local_i]
    oof_modality_results = [[oof_text[i], oof_image[i]] for i in range(n)]

    stacking = StackingFusion()
    stacking.fit(oof_modality_results, y_train)
    gating = LearnedGatingFusion()
    gating.fit(oof_modality_results, y_train)

    print("Scoring test split with all fusion methods...", flush=True)
    per_example = []
    for i, r in enumerate(test):
        mods = [text_test_preds[i], image_test_preds[i]]
        per_example.append({
            "id": r["id"], "true_label": r["label"], "degraded_modalities": r["degraded_modalities"],
            "text": text_test_preds[i], "image": image_test_preds[i],
            "equal_weight": equal_weight_fusion(mods)["label"],
            "confidence_weighted": confidence_weighted_fusion(mods)["label"],
            "stacking": stacking.predict(mods)["label"],
            "gating": gating.predict(mods)["label"],
        })

    def eval_method(key, idxs=None):
        idxs = idxs if idxs is not None else range(len(per_example))
        yt = [per_example[i]["true_label"] for i in idxs]
        yp = [per_example[i][key] for i in idxs]
        correct = [t == p for t, p in zip(yt, yp)]
        precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, labels=LABELS, average="macro", zero_division=0)
        return {
            "accuracy": round(sum(correct) / len(correct), 4),
            "macro_f1": round(float(f1), 4),
            "bootstrap_accuracy_ci": bootstrap_ci(correct),
        }, correct

    hard = [i for i, e in enumerate(per_example) if len(e["degraded_modalities"]) == 1]

    results = {"meta": {"seed": SEED, "n_folds": N_FOLDS, "n_bootstrap": N_BOOTSTRAP,
                         "domain": "public_mmimdb", "dataset_size": len(scenarios)},
               "dataset": {"n_total": len(scenarios), "n_train": len(train), "n_test": len(test),
                           "n_hard_subset": len(hard)}}

    correct_by_method = {}
    for key in ["equal_weight", "confidence_weighted", "stacking", "gating"]:
        overall_metrics, overall_correct = eval_method(key)
        hard_metrics, hard_correct = eval_method(key, hard)
        results[key] = {"overall": overall_metrics, "hard_subset": hard_metrics}
        correct_by_method[key] = {"overall": overall_correct, "hard": hard_correct}

    results["significance_vs_equal_weight"] = {
        m: {"overall": mcnemar(correct_by_method[m]["overall"], correct_by_method["equal_weight"]["overall"]),
            "hard_subset": mcnemar(correct_by_method[m]["hard"], correct_by_method["equal_weight"]["hard"])}
        for m in ["confidence_weighted", "stacking", "gating"]
    }
    results["significance_vs_confidence_weighted"] = {
        m: {"overall": mcnemar(correct_by_method[m]["overall"], correct_by_method["confidence_weighted"]["overall"]),
            "hard_subset": mcnemar(correct_by_method[m]["hard"], correct_by_method["confidence_weighted"]["hard"])}
        for m in ["stacking", "gating"]
    }

    conf_check = {}
    for mod in ["text", "image"]:
        deg = [e[mod]["confidence"] for e in per_example if mod in e["degraded_modalities"]]
        clean = [e[mod]["confidence"] for e in per_example if mod not in e["degraded_modalities"]]
        conf_check[mod] = {
            "mean_confidence_when_degraded": round(float(np.mean(deg)), 4) if deg else None,
            "mean_confidence_when_not_degraded": round(float(np.mean(clean)), 4) if clean else None,
            "n_degraded": len(deg),
        }
    results["modality_confidence_reliability_check"] = conf_check

    cw_pred = [e["confidence_weighted"] for e in per_example]
    results["confidence_weighted_confusion_matrix"] = confusion_matrix(y_test, cw_pred, labels=LABELS).tolist()
    results["confusion_matrix_labels"] = LABELS

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "per_example.json", "w") as f:
        json.dump(per_example, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k not in
                       ["confidence_weighted_confusion_matrix"]}, indent=2))
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
