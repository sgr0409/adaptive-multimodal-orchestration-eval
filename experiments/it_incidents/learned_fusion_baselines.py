"""Domain-2 (IT incidents) copy of experiments/learned_fusion_baselines.py.
Same methodology; see that file's docstring and framework/learned_fusion.py's
docstring for rationale."""
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
from framework.telemetry_scorer import extract_features as telemetry_features
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion
from framework.learned_fusion import StackingFusion, LearnedGatingFusion

SEED = 42
DATA_DIR = ROOT / "data_it_incidents"
RESULTS_DIR = ROOT / "experiments" / "it_incidents" / "results"
N_FOLDS = 5


def _fit_predict_probs(X_train, y_train, X_eval, modality_name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
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


def main():
    print("Loading and splitting scenarios...", flush=True)
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    rng = random.Random(SEED)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    n_train = len(shuffled) // 2
    train, test = shuffled[:n_train], shuffled[n_train:]
    y_train = [r["label"] for r in train]

    print("Embedding train and test splits once...", flush=True)
    text_encoder = TextScorer().encoder
    text_train_emb = text_encoder.encode([r["text"] for r in train], show_progress_bar=False)
    text_test_emb = text_encoder.encode([r["text"] for r in test], show_progress_bar=False)

    image_embedder = ImageScorer()
    image_train_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in train])
    image_test_emb = image_embedder.embed([str(DATA_DIR / r["image_path"]) for r in test])

    telemetry_train_feat = np.array([telemetry_features(r["telemetry"]) for r in train])
    telemetry_test_feat = np.array([telemetry_features(r["telemetry"]) for r in test])

    print(f"Generating {N_FOLDS}-fold out-of-fold meta-features on the train split...", flush=True)
    n = len(train)
    oof_text, oof_image, oof_telemetry = [None] * n, [None] * n, [None] * n
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (fit_idx, hold_idx) in enumerate(kf.split(np.arange(n))):
        y_fit = [y_train[i] for i in fit_idx]
        text_preds = _fit_predict_probs(text_train_emb[fit_idx], y_fit, text_train_emb[hold_idx], "text")
        image_preds = _fit_predict_probs(image_train_emb[fit_idx], y_fit, image_train_emb[hold_idx], "image")
        telemetry_preds = _fit_predict_probs(telemetry_train_feat[fit_idx], y_fit, telemetry_train_feat[hold_idx], "telemetry")
        for local_i, global_i in enumerate(hold_idx):
            oof_text[global_i] = text_preds[local_i]
            oof_image[global_i] = image_preds[local_i]
            oof_telemetry[global_i] = telemetry_preds[local_i]
        print(f"  fold {fold_idx + 1}/{N_FOLDS} done", flush=True)

    oof_modality_results = [[oof_text[i], oof_image[i], oof_telemetry[i]] for i in range(n)]

    print("Fitting meta-learners on out-of-fold features...", flush=True)
    stacking = StackingFusion()
    stacking.fit(oof_modality_results, y_train)
    gating = LearnedGatingFusion()
    gating.fit(oof_modality_results, y_train)

    print("Scoring test split with base scorers fit on the full train split...", flush=True)
    text_test_preds = _fit_predict_probs(text_train_emb, y_train, text_test_emb, "text")
    image_test_preds = _fit_predict_probs(image_train_emb, y_train, image_test_emb, "image")
    telemetry_test_preds = _fit_predict_probs(telemetry_train_feat, y_train, telemetry_test_feat, "telemetry")

    per_example = []
    for i, r in enumerate(test):
        mods = [text_test_preds[i], image_test_preds[i], telemetry_test_preds[i]]
        per_example.append({
            "id": r["id"], "true_label": r["label"], "degraded_modalities": r["degraded_modalities"],
            "confidence_weighted": confidence_weighted_fusion(mods)["label"],
            "equal_weight": equal_weight_fusion(mods)["label"],
            "stacking": stacking.predict(mods)["label"],
            "gating": gating.predict(mods)["label"],
        })

    def acc_and_correct(key, idxs=None):
        idxs = idxs if idxs is not None else range(len(per_example))
        correct = [per_example[i]["true_label"] == per_example[i][key] for i in idxs]
        return round(sum(correct) / len(correct), 4), correct

    hard = [i for i, e in enumerate(per_example) if len(e["degraded_modalities"]) == 2]

    overall = {k: acc_and_correct(k) for k in ["equal_weight", "confidence_weighted", "stacking", "gating"]}
    hard_res = {k: acc_and_correct(k, hard) for k in ["equal_weight", "confidence_weighted", "stacking", "gating"]}

    results = {
        "n_test": len(test),
        "n_hard_subset": len(hard),
        "n_folds": N_FOLDS,
        "overall_accuracy": {k: v[0] for k, v in overall.items()},
        "hard_subset_accuracy": {k: v[0] for k, v in hard_res.items()},
        "hard_subset_mcnemar_vs_confidence_weighted": {
            "stacking": mcnemar(hard_res["stacking"][1], hard_res["confidence_weighted"][1]),
            "learned_gating": mcnemar(hard_res["gating"][1], hard_res["confidence_weighted"][1]),
        },
        "overall_mcnemar_vs_confidence_weighted": {
            "stacking": mcnemar(overall["stacking"][1], overall["confidence_weighted"][1]),
            "learned_gating": mcnemar(overall["gating"][1], overall["confidence_weighted"][1]),
        },
        "note": "McNemar b = new-method-right/CW-wrong, c = new-method-wrong/CW-right.",
    }

    with open(RESULTS_DIR / "learned_fusion_baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'learned_fusion_baselines.json'}")


if __name__ == "__main__":
    main()
