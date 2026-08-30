"""Preregistered external validation of CAV-Portfolio on CMU-MOSEI.

The experiment uses the canonical video-level CMU-MOSEI folds.  Per-modality
models and all adaptive candidates are fitted only on the training fold.  The
validation fold is used once to certify a frozen portfolio path, and the test
fold is evaluated only after selection.  Inputs are the official high-level
computational sequences distributed by MultiBench: words, COVAREP audio, and
OpenFace 2 visual features.
"""

import argparse
import hashlib
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.cav_portfolio_guard import CAVPortfolioGuardFusion


REFERENCE = "equal_weight"
CANDIDATES = ("confidence_weighted", "stacking", "learned_gating")
MODALITIES = ("text", "audio", "vision")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_id(segment_id):
    return segment_id.rsplit("[", 1)[0]


def valid_rows(array):
    array = np.asarray(array, dtype=np.float64)
    finite = np.all(np.isfinite(array), axis=1)
    nonzero = np.any(array != 0.0, axis=1)
    return array[finite & nonzero]


def summarize_sequence(array):
    rows = valid_rows(array)
    if len(rows) == 0:
        width = np.asarray(array).shape[1]
        return np.zeros(width * 2, dtype=np.float64)
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)])


def decode_words(array):
    return " ".join(
        item.decode("utf-8", errors="replace")
        for item in np.asarray(array).reshape(-1)
        if item and item != b"sp"
    )


def load_dataset(hdf5_path, folds_path):
    folds = runpy.run_path(str(folds_path))
    fold_by_video = {}
    for split, variable in (
        ("train", "standard_train_fold"),
        ("calibration", "standard_valid_fold"),
        ("test", "standard_test_fold"),
    ):
        for identifier in folds[variable]:
            if identifier in fold_by_video:
                raise ValueError(f"video appears in multiple folds: {identifier}")
            fold_by_video[identifier] = split

    records = {split: [] for split in ("train", "calibration", "test")}
    with h5py.File(hdf5_path, "r") as dataset:
        shared = set(dataset["All Labels"])
        for sequence in ("words", "COVAREP", "OpenFace_2"):
            shared &= set(dataset[sequence])
        for segment in sorted(shared):
            split = fold_by_video.get(video_id(segment))
            if split is None:
                continue
            sentiment = float(dataset["All Labels"][segment]["features"][0, 0])
            if sentiment == 0.0:
                continue
            records[split].append({
                "id": segment,
                "label": int(sentiment > 0.0),
                "text": decode_words(dataset["words"][segment]["features"]),
                "audio": summarize_sequence(
                    dataset["COVAREP"][segment]["features"]
                ),
                "vision": summarize_sequence(
                    dataset["OpenFace_2"][segment]["features"]
                ),
            })
    if any(not rows for rows in records.values()):
        raise ValueError("every canonical split must contain examples")
    return records


def arrays(records, split):
    rows = records[split]
    return {
        "ids": [row["id"] for row in rows],
        "labels": np.asarray([row["label"] for row in rows], dtype=int),
        "text": [row["text"] for row in rows],
        "audio": np.stack([row["audio"] for row in rows]),
        "vision": np.stack([row["vision"] for row in rows]),
    }


def modality_models(seed):
    return {
        "text": make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=30000,
                sublinear_tf=True,
            ),
            LogisticRegression(
                C=2.0, max_iter=1000, class_weight="balanced",
                random_state=seed,
            ),
        ),
        "audio": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                random_state=seed,
            ),
        ),
        "vision": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                random_state=seed,
            ),
        ),
    }


def modal_probabilities(models, data):
    return np.stack([
        models[name].predict_proba(data[name]) for name in MODALITIES
    ], axis=1)


def confidence_weighted(modal_probs, power=3.0):
    confidence = modal_probs.max(axis=2) ** power
    weights = confidence / confidence.sum(axis=1, keepdims=True)
    return np.sum(modal_probs * weights[:, :, None], axis=1)


def meta_features(modal_probs):
    confidence = modal_probs.max(axis=2)
    entropy = -np.sum(
        modal_probs * np.log(np.clip(modal_probs, 1e-12, 1.0)), axis=2
    )
    return np.concatenate([
        modal_probs.reshape(len(modal_probs), -1), confidence, entropy
    ], axis=1)


def records_from_probs(probabilities):
    return [
        {"label": int(np.argmax(row)), "probs": row.tolist()}
        for row in probabilities
    ]


def path_probabilities(modal_probs, stacker, gate):
    equal = modal_probs.mean(axis=1)
    confidence = confidence_weighted(modal_probs)
    stacking = stacker.predict_proba(meta_features(modal_probs))
    selected_modality = gate.predict(meta_features(modal_probs))
    gating = modal_probs[np.arange(len(modal_probs)), selected_modality]
    return {
        REFERENCE: equal,
        "confidence_weighted": confidence,
        "stacking": stacking,
        "learned_gating": gating,
    }


def metrics(probabilities, labels):
    predictions = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "n": int(len(labels)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_dataset(args.data, args.folds)
    train = arrays(records, "train")
    calibration = arrays(records, "calibration")
    test = arrays(records, "test")
    assert not (set(train["ids"]) & set(calibration["ids"]))
    assert not (set(train["ids"]) & set(test["ids"]))
    assert not (set(calibration["ids"]) & set(test["ids"]))

    models = modality_models(args.seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = []
    for name in MODALITIES:
        oof.append(cross_val_predict(
            models[name], train[name], train["labels"], cv=cv,
            method="predict_proba", n_jobs=1,
        ))
        models[name].fit(train[name], train["labels"])
    oof = np.stack(oof, axis=1)

    stacker = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=args.seed,
    ).fit(meta_features(oof), train["labels"])
    correct = (
        np.argmax(oof, axis=2) == train["labels"][:, None]
    )
    gate_targets = np.argmax(
        correct.astype(float) + 1e-3 * oof.max(axis=2), axis=1
    )
    gate = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=args.seed,
    ).fit(meta_features(oof), gate_targets)

    calibration_modal = modal_probabilities(models, calibration)
    test_modal = modal_probabilities(models, test)
    calibration_paths = path_probabilities(calibration_modal, stacker, gate)
    test_paths = path_probabilities(test_modal, stacker, gate)

    selector = CAVPortfolioGuardFusion(
        reference_name=REFERENCE,
        familywise_error_rate=0.05,
        minimum_decisive_switches=5,
    ).fit(
        records_from_probs(calibration_paths[REFERENCE]),
        {
            name: records_from_probs(calibration_paths[name])
            for name in CANDIDATES
        },
        calibration["labels"],
        calibration_mode="canonical_validation_independent_exact",
    )
    selected = selector.stats_.selected_path
    all_metrics = {
        name: metrics(probabilities, test["labels"])
        for name, probabilities in test_paths.items()
    }
    all_metrics["cav_portfolio"] = metrics(
        test_paths[selected], test["labels"]
    )
    naive = max(
        calibration_paths,
        key=lambda name: accuracy_score(
            calibration["labels"], np.argmax(calibration_paths[name], axis=1)
        ),
    )
    all_metrics["naive_validation_selector"] = metrics(
        test_paths[naive], test["labels"]
    )

    artifact = {
        "dataset": "CMU-MOSEI",
        "task": "binary positive-vs-negative sentiment; zero labels excluded",
        "source_features": ["words", "COVAREP", "OpenFace_2"],
        "protocol": (
            "Canonical video-disjoint train/validation/test folds; candidates "
            "trained on train, certified on validation, evaluated once on test."
        ),
        "seed": args.seed,
        "data_sha256": sha256(args.data),
        "split_counts": {
            split: len(records[split]) for split in records
        },
        "selector_stats": asdict(selector.stats_),
        "naive_validation_selected_path": naive,
        "test_metrics": all_metrics,
        "test_accuracy_gaps_vs_equal_weight": {
            name: value["accuracy"] - all_metrics[REFERENCE]["accuracy"]
            for name, value in all_metrics.items()
        },
        "test_ids": test["ids"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "split_counts": artifact["split_counts"],
        "selected": selected,
        "certificate": selector.stats_.selected_lower_accuracy_gain,
        "test_metrics": all_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
