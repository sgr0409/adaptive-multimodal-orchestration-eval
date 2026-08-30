"""Locked natural three-modality validation on official MIntRec2.0 splits."""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from framework.cav_portfolio_guard import CAVPortfolioGuardFusion

REFERENCE = "equal_weight"
MODALITIES = ("text", "audio", "vision")
CANDIDATES = ("confidence_weighted", "stacking", "learned_gating")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path):
    data = np.load(path)
    return {key: row for key, row in zip(data["keys"], data["features"])}


def read_split(path, audio, vision):
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            key = f"dia{row['Dialogue_id']}_utt{row['Utterance_id']}"
            rows.append({
                "id": key,
                "group": row["Dialogue_id"],
                "text": row["Text"],
                "label": row["Label"],
                "audio": audio[key],
                "vision": vision[key],
            })
    return rows


def arrays(rows, label_map):
    return {
        "ids": [r["id"] for r in rows],
        "groups": np.asarray([r["group"] for r in rows]),
        "text": [r["text"] for r in rows],
        "audio": np.stack([r["audio"] for r in rows]),
        "vision": np.stack([r["vision"] for r in rows]),
        "labels": np.asarray([label_map[r["label"]] for r in rows]),
    }


def subset(data, indexes):
    return {
        key: ([value[i] for i in indexes] if isinstance(value, list)
              else value[indexes])
        for key, value in data.items()
    }


def models(seed):
    common = dict(C=1.0, max_iter=1500, class_weight="balanced",
                  random_state=seed)
    return {
        "text": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                            max_features=30000, sublinear_tf=True),
            LogisticRegression(**common),
        ),
        "audio": make_pipeline(StandardScaler(), LogisticRegression(**common)),
        "vision": make_pipeline(StandardScaler(), LogisticRegression(**common)),
    }


def modal_probs(fitted, data):
    return np.stack([fitted[m].predict_proba(data[m]) for m in MODALITIES], axis=1)


def meta_features(probs):
    confidence = probs.max(axis=2)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=2)
    return np.concatenate([probs.reshape(len(probs), -1), confidence, entropy], axis=1)


def paths(probs, stacker, gate):
    confidence = probs.max(axis=2) ** 3
    weights = confidence / confidence.sum(axis=1, keepdims=True)
    selected = gate.predict(meta_features(probs))
    return {
        REFERENCE: probs.mean(axis=1),
        "confidence_weighted": np.sum(probs * weights[:, :, None], axis=1),
        "stacking": stacker.predict_proba(meta_features(probs)),
        "learned_gating": probs[np.arange(len(probs)), selected],
    }


def records(probs):
    return [{"label": int(row.argmax()), "probs": row.tolist()} for row in probs]


def metrics(probs, labels):
    pred = probs.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
            "n": int(len(labels))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio, vision = load_summary(args.audio), load_summary(args.vision)
    # Test annotations are intentionally not loaded until every candidate and
    # both dev-set selectors have been frozen below.
    train_rows = read_split(args.data_dir / "train.tsv", audio, vision)
    dev_rows = read_split(args.data_dir / "dev.tsv", audio, vision)
    labels = sorted({row["label"] for row in train_rows})
    label_map = {label: i for i, label in enumerate(labels)}
    train, dev = arrays(train_rows, label_map), arrays(dev_rows, label_map)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    fit_idx, meta_idx = next(splitter.split(train["labels"], groups=train["groups"]))
    fit, meta = subset(train, fit_idx), subset(train, meta_idx)
    base = models(args.seed)
    for name in MODALITIES:
        base[name].fit(fit[name], fit["labels"])
    meta_probs = modal_probs(base, meta)
    stacker = LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced",
                                 random_state=args.seed).fit(
        meta_features(meta_probs), meta["labels"])
    correct = meta_probs.argmax(axis=2) == meta["labels"][:, None]
    gate_target = np.argmax(correct.astype(float) + 1e-3 * meta_probs.max(axis=2), axis=1)
    gate = LogisticRegression(C=1.0, max_iter=1500, class_weight="balanced",
                              random_state=args.seed).fit(
        meta_features(meta_probs), gate_target)
    for name in MODALITIES:
        base[name].fit(train[name], train["labels"])

    dev_paths = paths(modal_probs(base, dev), stacker, gate)
    primary = CAVPortfolioGuardFusion(REFERENCE, 0.05, 5).fit(
        records(dev_paths[REFERENCE]),
        {name: records(dev_paths[name]) for name in CANDIDATES},
        dev["labels"], "official_dev_independent_exact")
    transfer = CAVPortfolioGuardFusion(REFERENCE, 0.05, 5).fit(
        records(dev_paths[REFERENCE]), {"stacking": records(dev_paths["stacking"])},
        dev["labels"], "official_dev_transfer_locked_single_candidate_exact")

    # First access to the official test labels occurs after selector fitting.
    test_rows = read_split(args.data_dir / "test.tsv", audio, vision)
    test = arrays(test_rows, label_map)
    test_paths = paths(modal_probs(base, test), stacker, gate)
    result_metrics = {name: metrics(value, test["labels"])
                      for name, value in test_paths.items()}
    for name, selector in (("cav_portfolio", primary),
                           ("cav_transfer_stacking", transfer)):
        result_metrics[name] = metrics(
            test_paths[selector.stats_.selected_path], test["labels"])

    artifact = {
        "dataset": "MIntRec2.0",
        "protocol_sha256": sha256(args.protocol),
        "feature_sha256": {"audio": sha256(args.audio), "vision": sha256(args.vision)},
        "split_counts": {"train": len(train_rows), "dev": len(dev_rows),
                         "test": len(test_rows), "meta_train": len(meta_idx)},
        "primary_selector": asdict(primary.stats_),
        "transfer_selector": asdict(transfer.stats_),
        "test_metrics": result_metrics,
        "test_accuracy_gaps_vs_equal_weight": {
            name: value["accuracy"] - result_metrics[REFERENCE]["accuracy"]
            for name, value in result_metrics.items()
        },
        "test_ids": test["ids"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"split_counts": artifact["split_counts"],
                      "primary": asdict(primary.stats_),
                      "transfer": asdict(transfer.stats_),
                      "test_metrics": result_metrics}, indent=2))


if __name__ == "__main__":
    main()
