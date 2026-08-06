"""Downloads the "damage" config of QCRI/CrisisMMD (Alam et al., CrisisMMD:
Multimodal Twitter Datasets from Natural Disasters, 2018; CC BY-NC-SA 4.0)
from the Hugging Face Hub: real disaster-response tweets (text) paired with
real photos (image) and a genuine 3-class damage-severity label
{little_or_no_damage, mild_damage, severe_damage}. Used as a public,
non-procedurally-generated validation domain (see
experiments/public_crisismmd/), added specifically to test whether the
confidence-weighted-fusion result generalizes beyond this paper's own
synthetic data construction, per external reviewer feedback that all prior
evidence came from procedurally generated benchmarks.

Only text and image are real modalities here (no third, telemetry-like
channel exists in the public dataset); degradation is applied afterward by
data_public_crisismmd/generate_dataset.py using the same disclosed
corruption mechanisms as the synthetic domains (blur/noise for image,
word-shuffle/dropout for text), following the standard practice in
robustness-evaluation literature (e.g. ImageNet-C) of injecting controlled,
disclosed corruption on top of real data -- the base content is real, only
the degradation mechanism is synthetic, same as it must be for any
controlled ablation regardless of data source.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset

DATA_DIR = Path(__file__).parent
IMG_DIR = DATA_DIR / "images"
BASE_URL = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main/"
LABEL_NAMES = ["little_or_no_damage", "mild_damage", "severe_damage"]
N_WORKERS = 24


def download_one(image_path):
    dest = IMG_DIR / image_path
    if dest.exists() and dest.stat().st_size > 0:
        return image_path, True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(BASE_URL + image_path, timeout=30)
        if r.status_code == 200 and len(r.content) > 0:
            dest.write_bytes(r.content)
            return image_path, True
    except Exception:
        pass
    return image_path, False


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for split in ["train", "dev", "test"]:
        print(f"Loading {split} split metadata...", flush=True)
        ds = load_dataset("QCRI/CrisisMMD", "damage", split=split)
        for row in ds:
            all_rows.append({
                "split": split,
                "event_name": row["event_name"],
                "tweet_id": row["tweet_id"],
                "tweet_text": row["tweet_text"],
                "image_path": row["image_path"],
                "label": LABEL_NAMES[row["label"]],
            })
    print(f"Total rows: {len(all_rows)}. Downloading images with {N_WORKERS} workers...", flush=True)

    ok_count, fail_count = 0, 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(download_one, r["image_path"]): r for r in all_rows}
        for i, fut in enumerate(as_completed(futures)):
            _, ok = fut.result()
            ok_count += int(ok)
            fail_count += int(not ok)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(all_rows)} processed ({ok_count} ok, {fail_count} failed)", flush=True)

    print(f"Done: {ok_count} ok, {fail_count} failed.", flush=True)

    records = []
    for r in all_rows:
        img_file = IMG_DIR / r["image_path"]
        if img_file.exists() and img_file.stat().st_size > 0:
            records.append(r)

    out_path = DATA_DIR / "scenarios_raw.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    label_dist = Counter(r["label"] for r in records)
    split_dist = Counter(r["split"] for r in records)
    print(f"Wrote {len(records)} usable scenarios to {out_path}")
    print(f"  label distribution: {dict(label_dist)}")
    print(f"  split distribution: {dict(split_dist)}")


if __name__ == "__main__":
    main()
