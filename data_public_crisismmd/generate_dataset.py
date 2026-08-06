"""Applies controlled, disclosed degradation to the real CrisisMMD data
downloaded by download_dataset.py, producing a third, non-procedurally-
generated validation domain: real disaster-response tweets (text) and real
photos (image), each carrying a genuine 3-class damage-severity label
{little_or_no_damage, mild_damage, severe_damage}
(Alam et al.~\cite{alam2018crisismmd}).

Only two modalities exist here (no third, telemetry-like real-world signal
is available in the public dataset), unlike this paper's two synthetic
domains. Because there are only two modalities, at most one is degraded per
scenario -- there is no "two of three degraded, one clean" condition to
construct; the analogous hard case here is "one of two degraded," where
equal-weight fusion is forced into an even 50/50 average with an unreliable
signal while confidence-weighted fusion can down-weight it.

Degradation is injected the same way robustness-corruption benchmarks
(e.g. ImageNet-C) standardly do on top of real data: the base content
(text, photo) is genuinely real and unmodified when clean; when degraded,
it is corrupted by a disclosed, seeded mechanism, not replaced with
synthetic content. Degraded images are heavily Gaussian-blurred and noised
(same mechanism as the two synthetic domains). Degraded text is the same
real tweet with its words shuffled and 70% of them dropped, rather than
substituted with a generic template -- the vocabulary is still genuinely
from that tweet, just scrambled into an uninformative order with most of
it missing.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageFilter

SEED = 42
DATA_DIR = Path(__file__).parent
IMG_DIR = DATA_DIR / "images"
DEGRADED_IMG_DIR = DATA_DIR / "images_degraded"
DEGRADED_TEXT_DROP_FRACTION = 0.7
DEGRADED_MODALITY_WEIGHTS = {"none": 0.30, "text": 0.35, "image": 0.35}


def degrade_text(rng, text):
    words = text.split()
    rng.shuffle(words)
    keep_n = max(1, int(len(words) * (1 - DEGRADED_TEXT_DROP_FRACTION)))
    kept = words[:keep_n]
    rng.shuffle(kept)
    return " ".join(kept)


def degrade_image(rng, img):
    img = img.filter(ImageFilter.GaussianBlur(radius=12))
    noise = Image.effect_noise(img.size, 55).convert("RGB")
    return Image.blend(img, noise, alpha=0.5)


def main():
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(DATA_DIR / "scenarios_raw.jsonl")]
    rng.shuffle(rows)
    DEGRADED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, r in enumerate(rows):
        choice = rng.choices(list(DEGRADED_MODALITY_WEIGHTS.keys()),
                              weights=list(DEGRADED_MODALITY_WEIGHTS.values()))[0]
        degraded_modalities = [] if choice == "none" else [choice]

        if choice == "text":
            text = degrade_text(rng, r["tweet_text"])
        else:
            text = r["tweet_text"]

        src_path = IMG_DIR / r["image_path"]
        if choice == "image":
            img = Image.open(src_path).convert("RGB")
            img = degrade_image(rng, img)
            out_rel = f"images_degraded/{idx:05d}.jpg"
            img.save(DATA_DIR / out_rel, quality=85)
            image_path = out_rel
        else:
            image_path = f"images/{r['image_path']}"

        records.append({
            "id": idx,
            "text": text,
            "text_original": r["tweet_text"],
            "image_path": image_path,
            "label": r["label"],
            "event_name": r["event_name"],
            "degraded_modalities": degraded_modalities,
        })

    out_path = DATA_DIR / "scenarios.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    label_dist = Counter(r["label"] for r in records)
    deg_dist = Counter(len(r["degraded_modalities"]) for r in records)
    print(f"Wrote {len(records)} scenarios to {out_path}")
    print(f"  label distribution: {dict(label_dist)}")
    print(f"  degraded-modality-count distribution: {dict(deg_dist)}")


if __name__ == "__main__":
    main()
