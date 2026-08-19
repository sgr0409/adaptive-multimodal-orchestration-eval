"""Selects a genre subset and applies controlled, disclosed degradation to
the real MM-IMDb data downloaded by download_dataset.py, producing this
paper's fourth validation domain and second real (non-procedurally-
generated) domain: real movie posters (image) and real IMDb plot summaries
(text), each carrying a genuine single genre label
(Arevalo et al.~\\cite{arevalo2017gated}).

Genre selection: TARGET_GENRES below picks a small number of genres,
restricted to movies download_dataset.py already filtered to exactly one
genre, mirroring the 3-class structure used throughout the rest of this
paper (domain one and two's normal/warning/critical and
minor/degraded/outage, CrisisMMD's little/mild/severe damage). The
specific genres are chosen from download_dataset.py's printed counts for
(a) enough examples per class for a 30-seed robustness check and (b)
genuine visual/textual distinguishability (a poster's dominant color
palette and a plot's vocabulary plausibly differ between, say, Horror and
Documentary, the same way they should for genuinely different genres --
not asserted, but a property we check post hoc via each modality's
standalone accuracy, same as every other domain in this paper).

Only two modalities exist here (no third, telemetry-like real-world
signal), identical to CrisisMMD -- at most one is degraded per scenario.
Degradation uses the exact same disclosed mechanism as CrisisMMD and both
synthetic domains: degraded images are heavily Gaussian-blurred and
noised; degraded text has its words shuffled with 70% dropped, keeping the
vocabulary genuinely the movie's own plot summary, just scrambled and
mostly missing.
"""
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter

SEED = 42
DATA_DIR = Path(__file__).parent
DEGRADED_IMG_DIR = DATA_DIR / "images_degraded"
DEGRADED_TEXT_DROP_FRACTION = 0.7
DEGRADED_MODALITY_WEIGHTS = {"none": 0.30, "text": 0.35, "image": 0.35}

# Finalized after inspecting download_dataset.py's printed single-genre
# counts (see that script's docstring); update here if the genre subset
# changes.
TARGET_GENRES = ["Documentary", "Horror", "Comedy"]
MAX_PER_GENRE = 1200  # caps class imbalance; downsample larger genres to this


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
    by_genre = {}
    for r in rows:
        if r["genre"] in TARGET_GENRES:
            by_genre.setdefault(r["genre"], []).append(r)

    print("Available per target genre (before capping):", flush=True)
    for g in TARGET_GENRES:
        print(f"  {g:15s} {len(by_genre.get(g, []))}", flush=True)

    selected = []
    for g in TARGET_GENRES:
        pool = by_genre.get(g, [])
        rng.shuffle(pool)
        selected.extend(pool[:MAX_PER_GENRE])
    rng.shuffle(selected)
    print(f"\nSelected {len(selected)} movies across {len(TARGET_GENRES)} genres "
          f"(capped at {MAX_PER_GENRE}/genre).", flush=True)

    DEGRADED_IMG_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, r in enumerate(selected):
        choice = rng.choices(list(DEGRADED_MODALITY_WEIGHTS.keys()),
                              weights=list(DEGRADED_MODALITY_WEIGHTS.values()))[0]
        degraded_modalities = [] if choice == "none" else [choice]

        text = degrade_text(rng, r["text"]) if choice == "text" else r["text"]

        src_path = DATA_DIR / r["image_path"]
        if choice == "image":
            img = Image.open(src_path).convert("RGB")
            img = degrade_image(rng, img)
            out_rel = f"images_degraded/{idx:05d}.jpg"
            img.save(DATA_DIR / out_rel, quality=85)
            image_path = out_rel
        else:
            image_path = r["image_path"]

        records.append({
            "id": idx,
            "title": r["title"],
            "text": text,
            "text_original": r["text"],
            "image_path": image_path,
            "label": r["genre"],
            "degraded_modalities": degraded_modalities,
        })

    out_path = DATA_DIR / "scenarios.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    label_dist = Counter(r["label"] for r in records)
    deg_dist = Counter(len(r["degraded_modalities"]) for r in records)
    print(f"\nWrote {len(records)} scenarios to {out_path}")
    print(f"  label distribution: {dict(label_dist)}")
    print(f"  degraded-modality-count distribution: {dict(deg_dist)}")


if __name__ == "__main__":
    main()
