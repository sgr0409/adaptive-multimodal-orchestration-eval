"""Downloads and extracts MM-IMDb (Arevalo, Solorio, Montes-y-Gomez,
Gonzalez, "Gated Multimodal Units for Information Fusion," ICLR 2017
Workshop Track): real movie posters (image) paired with real IMDb plot
summaries (text) and genuine, human-assigned genre labels. Used as this
paper's fourth validation domain and second real (non-procedurally-
generated) domain, added specifically to test whether the synthetic-to-
real transfer-failure finding (Section~\\ref{sec:results-public},
CrisisMMD) replicates on an independent real dataset from a completely
different task and content domain (movie genre, not disaster response),
rather than being a property of one dataset.

The original academic host (lisi1.unal.edu.co) is no longer reachable;
this downloads the same raw release (mmimdb.tar.gz, posters + per-movie
JSON metadata, one file pair per movie) from its Internet Archive mirror
at https://archive.org/details/mmimdb instead. Only "title", "plot", and
"genres" are extracted from each JSON file; the dataset also carries
extensive IMDb crew/cast metadata (director, cast, production companies,
etc.) not used by this paper and discarded during extraction.

MM-IMDb's genre labels are natively multi-label (a movie can be both
Action and Comedy). This paper's entire fusion/evaluation methodology is
built around single-label classification (argmax fusion, McNemar's test
on binary correctness), so this script restricts to movies with EXACTLY
ONE genre and reports the resulting per-genre counts; a specific 3-4-genre
subset mirroring CrisisMMD's 3-class structure is then selected in
generate_dataset.py, after these counts are known.
"""
import json
import tarfile
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_PATH = RAW_DIR / "mmimdb.tar.gz"
EXTRACT_DIR = RAW_DIR / "mmimdb"
ARCHIVE_URL = "https://archive.org/download/mmimdb/mmimdb.tar.gz"


def extract_archive():
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.glob("dataset/*.jpeg")):
        print(f"Already extracted at {EXTRACT_DIR}, skipping.", flush=True)
        return
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"{ARCHIVE_PATH} not found. Download it first, e.g.:\n"
            f"  curl -L -o {ARCHIVE_PATH} {ARCHIVE_URL}"
        )
    print(f"Extracting {ARCHIVE_PATH} (this is a ~8.7GB archive, takes a "
          f"while)...", flush=True)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tf:
        tf.extractall(RAW_DIR)
    print("Extraction complete.", flush=True)


def main():
    extract_archive()
    dataset_dir = EXTRACT_DIR / "dataset"
    json_files = sorted(dataset_dir.glob("*.json"))
    print(f"Found {len(json_files)} movies with metadata.", flush=True)

    records = []
    skipped_no_image, skipped_multi_genre, skipped_no_plot = 0, 0, 0
    for jf in json_files:
        movie_id = jf.stem
        img_path = dataset_dir / f"{movie_id}.jpeg"
        if not img_path.exists():
            skipped_no_image += 1
            continue
        meta = json.loads(jf.read_text())
        genres = meta.get("genres", [])
        if len(genres) != 1:
            skipped_multi_genre += 1
            continue
        plot = meta.get("plot", [])
        if not plot or not plot[0].strip():
            skipped_no_plot += 1
            continue
        records.append({
            "id": movie_id,
            "title": meta.get("title", ""),
            "year": meta.get("year"),
            "text": plot[0].strip(),
            "image_path": f"raw/mmimdb/dataset/{movie_id}.jpeg",
            "genre": genres[0],
        })

    out_path = DATA_DIR / "scenarios_raw.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    genre_counts = Counter(r["genre"] for r in records)
    print(f"\nSkipped: {skipped_no_image} no image, "
          f"{skipped_multi_genre} multi-genre (or zero-genre), "
          f"{skipped_no_plot} no plot text.", flush=True)
    print(f"Wrote {len(records)} single-genre usable movies to {out_path}", flush=True)
    print("\nSingle-genre movie counts, most common first "
          "(pick the genre subset for generate_dataset.py from this):", flush=True)
    for genre, count in genre_counts.most_common(30):
        print(f"  {genre:20s} {count:5d}")


if __name__ == "__main__":
    main()
