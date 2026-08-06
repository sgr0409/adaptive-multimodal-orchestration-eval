"""Second task domain: IT/network incident triage. Same three-modality
schema as data/generate_dataset.py's equipment-maintenance-triage task
(text ticket, image, telemetry, severity label, disclosed degradation
mechanism), but a genuinely different task -- different labels, different
text content, a different visual representation (a horizontal system-health
status bar rather than a semicircular gauge), and different telemetry
channels (CPU utilization, request latency, error rate rather than
temperature/vibration/pressure) -- built to test whether the confidence-aware
fusion result generalizes beyond the first domain, not to relabel the same
data.

This generator incorporates two lessons learned from building the first
domain's generator rather than repeating them: the image degradation zones
are a single monotonic gradient (not a symmetric arc that would scramble
the angle-to-severity mapping), and degraded telemetry is wide-range
uniform noise independent of the true label (not centered on any one
class, which would make "degraded" mean "confidently wrong" rather than
"genuinely ambiguous").
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SEED = 42
DATA_DIR = Path(__file__).parent
IMG_DIR = DATA_DIR / "images"
N_PER_CLASS = 500
LABELS = ["minor", "degraded", "outage"]
ACTION_OF = {
    "minor": "monitor",
    "degraded": "investigate",
    "outage": "page_oncall",
}

TEXT_TEMPLATES = {
    "minor": [
        "One user reported a brief page load delay, resolved on retry.",
        "Routine health check passed, all services nominal.",
        "Minor latency blip observed, no user impact reported.",
        "Scheduled maintenance completed successfully, no issues.",
        "Support ticket closed, cosmetic UI glitch only.",
    ],
    "degraded": [
        "Multiple users reporting intermittent slow page loads over the last hour.",
        "Checkout service showing elevated error rates, some transactions failing.",
        "API response times degraded for a subset of endpoints, investigating.",
        "Search feature returning incomplete results for some queries.",
        "Background job queue backing up, processing delays growing.",
    ],
    "outage": [
        "Complete service outage, all users unable to log in, page immediately.",
        "Database cluster unreachable, every request failing, critical.",
        "Payment processing down site-wide, revenue-impacting, escalate now.",
        "Full regional outage confirmed, all customer traffic affected.",
        "Core API returning 500 for 100% of requests, sev-1 declared.",
    ],
    "degraded_text": [
        "Ticket opened for review per standard process.",
        "Please look into this when you have a chance.",
        "General status check requested by team.",
        "Support queue item logged, awaiting triage.",
        "Routine ticket filed for tracking purposes.",
    ],
}

# Health-bar fill fraction ranges (0=empty/worst, 1=full/best), matching a
# monotonic red[0,0.33)->yellow[0.33,0.66)->green[0.66,1.0] gradient drawn
# below, so severity increases as the health fraction decreases.
HEALTH_ZONES = {"outage": (0.02, 0.28), "degraded": (0.38, 0.62), "minor": (0.72, 0.98)}

TELEMETRY_PROFILES = {
    # (mean, std) per channel: cpu_pct, latency_ms, error_rate_pct
    "minor": {"cpu_pct": (35, 5), "latency_ms": (80, 10), "error_rate_pct": (0.2, 0.1)},
    "degraded": {"cpu_pct": (65, 8), "latency_ms": (350, 60), "error_rate_pct": (5.0, 1.5)},
    "outage": {"cpu_pct": (92, 6), "latency_ms": (2500, 400), "error_rate_pct": (45.0, 10.0)},
}
N_TIMESTEPS = 20
MODALITY_NAMES = ["text", "image", "telemetry"]


def draw_health_bar(rng, fraction, degraded):
    """Draws a 160x80 horizontal status bar: a fixed red/yellow/green
    background gradient with a fill overlay whose right edge sits at
    `fraction` of the bar's width (higher fraction = healthier). If
    degraded, the bar is heavily blurred/noised so the fill level is not
    legible."""
    w, h = 160, 80
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    bar_x0, bar_y0, bar_x1, bar_y1 = 10, 30, 150, 55

    seg_w = (bar_x1 - bar_x0) / 3
    draw.rectangle([bar_x0, bar_y0, bar_x0 + seg_w, bar_y1], fill=(200, 40, 40))
    draw.rectangle([bar_x0 + seg_w, bar_y0, bar_x0 + 2 * seg_w, bar_y1], fill=(220, 180, 20))
    draw.rectangle([bar_x0 + 2 * seg_w, bar_y0, bar_x1, bar_y1], fill=(40, 160, 40))

    fill_x = bar_x0 + fraction * (bar_x1 - bar_x0)
    draw.rectangle([fill_x, bar_y0 - 4, bar_x1, bar_y1 + 4], fill=(255, 255, 255))
    draw.rectangle([bar_x0 - 2, bar_y0 - 2, fill_x, bar_y1 + 2], outline=(20, 20, 20), width=2)
    draw.line([fill_x, bar_y0 - 6, fill_x, bar_y1 + 6], fill=(20, 20, 20), width=3)

    if degraded:
        img = img.filter(ImageFilter.GaussianBlur(radius=5))
        noise = Image.effect_noise((w, h), 60).convert("RGB")
        img = Image.blend(img, noise, alpha=0.4)
    return img


def gen_telemetry(rng, label, degraded):
    if degraded:
        series = {}
        for channel in TELEMETRY_PROFILES["minor"]:
            all_means = [TELEMETRY_PROFILES[l][channel][0] for l in LABELS]
            all_stds = [TELEMETRY_PROFILES[l][channel][1] for l in LABELS]
            lo = min(all_means) - 3 * all_stds[all_means.index(min(all_means))]
            hi = max(all_means) + 3 * all_stds[all_means.index(max(all_means))]
            lo = max(lo, 0.0)
            series[channel] = [round(rng.uniform(lo, hi), 3) for _ in range(N_TIMESTEPS)]
        return series
    series = {}
    for channel, (mean, std) in TELEMETRY_PROFILES[label].items():
        series[channel] = [round(max(0.0, rng.gauss(mean, std)), 3) for _ in range(N_TIMESTEPS)]
    return series


def main():
    rng = random.Random(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    idx = 0
    for label in LABELS:
        for _ in range(N_PER_CLASS):
            n_degraded = rng.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
            degraded_modalities = rng.sample(MODALITY_NAMES, n_degraded)

            text = rng.choice(TEXT_TEMPLATES["degraded_text" if "text" in degraded_modalities else label])

            if "image" in degraded_modalities:
                fraction = rng.uniform(0.0, 1.0)
            else:
                lo, hi = HEALTH_ZONES[label]
                fraction = rng.uniform(lo, hi)
            img = draw_health_bar(rng, fraction, degraded=("image" in degraded_modalities))
            img_path = IMG_DIR / f"scenario_{idx:04d}.png"
            img.save(img_path)

            telemetry = gen_telemetry(rng, label, degraded=("telemetry" in degraded_modalities))

            records.append({
                "id": idx,
                "text": text,
                "image_path": str(img_path.relative_to(DATA_DIR)),
                "telemetry": telemetry,
                "label": label,
                "action": ACTION_OF[label],
                "degraded_modalities": degraded_modalities,
            })
            idx += 1

    rng.shuffle(records)
    for i, r in enumerate(records):
        r["id"] = i

    out_path = DATA_DIR / "scenarios.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    count_dist = Counter(len(r["degraded_modalities"]) for r in records)
    print(f"Wrote {len(records)} scenarios to {out_path}")
    print(f"  degraded-modality-count distribution: "
          f"0={count_dist[0]}, 1={count_dist[1]}, 2={count_dist[2]}")
    for m in MODALITY_NAMES:
        n = sum(1 for r in records if m in r["degraded_modalities"])
        print(f"  {m} degraded in {n} scenarios")


if __name__ == "__main__":
    main()
