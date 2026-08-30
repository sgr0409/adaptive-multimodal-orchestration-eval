"""Replay strict CAV counts and simulate anytime false-admission control."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from framework.sequential_cav_admission import SequentialCAVAdmission


def outcome_matrix(evidence, n, rng):
    names = tuple(evidence)
    matrix = {name: np.concatenate([
        np.ones(evidence[name]["benefits"], dtype=np.int8),
        -np.ones(evidence[name]["harms"], dtype=np.int8),
        np.zeros(n - evidence[name]["benefits"] - evidence[name]["harms"], dtype=np.int8),
    ]) for name in names}
    for name in names:
        rng.shuffle(matrix[name])
    return matrix


def replay_partition(partition, repeats, seed):
    evidence = partition["portfolio_stats"]["candidate_evidence"]
    names = tuple(evidence)
    rng = np.random.default_rng(seed)
    admissions, times, paths = 0, [], {name: 0 for name in names}
    for _ in range(repeats):
        matrix = outcome_matrix(evidence, partition["n_calibration"], rng)
        controller = SequentialCAVAdmission(names)
        for t in range(partition["n_calibration"]):
            stats = controller.update({name: matrix[name][t] for name in names})
            if stats.selected_path != controller.reference_name:
                admissions += 1
                times.append(stats.admission_time)
                paths[stats.selected_path] += 1
                break
    return {
        "seed": partition["seed"],
        "replays": repeats,
        "admission_rate": admissions / repeats,
        "median_admission_example": float(np.median(times)) if times else None,
        "path_counts": paths,
    }


def null_simulation(n, candidates, simulations, seed):
    rng = np.random.default_rng(seed)
    false = 0
    grid = np.linspace(0.51, 0.99, 25)
    log_benefit = np.log(2 * grid)
    log_harm = np.log(2 * (1 - grid))
    log_threshold = np.log(candidates / 0.05)
    batch_size = 250
    for start in range(0, simulations, batch_size):
        size = min(batch_size, simulations - start)
        crossed = np.zeros(size, dtype=bool)
        for _ in range(candidates):
            benefits = rng.random((size, n)) < 0.5
            b = np.cumsum(benefits, axis=1)
            h = np.arange(1, n + 1)[None, :] - b
            logs = (b[:, :, None] * log_benefit[None, None, :]
                    + h[:, :, None] * log_harm[None, None, :])
            mixture = np.logaddexp.reduce(logs, axis=2) - np.log(len(grid))
            crossed |= np.any(mixture >= log_threshold, axis=1)
        false += int(crossed.sum())
    rate = false / simulations
    se = float(np.sqrt(rate * (1 - rate) / simulations))
    return {"horizon": n, "candidates": candidates, "simulations": simulations,
            "false_admissions": false, "empirical_fwer": rate,
            "monte_carlo_se": se}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--null-simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    source = json.loads(args.artifact.read_text())
    domains = {}
    for d, (name, value) in enumerate(source["domains"].items()):
        rows = [replay_partition(p, args.replays, args.seed + d * 1000 + i)
                for i, p in enumerate(value["sensitivity_partitions"])]
        domains[name] = {
            "partitions": rows,
            "mean_admission_rate": float(np.mean([r["admission_rate"] for r in rows])),
            "partitions_majority_admitted": sum(r["admission_rate"] > 0.5 for r in rows),
            "partitions_ever_admitted": sum(r["admission_rate"] > 0 for r in rows),
        }
    null = [null_simulation(n, 3, args.null_simulations, args.seed + n)
            for n in (450, 1106)]
    artifact = {
        "method": "Sequential CAV Admission finite-grid mixture e-process",
        "familywise_error_rate": 0.05,
        "alternative_grid": {"start": 0.51, "stop": 0.99, "count": 25},
        "replay_semantics": "independent random permutations of stored benefit, harm, and neutral counts",
        "domains": domains,
        "null_simulations": null,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"domains": {k: {x: v[x] for x in
          ("mean_admission_rate", "partitions_majority_admitted", "partitions_ever_admitted")}
          for k, v in domains.items()}, "null_simulations": null}, indent=2))


if __name__ == "__main__":
    main()
