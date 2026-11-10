"""Analyze matched replay frames with a whole-round bootstrap."""

import argparse
import bisect
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROBS = (0.25, 0.50, 0.75, 0.95, 0.99)
NAMES = ("q1_ms", "median_ms", "q3_ms", "p95_ms", "p99_ms")
SLOW_LIMIT = 33.3
TELEMETRY = ("cpu_percent", "process_cpu_percent", "sm_mhz", "mem_mhz", "temperature_c",
             "power_w", "gpu_percent", "gpu_memory_used_bytes", "pstate")
TRAFFIC = ("upload_bytes", "warm_total_h2d_bytes", "prefetch_bytes", "prefetch_wasted_bytes",
           "total_evicted_bytes", "prefetch_evicted_bytes", "prefetch_d2d_logical_bytes")
DIAGNOSTIC_PREDECESSOR = {"pinned": "touch", "reuse": "pinned"}


def distribution(values, resamples):
    estimates = np.quantile(values, PROBS)
    boot = np.concatenate([
        np.quantile(values[part].reshape(len(part), -1), PROBS, axis=1).T
        for part in np.array_split(resamples, max(1, len(resamples) // 32))
    ])
    report = {key: {"estimate": float(estimates[i]),
                    "ci95": np.percentile(boot[:, i], [2.5, 97.5]).tolist()}
              for i, key in enumerate(NAMES)}
    report["iqr_ms"] = {"estimate": float(estimates[2] - estimates[0]),
                        "ci95": np.percentile(boot[:, 2] - boot[:, 0], [2.5, 97.5]).tolist()}
    return report, boot


def contrast(a, b, boot_a, boot_b, resamples):
    delta = a - b
    report, _ = distribution(delta, resamples)
    a_slow, b_slow = a > SLOW_LIMIT, b > SLOW_LIMIT
    only_a = a_slow & ~b_slow
    only_b = b_slow & ~a_slow
    per_round_slow_difference = 100 * (only_a.mean(axis=1) - only_b.mean(axis=1))
    return {"frame_deltas_ms": report, "frames_faster": int((delta < 0).sum()), "matched_frames": delta.size,
            "slow_only_a": int(only_a.sum()), "slow_only_b": int(only_b.sum()),
            "slow_rate_difference_pp": {
                "estimate": float(per_round_slow_difference.mean()),
                "ci95": np.percentile(per_round_slow_difference[resamples].mean(axis=1),
                                      [2.5, 97.5]).tolist()},
            "difference_of_p99s_ms": {
                "estimate": float(np.quantile(a, .99) - np.quantile(b, .99)),
                "ci95": np.percentile(boot_a[:, 4] - boot_b[:, 4], [2.5, 97.5]).tolist()}}


def dominant_stage(row):
    keys = ("select_cpu_ms", "demand_submit_cpu_ms", "packing_submit_cpu_ms", "predict_cpu_ms",
            "prefetch_submit_cpu_ms", "render_submit_cpu_ms", "render_wait_cpu_ms", "prefetch_tail_wait_cpu_ms")
    return max(keys, key=lambda key: float(row[key]))


def analyze(folder, telemetry, resample_count):
    experiment = json.loads((folder / "runs.json").read_text(encoding="utf-8"))
    rounds, frames = experiment["rounds"], experiment["frame_count"] - 1
    groups = {}
    stalls = defaultdict(Counter)
    slow_rows = []
    telemetry_times = [row["perf_ns"] for row in telemetry]
    with (folder / "frames.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["frame"])
            if frame == 0:
                continue
            key = (int(row["budget_mib"]), row["policy"])
            if key not in groups:
                groups[key] = np.full((rounds, frames), np.nan)
            index = int(row["round"]) - 1
            wall = float(row["wall_ms"])
            if not np.isnan(groups[key][index, frame - 1]):
                raise ValueError("Duplicate frame in paired sample")
            groups[key][index, frame - 1] = wall
            if wall <= SLOW_LIMIT:
                continue
            stage = dominant_stage(row)
            stalls[key][stage] += 1
            stamp = int(row["start_perf_ns"])
            pos = bisect.bisect_left(telemetry_times, stamp)
            candidates = telemetry[max(0, pos - 1):pos + 1]
            nearest = min(candidates, key=lambda sample: abs(sample["perf_ns"] - stamp))
            slow_rows.append({
                "round": index + 1, "policy": key[1], "budget_mib": key[0], "frame": frame,
                "wall_ms": wall, "largest_measured_host_stage": stage, "stage_ms": row[stage],
                "demand_page_faults": row["demand_page_faults"], "demand_bytes": row["upload_bytes"],
                "demand_window_ms": row["demand_window_ms"], "packing_window_ms": row["packing_window_ms"],
                "render_window_ms": row["render_window_ms"], "prefetch_window_ms": row["prefetch_window_ms"],
                "sample_distance_ms": abs(nearest["perf_ns"] - stamp) / 1e6,
                **{field: nearest.get(field) for field in TELEMETRY},
            })
    rng = np.random.default_rng(20261110)
    resamples = rng.integers(rounds, size=(resample_count, rounds))
    distributions, boot, data = {}, {}, {}
    for key, matrix in groups.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"Missing frame data for {key}")
        distributions[key], boot[key] = distribution(matrix, resamples)
        arm_runs = [r for r in experiment["runs"] if (r["budget_mib"], r["policy"]) == key]
        positions = Counter(r["position"] for r in arm_runs)
        order_size = experiment.get("order_size", 4)
        if positions != {position: rounds // order_size for position in range(1, order_size + 1)}:
            raise ValueError(f"Unbalanced policy positions for {key}: {positions}")
        states = {}
        for endpoint in ("before", "after"):
            states[endpoint] = {}
            for field in TELEMETRY:
                measurements = [r[endpoint][field] for r in arm_runs if r[endpoint][field] is not None]
                if measurements:
                    states[endpoint][field] = np.percentile(measurements, [0, 50, 100]).tolist()
        data[f"{key[0]}/{key[1]}"] = {
            "latency": distributions[key], "frames": matrix.size,
            "frames_over_33_3ms": int((matrix > SLOW_LIMIT).sum()),
            "largest_stage_counts": dict(stalls[key]),
            "warm_traffic_min_median_max_bytes": {
                field: np.percentile([r[field] for r in arm_runs], [0, 50, 100]).tolist()
                for field in TRAFFIC},
            "p99_by_round_ms": [r["p99_ms"] for r in arm_runs],
            "positions": dict(positions),
            "median_p99_by_position_ms": {
                position: float(np.median([r["p99_ms"] for r in arm_runs if r["position"] == position]))
                for position in positions},
            "run_state_min_median_max": states,
            "min_sampled_residency": min(
                r[which]["resident_pages"] / r[which]["sampled_pages"]
                for r in arm_runs for which in ("payload_before", "payload_after")),
            "total_run_page_faults": sum(r["after"]["page_faults"] - r["before"]["page_faults"] for r in arm_runs),
            "allocator_retries": sum(r["allocator_deltas"]["num_alloc_retries"] or 0 for r in arm_runs),
        }
    comparisons = {}
    baseline = "legacy" if experiment["diagnostic"] else "lru"
    base_budget = min(experiment["budgets_mib"])
    for key, matrix in groups.items():
        budget, policy = key
        if experiment.get("ablation"):
            candidates = [(budget, ref) for arm, ref in experiment["ablation_contrasts"] if arm == policy]
        else:
            candidates = [(budget, baseline)]
            if policy in DIAGNOSTIC_PREDECESSOR:
                candidates.append((budget, DIAGNOSTIC_PREDECESSOR[policy]))
            if policy in ("adaptive", "guarded"):
                candidates.append((budget, "fixed"))
        if budget != base_budget:
            candidates.append((base_budget, policy))
        for other in candidates:
            if other not in groups or other == key:
                continue
            name = f"{budget}/{policy} minus {other[0]}/{other[1]}"
            comparisons[name] = contrast(matrix, groups[other], boot[key], boot[other], resamples)
    report = {"track": experiment["track"], "rounds": rounds,
              "bootstrap_replicates": resample_count, "slow_threshold_ms": SLOW_LIMIT,
              "scope": "Matched warm frames of fixed paths; bootstrap resamples whole paired rounds",
              "groups": data, "comparisons": comparisons,
              "budget_positions": {
                  budget: dict(Counter(r["budget_position"] for r in experiment["runs"]
                                       if r["budget_mib"] == budget))
                  for budget in experiment["budgets_mib"]}}
    (folder / "paired_statistics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if slow_rows:
        with (folder / "slow_frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(slow_rows[0]))
            writer.writeheader()
            writer.writerows(slow_rows)
    print(f"{folder.name}: {len(groups)} arms, {len(slow_rows)} warm frames >33.3 ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("Bootstrap replicate count must be positive")
    telemetry = [json.loads(line) for line in (args.session / "telemetry.jsonl").read_text().splitlines()]
    telemetry = sorted((row for row in telemetry if "temperature_c" in row), key=lambda row: row["perf_ns"])
    if not telemetry:
        raise ValueError("No valid telemetry samples")
    for file in sorted(args.session.glob("*/runs.json")):
        analyze(file.parent, telemetry, args.resamples)


if __name__ == "__main__":
    main()
