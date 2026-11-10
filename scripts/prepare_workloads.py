"""Build camera paths and screen cache budgets with the actual block selector."""

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np

from .graphdeco_scene import load_cameras
from .spatial_blocks import Blocks
from .stream_blocks import BlockCache, MIB


def rotation_angle(rotation):
    return np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def interpolate_rotation(start, end, fraction):
    relative = start.T @ end
    angle = rotation_angle(relative)
    if angle < 1e-7:
        return start
    skew = (relative - relative.T) / (2 * np.sin(angle))
    partial = fraction * angle
    return start @ (np.eye(3) + np.sin(partial) * skew + (1 - np.cos(partial)) * (skew @ skew))


def path_anchors(centers, rotations):
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    diameter = float(distances.max())
    angles = rotation_angle(rotations[:, None].transpose(0, 1, 3, 2) @ rotations[None])
    max_step = 0.12 * diameter
    max_angle = np.radians(60)
    graph = (distances <= max_step) & (angles <= max_angle)
    np.fill_diagonal(graph, False)
    neighbors = [np.flatnonzero(row).tolist() for row in graph]
    unseen = set(range(len(centers)))
    best_span, start, goal = 0, None, None
    while unseen:
        component = [min(unseen)]
        unseen.remove(component[0])
        for node in component:
            for neighbor in neighbors[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.append(neighbor)
        subdistances = distances[np.ix_(component, component)]
        i, j = np.unravel_index(subdistances.argmax(), subdistances.shape)
        if subdistances[i, j] > best_span:
            best_span, start, goal = subdistances[i, j], component[i], component[j]
    if start is None:
        raise ValueError("The available cameras have no connected spatial route")
    best = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        if node == goal:
            break
        for neighbor in neighbors[node]:
            score = cost + distances[node, neighbor] / max_step + angles[node, neighbor] / max_angle
            if score < best.get(neighbor, math.inf):
                best[neighbor] = score
                previous[neighbor] = node
                heapq.heappush(queue, (score, neighbor))
    route = [goal]
    while route[-1] != start:
        route.append(previous[route[-1]])
    return route[::-1], diameter


def continuous_path(views, intrinsics, names):
    camera_to_world = np.linalg.inv(views.astype(np.float64))
    centers = camera_to_world[:, :3, 3]
    rotations = camera_to_world[:, :3, :3]
    anchors, diameter = path_anchors(centers, rotations)
    route_views = [views[anchors[0]]]
    route_intrinsics = [intrinsics[anchors[0]]]
    for start, end in zip(anchors, anchors[1:]):
        distance = np.linalg.norm(centers[end] - centers[start])
        angle = rotation_angle(rotations[start].T @ rotations[end])
        steps = max(1, math.ceil(distance / (diameter / 120)), math.ceil(angle / np.radians(4)))
        for step in range(1, steps + 1):
            fraction = step / steps
            rotation = interpolate_rotation(rotations[start], rotations[end], fraction)
            center = (1 - fraction) * centers[start] + fraction * centers[end]
            view = np.eye(4, dtype=np.float32)
            view[:3, :3] = rotation.T
            view[:3, 3] = -(rotation.T @ center)
            route_views.append(view)
            calibration = intrinsics[start].copy()
            calibration[0, 0] += fraction * (intrinsics[end, 0, 0] - calibration[0, 0])
            calibration[1, 1] += fraction * (intrinsics[end, 1, 1] - calibration[1, 1])
            route_intrinsics.append(calibration)
    route_info = {"anchor_cameras": [names[index] for index in anchors],
                  "endpoint_distance": float(np.linalg.norm(centers[anchors[0]] - centers[anchors[-1]])),
                  "capture_diameter": diameter}
    return np.stack(route_views), np.stack(route_intrinsics), route_info


def screen(blocks, views, intrinsics, width, height):
    requests = [blocks.visible(view, calibration, width, height)
                for view, calibration in zip(views, intrinsics)]
    if not all(requests):
        raise ValueError("A trajectory frame has no required blocks")
    demands = [int(blocks.sizes[request].sum()) for request in requests]
    total = int(blocks.sizes.sum())
    peak = max(demands)
    window = min(30, len(requests))
    union = [int(blocks.sizes[list(set().union(*requests[index:index + window]))].sum())
             for index in range(len(requests) - window + 1)]
    new = [int(blocks.sizes[list(set(curr) - set(prev))].sum())
           for prev, curr in zip(requests, requests[1:])]
    budgets = sorted({math.ceil(total * ratio / MIB) for ratio in (0.25, 0.50, 0.75)}
                     | {math.ceil(peak / MIB) + margin for margin in (4, 16, 32)})
    outcomes = []
    for budget_mib in budgets:
        cap = budget_mib * MIB
        infeasible = sum(size > cap for size in demands)
        outcome = {"budget_mib": budget_mib, "infeasible_frames": infeasible}
        if not infeasible:
            cache = BlockCache(blocks, cap, upload=lambda block_id: None)
            replay = [cache.prepare(request) for request in requests]
            warm = replay[1:]
            if warm:
                misses = [row["upload_bytes"] for row in warm]
                outcome.update({"warm_upload_mib": sum(misses) / MIB,
                                "warm_miss_frames": sum(value > 0 for value in misses),
                                "warm_p95_upload_mib": float(np.percentile(misses, 95) / MIB),
                                "warm_max_upload_mib": max(misses) / MIB,
                                "warm_byte_hit_rate": sum(row["hit_bytes"] for row in warm)
                                / sum(row["required_bytes"] for row in warm)})
        outcomes.append(outcome)
    report = {"frames": len(requests), "payload_mib": total / MIB,
              "median_required_mib": float(np.median(demands) / MIB), "peak_required_mib": peak / MIB,
              "median_required_fraction": float(np.median(demands) / total),
              "peak_required_fraction": peak / total,
              "p95_adjacent_new_mib": float(np.percentile(new, 95) / MIB) if new else 0,
              "max_window_union_mib": max(union) / MIB, "window_frames": window,
              "budget_screen": outcomes, "measurement": "CPU simulation of block requests and byte-budgeted LRU"}
    return report, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    width, height = 1280, 720
    views, intrinsics, names = load_cameras(args.scene / "cameras.json", width, height)
    route, route_intrinsics, metadata = continuous_path(views, intrinsics, names)
    return_views = np.concatenate((route, route[-2::-1]))
    return_intrinsics = np.concatenate((route_intrinsics, route_intrinsics[-2::-1]))
    blocks = Blocks(args.blocks)
    args.output.mkdir(parents=True, exist_ok=True)
    tracks = {"capture_order": (views, intrinsics, names),
              "continuous": (route, route_intrinsics, [f"path_{i:04d}" for i in range(len(route))]),
              "return": (return_views, return_intrinsics, [f"return_{i:04d}" for i in range(len(return_views))])}
    summary = {"scene": args.scene.name, "resolution": [width, height], "continuous_path": metadata,
               "tracks": {}}
    for key, (track_views, track_intrinsics, labels) in tracks.items():
        result, requests = screen(blocks, track_views, track_intrinsics, width, height)
        summary["tracks"][key] = result
        trajectory = {"resolution": [width, height], "camera_names": labels,
                      "viewmats": track_views.tolist(), "intrinsics": track_intrinsics.tolist()}
        (args.output / f"{key}.json").write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
        (args.output / f"{key}_requests.json").write_text(json.dumps(requests) + "\n", encoding="utf-8")
        print(f"{key}: {result['frames']} frames, median {100*result['median_required_fraction']:.1f}%, "
              f"peak {100*result['peak_required_fraction']:.1f}%, "
              f"adjacent new p95 {result['p95_adjacent_new_mib']:.2f} MiB", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
