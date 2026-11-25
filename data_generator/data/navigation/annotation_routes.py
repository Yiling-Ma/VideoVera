"""Final data-construction component: data/navigation/annotation_routes.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from data_generator.io import preserve
import argparse
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
SENSOR_HEIGHT_M = 1.5
DEFAULT_INITIAL_TURN_BACK_RATIO = 0.19

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--dataset-name', default='r2r')
    parser.add_argument('--stem-prefix', default='r2rsub')
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--min-legs', type=int, default=2)
    parser.add_argument('--max-legs', type=int, default=3)
    parser.add_argument('--min-leg-m', type=float, default=0.8)
    parser.add_argument('--min-turn-deg', type=float, default=30.0)
    parser.add_argument('--max-turn-deg', type=float, default=150.0)
    parser.add_argument('--initial-turn-back-ratio', type=float, default=DEFAULT_INITIAL_TURN_BACK_RATIO, help='Deterministic fraction of subroutes that start facing opposite the first travel leg. The rendered route then begins with a real in-place TURN_BACK; 0.19 matches the ReVSI route-planning distribution.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.shards < 1:
        parser.error('--shards must be positive')
    if not 2 <= args.min_legs <= args.max_legs:
        parser.error('Require 2 <= --min-legs <= --max-legs')
    if not 0.0 <= args.initial_turn_back_ratio <= 1.0:
        parser.error('--initial-turn-back-ratio must be between 0 and 1')
    return args

def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0

def select_initial_turn_back(dataset_name: str, scene: str, trajectory: str, start_index: int, end_index: int, ratio: float) -> bool:
    """Select start-facing augmentation reproducibly without global RNG state."""
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    identity = f'{dataset_name}:{scene}:{trajectory}:{start_index}:{end_index}:initial-turn-back-v1'
    digest = hashlib.sha256(identity.encode('utf-8')).digest()
    unit_interval = int.from_bytes(digest[:8], 'big') / float(1 << 64)
    return unit_interval < ratio

def scene_id(row: dict[str, Any]) -> str:
    return Path(str(row['scene_id'])).stem

def instruction_text(row: dict[str, Any]) -> str:
    instruction = row.get('instruction') or ''
    if isinstance(instruction, dict):
        return str(instruction.get('instruction_text') or '').strip()
    return str(instruction).strip()

def benchmark_trajectories(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    excluded: set[tuple[str, str]] = set()
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('dataset_source') != 'r2r':
                continue
            metadata = row.get('metadata') or {}
            trajectory = metadata.get('trajectory_id')
            scene = row.get('scene_id') or metadata.get('scene_id')
            if trajectory is not None and scene:
                excluded.add((Path(str(scene)).stem, str(trajectory).split(':', 1)[0]))
    return excluded

def heading(left: list[float], right: list[float]) -> float:
    return math.degrees(math.atan2(-(right[0] - left[0]), -(right[2] - left[2])))

def clean_window(points: list[list[float]], args: argparse.Namespace) -> bool:
    distances = [math.hypot(right[0] - left[0], right[2] - left[2]) for left, right in zip(points, points[1:])]
    headings = [heading(left, right) for left, right in zip(points, points[1:])]
    turns = [wrap(right - left) for left, right in zip(headings, headings[1:])]
    return min(distances) >= args.min_leg_m and all((args.min_turn_deg <= abs(turn) < args.max_turn_deg for turn in turns))

def yaw_quaternion(heading_deg: float) -> list[float]:
    half = math.radians(heading_deg) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]

def make_sidecar(row: dict[str, Any], points: list[list[float]], start_index: int, end_index: int, dataset_name: str, initial_turn_back: bool=False) -> dict[str, Any]:
    scene = scene_id(row)
    episode = str(row['episode_id'])
    trajectory = str(row['trajectory_id'])
    first_leg_heading = heading(points[0], points[1])
    start_heading = wrap(first_leg_heading + 180.0) if initial_turn_back else first_leg_heading
    camera_points = [[p[0], p[1] + SENSOR_HEIGHT_M, p[2]] for p in points]
    distances = [0.0] + [math.hypot(right[0] - left[0], right[2] - left[2]) for left, right in zip(points, points[1:])]
    return {'schema_version': 'r2r-annotation-subroute-v1', 'episode': {'episode_id': f'{episode}:subroute:{start_index}-{end_index}', 'scene_id': scene, 'trajectory_id': f'{dataset_name}:{trajectory}:subroute:{start_index}-{end_index}', 'instruction': instruction_text(row), 'start_position_xyz_m': points[0], 'start_rotation_xyzw': yaw_quaternion(start_heading), 'goal_position_xyz_m': points[-1]}, 'reference_path': [{'waypoint_index': index, 'position_xyz_m': point, 'segment_distance_xz_m': distances[index]} for index, point in enumerate(points)], 'camera_trajectory': {'basis': 'annotation_endpoints_pending_navmesh_render', 'coordinate_system': 'habitat_y_up_camera_minus_z_xyzw', 'frames': [{'frame_index': 0, 'timestamp_sec': 0.0, 'position_xyz_m': camera_points[0], 'rotation_xyzw': yaw_quaternion(start_heading)}, {'frame_index': 1, 'timestamp_sec': 0.125, 'position_xyz_m': camera_points[-1], 'rotation_xyzw': yaw_quaternion(heading(points[-2], points[-1]))}]}, 'video': {'file': '', 'fps': 8, 'frame_count': 0, 'duration_sec': 0.0}, 'source': {'generator': f'JanusVLN {dataset_name.upper()} annotation subroute', 'dataset_name': dataset_name, 'original_episode_id': episode, 'original_trajectory_id': trajectory, 'subroute_start_index': start_index, 'subroute_end_index': end_index, 'initial_turn_back': initial_turn_back, 'initial_heading_offset_deg': 180.0 if initial_turn_back else 0.0}, 'pose_route_status': 'annotation_subroute_pending_navmesh_render'}

def main() -> None:
    args = parse_args()
    args.episodes = args.episodes.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.benchmark = args.benchmark.expanduser().resolve() if args.benchmark else None
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'Output exists: {args.output_dir}; pass --overwrite')
        preserve(args.output_dir)
    args.output_dir.mkdir(parents=True)
    shard_dirs = [args.output_dir / f'shard_{index:02d}' for index in range(args.shards)]
    for directory in shard_dirs:
        directory.mkdir()
    with gzip.open(args.episodes, 'rt', encoding='utf-8') as handle:
        payload = json.load(handle)
    rows = payload.get('episodes', payload)
    excluded = benchmark_trajectories(args.benchmark)
    seen: set[tuple[str, str]] = set()
    manifests: list[list[dict[str, Any]]] = [[] for _ in shard_dirs]
    counts = Counter()
    for row in rows:
        from data_generator.data.navigation.identity import adapt_episode
        row = adapt_episode(row)
        scene = scene_id(row)
        trajectory = str(row['trajectory_id'])
        key = (scene, trajectory)
        if key in seen:
            continue
        seen.add(key)
        if key in excluded:
            counts['benchmark_excluded_trajectories'] += 1
            continue
        full_path = row.get('reference_path') or []
        if len(full_path) < args.min_legs + 1:
            continue
        shard = int(hashlib.sha256(scene.encode()).hexdigest(), 16) % args.shards
        for legs in range(args.min_legs, args.max_legs + 1):
            for start in range(0, len(full_path) - legs):
                end = start + legs
                points = [[float(value) for value in point] for point in full_path[start:end + 1]]
                if not clean_window(points, args):
                    counts['geometry_prefilter_rejected'] += 1
                    continue
                stem = f'{args.stem_prefix}_{scene}_traj-{trajectory}_w-{start:02d}-{end:02d}'
                initial_turn_back = select_initial_turn_back(args.dataset_name, scene, trajectory, start, end, args.initial_turn_back_ratio)
                sidecar = make_sidecar(row, points, start, end, args.dataset_name, initial_turn_back=initial_turn_back)
                path = shard_dirs[shard] / f'{stem}.actions.json'
                path.write_text(json.dumps(sidecar, ensure_ascii=False) + '\n', encoding='utf-8')
                manifests[shard].append({'action_file': path.name, 'scene_id': scene, 'original_trajectory_id': trajectory, 'subroute': [start, end], 'leg_count': legs, 'initial_turn_back': initial_turn_back})
                counts[f'legs_{legs}'] += 1
                counts[f'initial_turn_back_{str(initial_turn_back).lower()}'] += 1
    for index, (directory, candidates) in enumerate(zip(shard_dirs, manifests)):
        manifest = {'schema_version': 'r2r-annotation-subroute-candidates-v1', 'shard_index': index, 'shard_count': args.shards, 'candidates': candidates}
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = {'dataset_name': args.dataset_name, 'unique_source_trajectories': len(seen), 'benchmark_trajectory_keys': len(excluded), 'candidate_count': sum((len(rows) for rows in manifests)), 'shard_counts': [len(rows) for rows in manifests], 'counts': counts}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=dict) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, default=dict))
if __name__ == '__main__':
    main()
