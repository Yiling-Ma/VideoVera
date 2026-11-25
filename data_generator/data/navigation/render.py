"""Final data-construction component: data/navigation/render.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from data_generator.io import preserve
import argparse
import copy
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any
import numpy as np
from data_generator.data.navigation.actions import bearing, extract_route, file_sha256, wrap
from data_generator.data.navigation.renderer import FPS, SENSOR_HEIGHT_M, VideoWriter, make_simulator, scene_file
from data_generator.data.navigation.navmesh import canonical_actions, reference_path, render_route

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--priority-manifest', type=Path)
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[], help='Exclude source sidecars already used by another generated dataset.')
    parser.add_argument('--scenes-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--all-valid', action='store_true', help='Process every source trajectory and keep every route that passes validation.')
    parser.add_argument('--simple-count', type=int, default=0, help='Require this many 2-leg routes.')
    parser.add_argument('--medium-count', type=int, default=0, help='Require this many 3-leg routes.')
    parser.add_argument('--complex-count', type=int, default=0, help='Require this many 4+ leg routes.')
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--unique-scenes', action='store_true', help='Emit at most one route from each Matterport scene.')
    parser.add_argument('--gpu-device-id', type=int, default=0)
    parser.add_argument('--speed-scale', type=float, default=0.35)
    parser.add_argument('--min-leg-m', type=float, default=0.8)
    parser.add_argument('--min-turn-deg', type=float, default=30.0)
    parser.add_argument('--max-legs', type=int, default=6)
    parser.add_argument('--direct-ratio', type=float, default=1.03)
    parser.add_argument('--direct-slack-m', type=float, default=0.08)
    parser.add_argument('--max-near-black-ratio', type=float, default=0.1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not 0 < args.speed_scale <= 1:
        parser.error('--speed-scale must be in (0, 1]')
    stratified = args.simple_count + args.medium_count + args.complex_count
    if args.all_valid and stratified:
        parser.error('--all-valid cannot be combined with stratified count arguments')
    if stratified:
        args.count = stratified
    if min(args.simple_count, args.medium_count, args.complex_count) < 0:
        parser.error('complexity counts must be non-negative')
    return args

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def shortest_path(pathfinder: Any, habitat_sim: Any, start: np.ndarray, end: np.ndarray):
    query = habitat_sim.ShortestPath()
    query.requested_start = start
    query.requested_end = end
    if not pathfinder.find_path(query) or not math.isfinite(float(query.geodesic_distance)):
        raise RuntimeError('no_navmesh_path')
    return (float(query.geodesic_distance), [np.asarray(point, dtype=float) for point in query.points])

def direct_on_navmesh(pathfinder: Any, habitat_sim: Any, start: np.ndarray, end: np.ndarray, ratio: float, slack: float) -> bool:
    euclidean = float(np.linalg.norm(end - start))
    geodesic, _ = shortest_path(pathfinder, habitat_sim, start, end)
    if geodesic > euclidean * ratio + slack:
        return False
    steps = max(2, int(math.ceil(euclidean / 0.1)))
    return all((pathfinder.is_navigable(start + (end - start) * index / steps) for index in range(steps + 1)))

def simplify_geodesic(pathfinder: Any, habitat_sim: Any, points: list[np.ndarray], args: argparse.Namespace) -> list[np.ndarray]:
    result = [points[0]]
    index = 0
    while index < len(points) - 1:
        chosen = index + 1
        for candidate in range(len(points) - 1, index, -1):
            if direct_on_navmesh(pathfinder, habitat_sim, points[index], points[candidate], args.direct_ratio, args.direct_slack_m):
                chosen = candidate
                break
        result.append(points[chosen])
        index = chosen
    return result

def trim_short_endpoint_legs(points: list[np.ndarray], min_leg_m: float) -> list[np.ndarray]:
    """Drop only tiny start/end navmesh connector edges.

    Interior short edges may encode an obstacle boundary and are never removed.
    """
    result = list(points)
    while len(result) > 3:
        first = float(np.linalg.norm((result[1] - result[0])[[0, 2]]))
        last = float(np.linalg.norm((result[-1] - result[-2])[[0, 2]]))
        if first < min_leg_m:
            result.pop(0)
        elif last < min_leg_m:
            result.pop()
        else:
            break
    return result

def anchored_navmesh_path(payload: dict[str, Any], pathfinder: Any, habitat_sim: Any, start: np.ndarray, end: np.ndarray, args: argparse.Namespace) -> tuple[float, list[np.ndarray], str]:
    reference = payload.get('reference_path') or []
    anchors = []
    for item in reference:
        position = item.get('position_xyz_m') if isinstance(item, dict) else None
        if isinstance(position, list) and len(position) == 3:
            anchors.append(np.asarray(pathfinder.snap_point(np.asarray(position, dtype=float)), dtype=float))
    if len(anchors) < 3:
        distance, points = shortest_path(pathfinder, habitat_sim, start, end)
        return (distance, simplify_geodesic(pathfinder, habitat_sim, points, args), 'endpoint_geodesic')
    points: list[np.ndarray] = []
    total_distance = 0.0
    for left, right in zip(anchors, anchors[1:]):
        distance, section = shortest_path(pathfinder, habitat_sim, left, right)
        section = simplify_geodesic(pathfinder, habitat_sim, section, args)
        total_distance += distance
        points.extend(section if not points else section[1:])
    points = simplify_geodesic(pathfinder, habitat_sim, points, args)
    return (total_distance, points, 'reference_waypoint_geodesics')

def camera_route(points: list[np.ndarray], args: argparse.Namespace, initial_turn_back: bool=False, final_turn_back: bool=False) -> dict[str, Any]:
    camera_points = [[float(point[0]), float(point[1] + SENSOR_HEIGHT_M), float(point[2])] for point in points]
    headings = [bearing(left, right) for left, right in zip(camera_points, camera_points[1:])]
    lengths = [math.hypot(right[0] - left[0], right[2] - left[2]) for left, right in zip(camera_points, camera_points[1:])]
    if not 2 <= len(lengths) <= args.max_legs:
        raise RuntimeError(f'unclean_leg_count:{len(lengths)}')
    if min(lengths) < args.min_leg_m:
        raise RuntimeError(f'short_navmesh_leg:{min(lengths):.3f}')
    turns = [wrap(right - left) for left, right in zip(headings, headings[1:])]
    if any((abs(turn) < args.min_turn_deg for turn in turns)):
        raise RuntimeError(f'minor_required_navmesh_turn:{turns}')
    if any((abs(turn) >= 150.0 for turn in turns)):
        raise RuntimeError(f'u_turn_geodesic:{turns}')
    boundaries = [{'waypoint_index': index, 'frame_index': 0, 'position_xyz_m': point} for index, point in enumerate(camera_points)]
    legs = [{'distance_m': length, 'heading_deg': heading} for length, heading in zip(lengths, headings)]
    return {'boundaries': boundaries, 'legs': legs, 'start_heading_deg': wrap(headings[0] + 180.0) if initial_turn_back else headings[0], 'end_heading_deg': wrap(headings[-1] + 180.0) if final_turn_back else headings[-1], 'navmesh_turn_degrees': turns, 'initial_turn_back': initial_turn_back, 'final_turn_back': final_turn_back}

def complexity_for_leg_count(leg_count: int) -> str:
    if leg_count == 2:
        return 'simple'
    if leg_count == 3:
        return 'medium'
    return 'complex'

def source_order(input_dir: Path, priority_manifest: Path | None, exclude_manifests: list[Path], seed: int) -> list[Path]:
    used_geometry = []
    all_paths = {path.name: path for path in input_dir.glob('*.actions.json')}
    for manifest_path in exclude_manifests:
        manifest = read_json(manifest_path)
        for row in manifest.get('videos') or []:
            used = input_dir / str(row.get('source_sidecar') or '')
            if not used.is_file():
                raise FileNotFoundError('Previously rendered source sidecar missing: '+str(used))
            used_geometry.append(used)
            all_paths.pop(str(row.get('source_sidecar') or ''), None)
    priority = []
    if priority_manifest:
        manifest = read_json(priority_manifest)
        for row in manifest.get('videos') or []:
            name = str(row.get('source_sidecar') or '')
            if name in all_paths:
                priority.append(all_paths.pop(name))
    remainder = list(all_paths.values())
    random.Random(seed).shuffle(remainder)
    remainder.sort(key=lambda path: str((read_json(path).get('episode') or {}).get('scene_id') or ''))
    from data_generator.data.navigation.identity import unique_sources
    return unique_sources(priority + remainder, used_geometry)

def render_candidate(source_path: Path, payload: dict[str, Any], sim: Any, habitat_sim: Any, args: argparse.Namespace, index: int, allowed_complexities: set[str] | None=None) -> dict[str, Any]:
    source_frames = (payload.get('camera_trajectory') or {}).get('frames') or []
    if len(source_frames) < 2:
        raise RuntimeError('missing_measured_camera_endpoints')
    start_camera = np.asarray(source_frames[0]['position_xyz_m'], dtype=float)
    end_camera = np.asarray(source_frames[-1]['position_xyz_m'], dtype=float)
    start = start_camera.copy()
    end = end_camera.copy()
    start[1] -= SENSOR_HEIGHT_M
    end[1] -= SENSOR_HEIGHT_M
    start = np.asarray(sim.pathfinder.snap_point(start), dtype=float)
    end = np.asarray(sim.pathfinder.snap_point(end), dtype=float)
    geodesic_distance, points, path_source = anchored_navmesh_path(payload, sim.pathfinder, habitat_sim, start, end, args)
    points = trim_short_endpoint_legs(points, args.min_leg_m)
    initial_turn_back = bool((payload.get('source') or {}).get('initial_turn_back'))
    final_turn_back = bool((payload.get('source') or {}).get('final_turn_back'))
    route = camera_route(points, args, initial_turn_back=initial_turn_back, final_turn_back=final_turn_back)
    complexity = complexity_for_leg_count(len(route['legs']))
    if allowed_complexities is not None and complexity not in allowed_complexities:
        raise RuntimeError(f'complexity_bucket_full:{complexity}')
    scene = str((payload.get('episode') or {})['scene_id'])
    stem = f"navmesh_macro_{index:02d}_{scene}_{source_path.stem.removesuffix('.actions')}"
    video_path = args.output_dir / f'{stem}.mp4'
    sidecar_path = args.output_dir / f'{stem}.actions.json'
    writer = VideoWriter(video_path)
    failed = False
    try:
        render_route(sim, route, writer, args.speed_scale)
        if writer.max_near_black_ratio > args.max_near_black_ratio:
            raise RuntimeError(f'near_black:{writer.max_near_black_ratio:.4f}')
        for frame in writer.camera_frames:
            point = np.asarray(frame['position_xyz_m'], dtype=float)
            point[1] -= SENSOR_HEIGHT_M
            if not sim.pathfinder.is_navigable(point):
                raise RuntimeError(f"rendered_pose_off_navmesh:{frame['frame_index']}")
    except Exception:
        failed = True
        raise
    finally:
        writer.close()
        if failed:
            preserve(video_path)
    result = copy.deepcopy(payload)
    result['schema_version'] = 'r2r-canonical-macro-actions-v1'
    for key in ('route_plan_landmarks', 'janus_raw_action_sequence', 'controller_action_sequence', 'actions'):
        result.pop(key, None)
    result['video'] = {**(result.get('video') or {}), 'file': video_path.name, 'fps': FPS, 'frame_count': len(writer.camera_frames), 'duration_sec': len(writer.camera_frames) / FPS}
    result['camera_trajectory'] = {'basis': 'measured_camera_pose', 'coordinate_system': 'habitat_y_up_camera_minus_z_xyzw', 'capture_source': 'Habitat navmesh geodesic with separated turn and travel phases', 'video_sha256': file_sha256(video_path), 'frames': writer.camera_frames}
    episode = result.setdefault('episode', {})
    episode['start_rotation_xyzw'] = writer.camera_frames[0]['rotation_xyzw']
    result['rendering'] = {**(result.get('rendering') or {}), 'max_near_black_pixel_ratio': writer.max_near_black_ratio, 'motion_policy': 'navmesh-geodesic-macro-v1', 'speed_scale': args.speed_scale}
    verified = extract_route(result, video_path)
    result['canonical_actions'] = canonical_actions(verified)
    result['ordered_action_sequence'] = verified['forward_actions']
    result['reference_path'] = reference_path(verified)
    result['pose_route_status'] = 'navmesh_macro_pose_verified'
    result['navmesh_route'] = {'source': 'Habitat ShortestPath', 'path_source': path_source, 'geodesic_distance_m': geodesic_distance, 'simplified_agent_points_xyz_m': [point.tolist() for point in points], 'turn_degrees': route['navmesh_turn_degrees'], 'direct_ratio': args.direct_ratio, 'direct_slack_m': args.direct_slack_m}
    result['source'] = {**(result.get('source') or {}), 'parent_sidecar': source_path.name, 'macro_policy': 'collision-aware-navmesh-geodesic-v1'}
    episode['episode_id'] = f"{episode.get('episode_id')}:navmesh:{index}"
    episode['trajectory_id'] = f"{episode.get('trajectory_id')}:navmesh:{index}"
    write_json(sidecar_path, result)
    return {'file': video_path.name, 'action_file': sidecar_path.name, 'scene_id': scene, 'episode_id': episode['episode_id'], 'trajectory_id': episode['trajectory_id'], 'ordered_action_sequence': verified['forward_actions'], 'source_sidecar': source_path.name, 'turn_degrees': route['navmesh_turn_degrees'], 'leg_count': len(route['legs']), 'complexity': complexity}

def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.scenes_dir = args.scenes_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.priority_manifest = args.priority_manifest.expanduser().resolve() if args.priority_manifest else None
    args.exclude_manifest = [path.expanduser().resolve() for path in args.exclude_manifest]
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'Output exists: {args.output_dir}; pass --overwrite')
        preserve(args.output_dir)
    args.output_dir.mkdir(parents=True)
    habitat_sim = __import__('habitat_sim')
    emitted = []
    failures = []
    requested = {'simple': args.simple_count, 'medium': args.medium_count, 'complex': args.complex_count}
    stratified = any(requested.values())
    accepted = {key: 0 for key in requested}
    accepted_scenes: set[str] = set()
    sim = None
    loaded_scene = None
    for source_path in source_order(args.input_dir, args.priority_manifest, args.exclude_manifest, args.seed):
        payload = read_json(source_path)
        scene = str((payload.get('episode') or {}).get('scene_id') or '')
        try:
            if args.unique_scenes and scene in accepted_scenes:
                raise RuntimeError('scene_already_selected')
            allowed = {key for key, count in requested.items() if accepted[key] < count} if stratified else None
            if sim is None or loaded_scene != scene:
                if sim is not None:
                    sim.close()
                sim = make_simulator(habitat_sim, scene_file(args.scenes_dir, scene), args.gpu_device_id)
                loaded_scene = scene
            record = render_candidate(source_path, payload, sim, habitat_sim, args, len(emitted) + 1, allowed)
            emitted.append(record)
            accepted[record['complexity']] += 1
            accepted_scenes.add(scene)
            progress = str(len(emitted)) if args.all_valid else f'{len(emitted)}/{args.count}'
            print(f'[{progress}] {source_path.name}', flush=True)
        except Exception as exc:
            failures.append({'source': source_path.name, 'reason': str(exc)})
            print(f'[skip] {source_path.name}: {exc}', flush=True)
            prefix = f'navmesh_macro_{len(emitted) + 1:02d}_'
            for partial in args.output_dir.glob(f'{prefix}*'):
                preserve(partial)
        if not args.all_valid and len(emitted) >= args.count:
            break
    if sim is not None:
        sim.close()
    write_json(args.output_dir / 'manifest.json', {'schema_version': 'r2r-navmesh-macro-pilot-v1', 'videos': emitted, 'failures': failures, 'complexity_counts': accepted, 'selection_mode': 'all_valid' if args.all_valid else 'count_limited', 'source_candidates': len(source_order(args.input_dir, args.priority_manifest, args.exclude_manifest, args.seed))})
    if not args.all_valid and len(emitted) != args.count:
        raise RuntimeError(f'Rendered only {len(emitted)}/{args.count}; failures={failures[-10:]}')
    print(json.dumps({'rendered': len(emitted), 'failed_candidates': len(failures)}))
if __name__ == '__main__':
    main()
