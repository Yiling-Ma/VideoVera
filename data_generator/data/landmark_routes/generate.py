"""Final data-construction component: data/landmark_routes/generate.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from data_generator.io import preserve
import argparse
import bisect
import itertools
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
from PIL import Image
SCRIPT_DIR = Path(__file__).resolve().parent
VISIBILITY_VENDOR = SCRIPT_DIR / 'vendor' / 'arkit_visibility'
from data_generator.sources.arkit_visibility import PoseSample, TimestampedPath, evaluate_obb_depth_support, list_timestamped_files, load_depth_png_meters, nearest_timestamped, obb_corners, parse_pincam, parse_trajectory, project_obb, scale_intrinsics
from data_generator.sources.spatial_geometry.arkitscenes import load_arkitscenes_objects
TURN_VALUES = ('Turn left', 'Turn right', 'Turn back')
EXCLUDED_LABELS = {'unknown', 'wall', 'floor', 'ceiling', 'door', 'window', 'shelf'}

@dataclass(frozen=True)
class VisibleAnchor:
    object_key: str
    uid: str
    label: str
    centroid: tuple[float, float, float]
    frame_index: int
    timestamp: float
    camera_xy: tuple[float, float]
    max_area_ratio: float
    visible_frame_count: int
    visible_frame_indices: tuple[int, ...] = ()
    max_depth_support_ratio: float = 0.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate high-confidence ReVSI-style route-plan QA from ARKitScenes 3DOD scenes.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, action='append', default=[])
    parser.add_argument('--target', type=int, default=10)
    parser.add_argument('--max-per-scene', type=int, default=2)
    parser.add_argument('--turn-sequence-quotas', type=Path, help='Training-only JSON answer-text to count mapping. Never fill shortfalls with majority answers.')
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--video-fps', type=float, default=10.0)
    parser.add_argument('--clip-padding-sec', type=float, default=0.5)
    parser.add_argument('--min-clip-seconds', type=float, default=8.0)
    parser.add_argument('--max-clip-seconds', type=float, default=40.0)
    parser.add_argument('--max-frame-gap', type=float, default=0.2)
    parser.add_argument('--min-horizontal-view-ratio', type=float, default=0.75)
    parser.add_argument('--max-pitch-p90', type=float, default=70.0)
    parser.add_argument('--min-visible-frames', type=int, default=3)
    parser.add_argument('--min-area-ratio', type=float, default=0.04)
    parser.add_argument('--max-area-ratio', type=float, default=0.45)
    parser.add_argument('--depth-support-ratio', type=float, default=0.1)
    parser.add_argument('--min-depth-coverage', type=float, default=0.5)
    parser.add_argument('--min-in-frame-fraction', type=float, default=0.6)
    parser.add_argument('--min-corner-depth', type=float, default=0.1)
    parser.add_argument('--model-sample-frames', type=int, default=24)
    parser.add_argument('--min-sampled-visible-frames', type=int, default=1)
    parser.add_argument('--include-reverse-routes', action='store_true', help='Also ask the geometrically verified reverse navigation problem for a connected landmark path. The video remains the same continuous scan and every named object must pass all visibility gates.')
    parser.add_argument('--reverse-routes-only', action='store_true', help='Generate only reverse-direction questions for verified connected paths.')
    parser.add_argument('--allow-shortfall', action='store_true', help='Export every valid candidate when a streaming batch contains fewer than --target.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def nearest_pose(timestamp: float, poses: Sequence[PoseSample], tolerance: float=0.06) -> PoseSample | None:
    stamps = [pose.timestamp for pose in poses]
    index = bisect.bisect_left(stamps, timestamp)
    candidates = poses[max(0, index - 1):min(len(poses), index + 1)]
    if not candidates:
        return None
    best = min(candidates, key=lambda pose: abs(pose.timestamp - timestamp))
    return best if abs(best.timestamp - timestamp) <= tolerance else None

def camera_center_xy(pose: PoseSample) -> tuple[float, float]:
    rotation = pose.world_to_camera[:3, :3]
    translation = pose.world_to_camera[:3, 3]
    center = -(rotation.T @ translation)
    return (float(center[0]), float(center[1]))

def scene_paths(scene_dir: Path) -> dict[str, Path]:
    scene_id = scene_dir.name
    frame_root = scene_dir / f'{scene_id}_frames'
    return {'annotation': scene_dir / f'{scene_id}_3dod_annotation.json', 'trajectory': frame_root / 'lowres_wide.traj', 'rgb': frame_root / 'lowres_wide', 'depth': frame_root / 'lowres_depth', 'intrinsics': frame_root / 'lowres_wide_intrinsics'}

def discover_scenes(root: Path) -> list[Path]:
    return sorted((path.parent for path in root.glob('*/*_3dod_annotation.json') if path.parent.name in path.name))

def load_benchmark_scene_ids(paths: Iterable[Path]) -> set[str]:
    scene_ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            values = [row.get('scene_id'), row.get('video_id'), row.get('id'), row.get('video_path')]
            for value in values:
                match = re.search('(?<!\\d)(\\d{8})(?!\\d)', str(value or ''))
                if match:
                    scene_ids.add(match.group(1))
    return scene_ids

def _frame_index(rows: Sequence[TimestampedPath], path: Path) -> int:
    for index, row in enumerate(rows):
        if row.path == path:
            return index
    raise ValueError(f'Frame not found in timeline: {path}')

def collect_visible_anchors(scene_dir: Path, min_visible_frames: int, min_area_ratio: float, max_area_ratio: float, depth_support_ratio: float, min_depth_coverage: float, min_in_frame_fraction: float, min_corner_depth: float) -> tuple[list[VisibleAnchor], list[TimestampedPath], list[tuple[float, float]]]:
    paths = scene_paths(scene_dir)
    if not all((path.exists() for path in paths.values())):
        return ([], [], [])
    objects = load_arkitscenes_objects(paths['annotation'])
    label_counts = Counter((obj.label for obj in objects))
    objects = [obj for obj in objects if label_counts[obj.label] == 1 and obj.label.lower() not in EXCLUDED_LABELS]
    if len(objects) < 4:
        return ([], [], [])
    poses = parse_trajectory(paths['trajectory'])
    rgb_rows = list_timestamped_files(paths['rgb'], '.png')
    depth_rows = list_timestamped_files(paths['depth'], '.png')
    intrinsic_rows = list_timestamped_files(paths['intrinsics'], '.pincam')
    if not rgb_rows or not depth_rows or (not intrinsic_rows):
        return ([], [], [])
    pose_by_frame: list[PoseSample | None] = [nearest_pose(row.timestamp, poses) for row in rgb_rows]
    camera_path = [camera_center_xy(pose) if pose is not None else (math.nan, math.nan) for pose in pose_by_frame]
    visible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    corners = {obj.object_key: obb_corners(obj) for obj in objects}
    for frame_index, rgb_row in enumerate(rgb_rows):
        pose = pose_by_frame[frame_index]
        if pose is None:
            continue
        depth_row = nearest_timestamped(rgb_row.timestamp, depth_rows, 0.06)
        intrinsic_row = nearest_timestamped(rgb_row.timestamp, intrinsic_rows, 0.06)
        if depth_row is None or intrinsic_row is None:
            continue
        try:
            depth = load_depth_png_meters(depth_row.path)
            intrinsics = parse_pincam(intrinsic_row.path, intrinsic_row.timestamp)
            matrix = scale_intrinsics(intrinsics, depth.shape[1], depth.shape[0])
        except Exception:
            continue
        for obj in objects:
            projection = project_obb(corners[obj.object_key], pose.world_to_camera, matrix, depth.shape[1], depth.shape[0])
            if projection is None:
                continue
            if not min_area_ratio <= projection['bbox_area_ratio'] <= max_area_ratio:
                continue
            if projection['in_frame_fraction'] < min_in_frame_fraction:
                continue
            if not projection['all_corners_in_front'] or projection['min_corner_depth_m'] < min_corner_depth:
                continue
            support = evaluate_obb_depth_support(depth, obj, pose.world_to_camera, matrix, projection, 0.12)
            if not support['visible'] or support['valid_depth_coverage'] < min_depth_coverage or support['depth_support_ratio'] < depth_support_ratio:
                continue
            visible[obj.object_key].append({'frame_index': frame_index, 'timestamp': rgb_row.timestamp, 'area_ratio': float(projection['bbox_area_ratio']), 'depth_support_ratio': float(support['depth_support_ratio']), 'camera_xy': camera_center_xy(pose)})
    anchors: list[VisibleAnchor] = []
    by_key = {obj.object_key: obj for obj in objects}
    for object_key, frames in visible.items():
        if len(frames) < min_visible_frames:
            continue
        best = max(frames, key=lambda row: (row['area_ratio'], -row['frame_index']))
        obj = by_key[object_key]
        anchors.append(VisibleAnchor(object_key=object_key, uid=obj.uid, label=obj.label.replace('_', ' '), centroid=tuple((float(value) for value in obj.centroid)), frame_index=int(best['frame_index']), timestamp=float(best['timestamp']), camera_xy=tuple((float(value) for value in best['camera_xy'])), max_area_ratio=float(best['area_ratio']), visible_frame_count=len(frames), visible_frame_indices=tuple((int(row['frame_index']) for row in frames)), max_depth_support_ratio=max((float(row['depth_support_ratio']) for row in frames))))
    return (sorted(anchors, key=lambda row: (row.frame_index, row.label)), rgb_rows, camera_path)

def vec(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    return np.asarray((b[0] - a[0], b[1] - a[1]), dtype=np.float64)

def signed_angle_deg(source: Sequence[float], target: Sequence[float]) -> float:
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if np.linalg.norm(a) < 1e-06 or np.linalg.norm(b) < 1e-06:
        raise ValueError('Degenerate direction vector')
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cross = a[0] * b[1] - a[1] * b[0]
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.atan2(cross, dot))

def classify_turn(angle: float) -> str | None:
    magnitude = abs(angle)
    if 50.0 <= magnitude <= 125.0:
        return 'Turn left' if angle > 0 else 'Turn right'
    if magnitude >= 150.0:
        return 'Turn back'
    return None

def smoothed_path_length(camera_path: Sequence[tuple[float, float]], start: int, end: int, stride: int=5) -> float:
    if end <= start:
        return math.inf
    indices = list(range(start, end + 1, stride))
    if indices[-1] != end:
        indices.append(end)
    points = [np.asarray(camera_path[index], dtype=np.float64) for index in indices]
    if any((not np.all(np.isfinite(point)) for point in points)):
        return math.inf
    return float(sum((np.linalg.norm(right - left) for left, right in zip(points, points[1:]))))

def route_is_supported(route: Sequence[VisibleAnchor], camera_path: Sequence[tuple[float, float]]) -> bool:
    for start, end in zip(route, route[1:]):
        if end.frame_index - start.frame_index < 8:
            return False
        direct = float(np.linalg.norm(vec(start.centroid, end.centroid)))
        if not 0.7 <= direct <= 5.5:
            return False
        path_length = smoothed_path_length(camera_path, start.frame_index, end.frame_index)
        if not math.isfinite(path_length) or path_length > max(1.5, direct * 3.0):
            return False
    return True

def route_turns(route: Sequence[VisibleAnchor], facing: VisibleAnchor) -> tuple[list[str], list[float]] | None:
    orientation = vec(route[0].centroid, facing.centroid)
    turns: list[str] = []
    angles: list[float] = []
    for start, end in zip(route, route[1:]):
        direction = vec(start.centroid, end.centroid)
        try:
            angle = signed_angle_deg(orientation, direction)
        except ValueError:
            return None
        turn = classify_turn(angle)
        if turn is None:
            return None
        turns.append(turn)
        angles.append(angle)
        orientation = direction
    return (turns, angles)

def candidate_routes(anchors: Sequence[VisibleAnchor], camera_path: Sequence[tuple[float, float]], include_reverse_routes: bool=False, reverse_routes_only: bool=False) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for route_length in (3, 4):
        for route in itertools.combinations(anchors, route_length):
            if not route_is_supported(route, camera_path):
                continue
            route_variants = [] if reverse_routes_only else [(route, 'forward')]
            if include_reverse_routes or reverse_routes_only:
                route_variants.append((tuple(reversed(route)), 'reverse'))
            for route_variant, direction in route_variants:
                for facing in anchors:
                    if facing in route_variant:
                        continue
                    facing_distance = float(np.linalg.norm(vec(route_variant[0].centroid, facing.centroid)))
                    if not 0.75 <= facing_distance <= 6.0:
                        continue
                    result = route_turns(route_variant, facing)
                    if result is None:
                        continue
                    turns, angles = result
                    candidates.append({'route': list(route_variant), 'facing': facing, 'turns': turns, 'angles': angles, 'route_direction_relative_to_scan': direction})
    return candidates

def answer_text(turns: Sequence[str]) -> str:
    return ', '.join(turns)

def select_with_quotas(scene_candidates, quotas, max_per_scene, rng):
    """Select genuine candidates only; unmet quotas remain explicit shortfalls."""
    if max_per_scene < 1 or not quotas or any((not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in quotas.values())):
        raise ValueError('Invalid turn-sequence quotas or scene cap')
    for answer in quotas:
        if not answer or any((t not in TURN_VALUES for t in answer.split(', '))):
            raise ValueError('Unrecognized quota answer: ' + answer)
    buckets = defaultdict(list)
    for scene, (candidates, rgb) in scene_candidates.items():
        for candidate in candidates:
            buckets[answer_text(candidate['turns'])].append((scene, candidate, rgb))
    for values in buckets.values():
        rng.shuffle(values)
    selected, scenes, achieved = ([], Counter(), Counter())
    order = sorted(quotas, key=lambda answer: len(buckets[answer]) / max(quotas[answer], 1))
    for answer in order:
        available = list(buckets[answer])
        while achieved[answer] < quotas[answer]:
            eligible = [i for i, value in enumerate(available) if scenes[value[0]] < max_per_scene]
            if not eligible:
                break
            i = min(eligible, key=lambda index: scenes[available[index][0]])
            value = available.pop(i)
            selected.append(value)
            scenes[value[0]] += 1
            achieved[answer] += 1
    rng.shuffle(selected)
    return (selected, {answer: target - achieved[answer] for answer, target in quotas.items() if achieved[answer] != target})

def make_choices(turns: Sequence[str], answer_index: int, rng: random.Random) -> tuple[dict[str, str], str]:
    correct = tuple(turns)
    if not correct or any((turn not in TURN_VALUES for turn in correct)):
        raise ValueError('Expected a nonempty sequence of recognized turns')
    if len(correct) == 1:
        distractors = [(turn,) for turn in TURN_VALUES if turn != correct[0]]
    else:
        positions = list(range(len(correct)))
        rng.shuffle(positions)
        split = rng.randrange(1, len(correct))
        first = set(positions[:split])
        alternate = [rng.choice([v for v in TURN_VALUES if v != turn]) for turn in correct]
        distractors = [tuple((alternate[i] if i in first and a or (i not in first and b) else turn for i, turn in enumerate(correct))) for a, b in ((1, 0), (0, 1), (1, 1))]
    rng.shuffle(distractors)
    options = distractors[:]
    options.insert(answer_index % (len(options) + 1), correct)
    letters = 'ABCD'[:len(options)]
    choices = {letter: answer_text(sequence) for letter, sequence in zip(letters, options)}
    return (choices, letters[options.index(correct)])

def build_question(route: Sequence[VisibleAnchor], facing: VisibleAnchor) -> str:
    actions: list[str] = []
    step = 1
    for destination in route[1:]:
        actions.append(f'{step}. [please fill in].')
        step += 1
        actions.append(f'{step}. Go forward until the {destination.label}.')
        step += 1
    action_block = '\n'.join(actions)
    return f"You are a robot beginning at the {route[0].label} and facing the {facing.label}. You want to navigate to the {route[-1].label}. You will perform the following actions (for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'):\n{action_block}\nYou have reached the final destination. Which option correctly fills the blanks in order?"

def build_reasoning(route: Sequence[VisibleAnchor], facing: VisibleAnchor, turns: Sequence[str]) -> list[str]:
    steps = [f'Use the {route[0].label} as the starting position.', f'At the start, face toward the {facing.label} to establish the initial orientation.']
    for index, (turn, destination) in enumerate(zip(turns, route[1:]), start=1):
        prefix = 'First' if index == 1 else 'Then'
        steps.append(f'{prefix}, {turn.lower()}.')
        steps.append(f'Go forward until the {destination.label}.')
    steps.append(f'The ordered missing turns are: {answer_text(turns)}.')
    return steps

def sky_up_rotation_deg(pose: PoseSample) -> int:
    """Return the nearest right-angle PIL rotation that makes world +Z point up."""
    world_up_in_camera = pose.world_to_camera[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    image_angle = math.degrees(math.atan2(world_up_in_camera[1], world_up_in_camera[0]))
    rotation = int(round((image_angle + 90.0) / 90.0) * 90) % 360
    return rotation

def camera_pitch_deg(pose: PoseSample) -> float:
    camera_forward_world = pose.world_to_camera[:3, :3].T @ np.asarray((0.0, 0.0, 1.0))
    camera_forward_world /= np.linalg.norm(camera_forward_world)
    return math.degrees(math.asin(float(np.clip(camera_forward_world[2], -1.0, 1.0))))

def candidate_clip_bounds(candidate: dict[str, Any], rows: Sequence[TimestampedPath], padding_sec: float) -> tuple[int, int]:
    anchors = [*candidate['route'], candidate['facing']]
    anchor_indices = [anchor.frame_index for anchor in anchors]
    if len(rows) < 2:
        return (0, max(0, len(rows) - 1))
    intervals = [right.timestamp - left.timestamp for left, right in zip(rows, rows[1:])]
    positive_intervals = [value for value in intervals if value > 0]
    median_interval = float(np.median(positive_intervals)) if positive_intervals else 0.1
    padding_frames = max(0, int(round(padding_sec / median_interval)))
    return (max(0, min(anchor_indices) - padding_frames), min(len(rows) - 1, max(anchor_indices) + padding_frames))

def optimized_continuous_clip_bounds(candidate: dict[str, Any], rows: Sequence[TimestampedPath], padding_sec: float, min_clip_seconds: float, max_clip_seconds: float, model_sample_frames: int, min_sampled_visible_frames: int) -> tuple[int, int]:
    """Choose one continuous interval whose uniform model view sees all landmarks.

    Only the interval boundaries may move.  The saved clip still contains
    every source frame between them, in order.  Searching boundary offsets is
    necessary because a fixed sampling grid can otherwise fall between a
    landmark's few visible frames.  This is not evidence-frame insertion: if
    no continuous interval has adequate uniform coverage, the caller rejects
    the candidate.
    """
    if not rows:
        return (0, -1)
    anchors = [*candidate['route'], candidate['facing']]
    anchor_indices = [anchor.frame_index for anchor in anchors]
    base_start, base_end = (min(anchor_indices), max(anchor_indices))
    if len(rows) == 1:
        return (0, 0)
    intervals = [right.timestamp - left.timestamp for left, right in zip(rows, rows[1:])]
    positive = [value for value in intervals if value > 0]
    median_interval = float(np.median(positive)) if positive else 0.1
    preferred_padding = max(0, int(round(padding_sec / median_interval)))
    alignment_margin = max(preferred_padding, int(round(2.0 / median_interval)))
    min_frames = max(2, int(math.ceil(min_clip_seconds / median_interval)) + 1)
    max_frames = max(2, int(math.floor(max_clip_seconds / median_interval)) + 1)
    required_extra = max(0, min_frames - (base_end - base_start + 1))
    left_limit = max(0, base_start - alignment_margin - required_extra)
    right_limit = min(len(rows) - 1, base_end + alignment_margin + required_extra)
    best: tuple[tuple[int, int, int, int], int, int] | None = None
    for start in range(left_limit, base_start + 1):
        minimum_end = max(base_end, start + min_frames - 1)
        maximum_end = min(right_limit, start + max_frames - 1)
        for end in range(minimum_end, maximum_end + 1):
            duration = rows[end].timestamp - rows[start].timestamp
            if not min_clip_seconds <= duration <= max_clip_seconds:
                continue
            sampled = select_evidence_frame_indices(candidate, start, end, model_sample_frames, min_sampled_visible_frames)
            hits = [sum((index in set(anchor.visible_frame_indices) for index in sampled)) for anchor in anchors]
            if min(hits, default=0) < min_sampled_visible_frames:
                continue
            padding_deviation = abs(base_start - start - preferred_padding) + abs(end - base_end - preferred_padding)
            score = (min(hits), sum(hits), -padding_deviation, -(end - start))
            if best is None or score > best[0]:
                best = (score, start, end)
    if best is not None:
        return (best[1], best[2])
    return candidate_clip_bounds(candidate, rows, padding_sec)

def candidate_video_quality(candidate: dict[str, Any], rows: Sequence[TimestampedPath], poses: Sequence[PoseSample], padding_sec: float, min_clip_seconds: float, max_clip_seconds: float, max_frame_gap: float, min_horizontal_view_ratio: float, max_pitch_p90: float, model_sample_frames: int, min_sampled_visible_frames: int) -> dict[str, Any]:
    start, end = optimized_continuous_clip_bounds(candidate, rows, padding_sec, min_clip_seconds, max_clip_seconds, model_sample_frames, min_sampled_visible_frames)
    clip_rows = rows[start:end + 1]
    reasons: list[str] = []
    if len(clip_rows) < 2:
        reasons.append('too_few_frames')
        duration = 0.0
        gaps: list[float] = []
    else:
        duration = clip_rows[-1].timestamp - clip_rows[0].timestamp
        gaps = [right.timestamp - left.timestamp for left, right in zip(clip_rows, clip_rows[1:])]
        if not min_clip_seconds <= duration <= max_clip_seconds:
            reasons.append('duration_out_of_range')
        if any((gap <= 0 or gap > max_frame_gap for gap in gaps)):
            reasons.append('timestamp_discontinuity')
    clip_poses = [nearest_pose(row.timestamp, poses) for row in clip_rows]
    if any((pose is None for pose in clip_poses)):
        reasons.append('missing_camera_pose')
        pitches: list[float] = []
        rotations: list[int] = []
    else:
        valid_poses = [pose for pose in clip_poses if pose is not None]
        pitches = [abs(camera_pitch_deg(pose)) for pose in valid_poses]
        rotations = [sky_up_rotation_deg(pose) for pose in valid_poses]
    horizontal_ratio = float(np.mean(np.asarray(pitches) <= 45.0)) if pitches else 0.0
    pitch_p90 = float(np.percentile(pitches, 90)) if pitches else 180.0
    rotation_transitions = sum((left != right for left, right in zip(rotations, rotations[1:])))
    if horizontal_ratio < min_horizontal_view_ratio:
        reasons.append('excessive_camera_pitch')
    if pitch_p90 > max_pitch_p90:
        reasons.append('pitch_p90_too_high')
    if rotation_transitions:
        reasons.append('frame_orientation_flip')
    sampled_source_indices: list[int] = []
    sampled_visibility: list[dict[str, Any]] = []
    if end >= start:
        sampled_source_indices = select_evidence_frame_indices(candidate, start, end, model_sample_frames, min_sampled_visible_frames)
        for anchor in [*candidate['route'], candidate['facing']]:
            visible_indices = set(anchor.visible_frame_indices)
            hit_indices = [index for index in sampled_source_indices if index in visible_indices]
            sampled_visibility.append({'object_key': anchor.object_key, 'label': anchor.label, 'sampled_visible_frame_count': len(hit_indices), 'sampled_visible_frame_indices': hit_indices})
            if len(hit_indices) < min_sampled_visible_frames:
                reasons.append(f'model_sampling_misses_anchor:{anchor.object_key}')
    return {'valid': not reasons, 'rejection_reasons': reasons, 'start_frame_index': start, 'end_frame_index': end, 'frame_count': len(clip_rows), 'duration_sec': round(float(duration), 3), 'max_frame_gap_sec': round(float(max(gaps, default=0.0)), 4), 'horizontal_view_ratio': round(horizontal_ratio, 4), 'pitch_p90_deg': round(pitch_p90, 3), 'orientation_transition_count': rotation_transitions, 'model_sample_frames': model_sample_frames, 'clip_selection': 'continuous_interval_boundary_search_v1', 'sampled_source_frame_indices': sampled_source_indices, 'sampled_anchor_visibility': sampled_visibility}

def select_evidence_frame_indices(candidate: dict[str, Any], start: int, end: int, frame_budget: int, min_per_anchor: int) -> list[int]:
    """Return the model's uniform samples from one continuous source clip.

    This function deliberately does *not* reserve or insert landmark frames.
    Reserving temporally distant evidence produced a discontinuous montage
    that looked longer but did not contain a natural traversal.  A candidate
    now survives only when ordinary uniform sampling of the continuous
    ``[start, end]`` interval already observes every named landmark at least
    ``min_per_anchor`` times; that check is performed by
    :func:`candidate_video_quality`.

    ``candidate`` and ``min_per_anchor`` remain in the signature so older
    callers keep working, but they cannot influence which frames are sampled.
    """
    del candidate, min_per_anchor
    if frame_budget <= 0 or end < start:
        return []
    count = min(frame_budget, end - start + 1)
    return [int(value) for value in np.rint(np.linspace(start, end, count))]

def render_route_video(identifier: str, rows: Sequence[TimestampedPath], output: Path, start_frame: int, end_frame: int, fps: float, overwrite: bool, selected_frame_indices: Sequence[int] | None=None) -> None:
    if output.exists() and (not overwrite):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = [rows[index] for index in selected_frame_indices] if selected_frame_indices is not None else rows[start_frame:end_frame + 1]
    if not selected:
        raise RuntimeError(f'No source frames for {identifier}')
    trajectory_path = rows[0].path.parent.parent / 'lowres_wide.traj'
    poses = parse_trajectory(trajectory_path)
    with tempfile.TemporaryDirectory(prefix=f'{identifier}_', dir=output.parent) as temporary:
        temp_dir = Path(temporary)
        rotations: list[int] = []
        for index, row in enumerate(selected):
            pose = nearest_pose(row.timestamp, poses)
            if pose is None:
                raise RuntimeError(f'No camera pose for {row.path}')
            rotations.append(sky_up_rotation_deg(pose))
        if shutil.which('ffmpeg'):
            if len(set(rotations)) != 1:
                raise RuntimeError(f'Orientation changes inside continuous clip {identifier}')
            for index, row in enumerate(selected):
                (temp_dir / f'frame_{index:03d}.png').symlink_to(row.path)
            rotation_filter = {0: '', 90: 'transpose=2,', 180: 'hflip,vflip,', 270: 'transpose=1,'}[rotations[0]]
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps), '-i', str(temp_dir / 'frame_%03d.png'), '-vf', rotation_filter + 'scale=640:480:force_original_aspect_ratio=decrease:flags=lanczos,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
            subprocess.run(command, check=True)
        else:
            import cv2
            for index, (row, rotation) in enumerate(zip(selected, rotations)):
                with Image.open(row.path) as image:
                    image.rotate(rotation, expand=True).save(temp_dir / f'frame_{index:03d}.png')
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (640, 480))
            if not writer.isOpened():
                raise RuntimeError('Could not initialize MP4 writer')
            try:
                for index in range(len(selected)):
                    frame = cv2.imread(str(temp_dir / f'frame_{index:03d}.png'))
                    if frame is None:
                        raise RuntimeError(f'Could not read rendered frame {index}')
                    height, width = frame.shape[:2]
                    scale = min(640.0 / width, 480.0 / height)
                    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_LANCZOS4)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    y0 = (480 - resized.shape[0]) // 2
                    x0 = (640 - resized.shape[1]) // 2
                    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
                    writer.write(canvas)
            finally:
                writer.release()

def record_from_candidate(scene_id: str, candidate: dict[str, Any], video_path: Path, answer_index: int, rng: random.Random, ordinal: int) -> dict[str, Any]:
    route: list[VisibleAnchor] = candidate['route']
    facing: VisibleAnchor = candidate['facing']
    turns: list[str] = candidate['turns']
    choices, answer_key = make_choices(turns, answer_index, rng)
    reasoning = build_reasoning(route, facing, turns)
    identifier = f'arkit_{scene_id}_object_grounded_route_planning_visv3_{ordinal:03d}'
    reasoning_steps = [{'step_id': index, 'claim_type': 'route_planning', 'text': text} for index, text in enumerate(reasoning, start=1)]
    return {'id': identifier, 'instance_id': identifier, 'scene_id': scene_id, 'video_id': scene_id, 'task_type': 'object_grounded_route_planning', 'dataset_source': 'arkitscenes', 'video_path': str(video_path.resolve()), 'question': build_question(route, facing), 'choices': choices, 'answer_key': answer_key, 'answer': choices[answer_key], 'gold_reasoning': reasoning, 'gold_reasoning_steps': reasoning_steps, 'step_reward_labels': [{'step_id': row['step_id'], 'claim_type': row['claim_type'], 'reward': None, 'label': None, 'scorable': False, 'reward_basis': 'reference_pending_review'} for row in reasoning_steps], 'metadata': {'format': 'arkitscenes_revsi_style_route_plan_v1', 'choice_policy': 'symmetric_hamming_rectangle_v2', 'source_split': 'Training', 'route_object_keys': [anchor.object_key for anchor in route], 'route_labels': [anchor.label for anchor in route], 'facing_object_key': facing.object_key, 'facing_label': facing.label, 'turn_sequence': turns, 'route_direction_relative_to_scan': candidate.get('route_direction_relative_to_scan', 'forward'), 'signed_turn_angles_deg': [round(float(value), 2) for value in candidate['angles']], 'anchor_frame_indices': [anchor.frame_index for anchor in route], 'facing_frame_index': facing.frame_index, 'anchor_visibility': [{'label': anchor.label, 'visible_frame_count': anchor.visible_frame_count, 'max_area_ratio': round(anchor.max_area_ratio, 6), 'max_depth_support_ratio': round(anchor.max_depth_support_ratio, 6)} for anchor in [*route, facing]], 'gt_source': 'deterministic_3d_geometry_and_measured_camera_trajectory', 'video_source': 'single_continuous_arkitscenes_rgb_interval', 'video_quality': candidate['video_quality'], 'requires_human_review': True}}

def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)

def validate(rows: Sequence[dict[str, Any]], target: int) -> dict[str, Any]:
    errors: list[str] = []
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append('duplicate_ids')
    for row in rows:
        if not Path(row['video_path']).is_file():
            errors.append(f"missing_video:{row['id']}")
        choices = row['choices']
        if len(choices) != len(set(choices.values())):
            errors.append(f"duplicate_choices:{row['id']}")
        if row['answer_key'] not in choices or row['answer'] != choices.get(row['answer_key']):
            errors.append(f"answer_mapping:{row['id']}")
        quality = row.get('metadata', {}).get('video_quality', {})
        if not quality.get('valid'):
            errors.append(f"invalid_video_quality:{row['id']}")
        if quality.get('continuous_clip') is not True:
            errors.append(f"not_continuous_clip:{row['id']}")
        if quality.get('discrete_evidence_frame_insertion') is not False:
            errors.append(f"evidence_frames_inserted:{row['id']}")
        start = int(quality.get('start_frame_index', -1))
        end = int(quality.get('end_frame_index', -2))
        expected_rendered = end - start + 1
        if expected_rendered <= 24 or quality.get('rendered_frame_count') != expected_rendered:
            errors.append(f"rendered_interval_mismatch:{row['id']}")
        sampled = quality.get('sampled_source_frame_indices', [])
        expected_sampled = [int(value) for value in np.rint(np.linspace(start, end, min(24, expected_rendered)))] if expected_rendered > 0 else []
        if sampled != expected_sampled:
            errors.append(f"nonuniform_model_sampling:{row['id']}")
        for anchor in quality.get('sampled_anchor_visibility', []):
            if int(anchor.get('sampled_visible_frame_count', 0)) < 1:
                errors.append(f"sampled_landmark_missing:{row['id']}:{anchor.get('object_key')}")
    video_paths = [row['video_path'] for row in rows]
    if len(video_paths) != len(set(video_paths)):
        errors.append('shared_video_paths')
    return {'valid': not errors and len(rows) == target, 'count': len(rows), 'target': target, 'scene_distribution': dict(Counter((row['scene_id'] for row in rows))), 'answer_distribution': dict(Counter((row['answer_key'] for row in rows))), 'turn_count_distribution': dict(Counter((len(row['metadata']['turn_sequence']) for row in rows))), 'turn_token_distribution': dict(Counter((turn for row in rows for turn in row['metadata']['turn_sequence']))), 'turn_sequence_distribution': dict(Counter((row['answer'] for row in rows))), 'errors': errors}

def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        preserve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_scenes = load_benchmark_scene_ids((path.expanduser().resolve() for path in args.benchmark))
    rng = random.Random(args.seed)
    scene_candidates: dict[str, tuple[list[dict[str, Any]], list[TimestampedPath]]] = {}
    audit: list[dict[str, Any]] = []
    for scene_dir in discover_scenes(data_root):
        scene_id = scene_dir.name
        if scene_id in excluded_scenes:
            audit.append({'scene_id': scene_id, 'status': 'excluded_benchmark_overlap'})
            continue
        anchors, rgb_rows, camera_path = collect_visible_anchors(scene_dir, args.min_visible_frames, args.min_area_ratio, args.max_area_ratio, args.depth_support_ratio, args.min_depth_coverage, args.min_in_frame_fraction, args.min_corner_depth)
        raw_candidates = candidate_routes(anchors, camera_path, args.include_reverse_routes, args.reverse_routes_only) if anchors else []
        rng.shuffle(raw_candidates)
        poses = parse_trajectory(scene_paths(scene_dir)['trajectory']) if raw_candidates else []
        quality_rejections: Counter[str] = Counter()
        candidates: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            quality = candidate_video_quality(candidate, rgb_rows, poses, args.clip_padding_sec, args.min_clip_seconds, args.max_clip_seconds, args.max_frame_gap, args.min_horizontal_view_ratio, args.max_pitch_p90, args.model_sample_frames, args.min_sampled_visible_frames)
            if quality['valid']:
                candidate['video_quality'] = quality
                candidates.append(candidate)
                if len(candidates) >= args.max_per_scene:
                    break
            else:
                quality_rejections.update(quality['rejection_reasons'])
        rng.shuffle(candidates)
        audit.append({'scene_id': scene_id, 'status': 'eligible' if candidates else 'no_valid_routes', 'unique_visible_anchors': len(anchors), 'candidate_routes': len(candidates), 'candidate_routes_before_video_quality': len(raw_candidates), 'video_quality_rejections': dict(quality_rejections)})
        if candidates:
            scene_candidates[scene_id] = (candidates, rgb_rows)
    selected: list[tuple[str, dict[str, Any], list[TimestampedPath]]] = []
    scene_ids = sorted(scene_candidates)
    rng.shuffle(scene_ids)
    for per_scene_index in range(args.max_per_scene):
        for scene_id in scene_ids:
            candidates, rgb_rows = scene_candidates[scene_id]
            if per_scene_index < len(candidates):
                selected.append((scene_id, candidates[per_scene_index], rgb_rows))
            if len(selected) >= args.target:
                break
        if len(selected) >= args.target:
            break
    if args.turn_sequence_quotas:
        quotas = json.loads(args.turn_sequence_quotas.read_text())
        if sum(quotas.values()) != args.target:
            raise ValueError('Turn-sequence quota counts must sum to --target')
        selected, shortfall = select_with_quotas(scene_candidates, quotas, args.max_per_scene, rng)
        write_jsonl(output_dir / 'turn_sequence_quota_audit.jsonl', [{'requested': quotas, 'shortfall': shortfall, 'satisfied': not shortfall}])
        if shortfall and (not args.allow_shortfall):
            raise RuntimeError('Insufficient genuine candidates for quotas: ' + json.dumps(shortfall))
    if len(selected) < args.target and (not args.allow_shortfall):
        write_jsonl(output_dir / 'scene_audit.jsonl', audit)
        raise RuntimeError(f'Only {len(selected)} valid routes found for target={args.target}; add more scenes without relaxing gates.')
    rows: list[dict[str, Any]] = []
    for index, (scene_id, candidate, rgb_rows) in enumerate(selected):
        identifier = f'arkit_{scene_id}_object_grounded_route_planning_visv3_{index + 1:03d}'
        video_path = output_dir / 'videos' / f'{identifier}.mp4'
        quality = candidate['video_quality']
        quality['continuous_clip'] = True
        quality['discrete_evidence_frame_insertion'] = False
        quality['rendered_frame_count'] = quality['frame_count']
        quality['rendered_fps'] = float(args.video_fps)
        quality['rendered_duration_sec'] = round(quality['rendered_frame_count'] / float(args.video_fps), 3)
        render_route_video(identifier, rgb_rows, video_path, quality['start_frame_index'], quality['end_frame_index'], args.video_fps, args.overwrite, None)
        rows.append(record_from_candidate(scene_id, candidate, video_path, index % 4, rng, index + 1))
    output_jsonl = output_dir / 'qa_candidates.jsonl'
    write_jsonl(output_jsonl, rows)
    write_jsonl(output_dir / 'scene_audit.jsonl', audit)
    summary = validate(rows, len(selected) if args.allow_shortfall else args.target)
    summary['requested_target'] = args.target
    summary['allow_shortfall'] = args.allow_shortfall
    summary['excluded_benchmark_scene_count'] = len(excluded_scenes)
    summary['data_root'] = str(data_root)
    summary['output_jsonl'] = str(output_jsonl)
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['valid'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
