"""Final data-construction component: data/navigation/actions.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
VERSION = 'measured-pose-route-v2-human-standard'

class UncertainRoute(ValueError):
    pass

@dataclass(frozen=True)
class Policy:
    straight_tolerance_m: float = 0.2
    min_leg_m: float = 0.6
    min_turn_deg: float = 35.0
    back_turn_deg: float = 150.0
    boundary_margin_deg: float = 5.0
    stable_seconds: float = 0.15
    heading_spread_deg: float = 15.0
    min_travel_seconds: float = 0.25
    max_step_m: float = 0.5

def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def video_timestamps(path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)], capture_output=True, text=True, check=True)
        return [float(f['best_effort_timestamp_time']) for f in json.loads(result.stdout)['frames']]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        raise UncertainRoute('cannot_verify_video_frame_timestamps') from exc

def distance(a, b):
    return math.hypot(b[0] - a[0], b[2] - a[2])

def bearing(a, b):
    return math.degrees(math.atan2(-(b[0] - a[0]), -(b[2] - a[2])))

def yaw_from_xyzw(q):
    if len(q) != 4 or not all((math.isfinite(float(v)) for v in q)):
        raise UncertainRoute('invalid_rotation')
    norm = math.sqrt(sum((float(v) ** 2 for v in q)))
    if abs(norm - 1) > 0.01:
        raise UncertainRoute('unnormalized_rotation')
    x, y, z, w = (float(v) / norm for v in q)
    fx, fy, fz = (-2 * (x * z + w * y), -2 * (y * z - w * x), -(1 - 2 * (x * x + y * y)))
    if math.hypot(fx, fz) < 0.5:
        raise UncertainRoute('camera_looking_nearly_vertical')
    return math.degrees(math.atan2(-fx, -fz))

def validate_trace(payload, video_path=None):
    trace = payload.get('camera_trajectory') or {}
    if trace.get('basis') != 'measured_camera_pose':
        raise UncertainRoute('missing_measured_camera_poses')
    if trace.get('coordinate_system') != 'habitat_y_up_camera_minus_z_xyzw':
        raise UncertainRoute('unknown_pose_coordinate_system')
    if not trace.get('capture_source'):
        raise UncertainRoute('missing_pose_capture_provenance')
    frames = trace.get('frames') or []
    fps = float((payload.get('video') or {}).get('fps', 0))
    count = (payload.get('video') or {}).get('frame_count')
    if not math.isfinite(fps) or fps <= 0 or count != len(frames) or (len(frames) < 3):
        raise UncertainRoute('pose_video_frame_count_mismatch')
    digest = trace.get('video_sha256', '')
    if len(digest) != 64 or any((c not in '0123456789abcdef' for c in digest)):
        raise UncertainRoute('missing_video_pose_binding')
    if video_path is not None and file_sha256(video_path) != digest:
        raise UncertainRoute('video_pose_hash_mismatch')
    if video_path is not None:
        timestamps = video_timestamps(video_path)
        if len(timestamps) != count or any((abs(t - i / fps) > 0.001 for i, t in enumerate(timestamps))):
            raise UncertainRoute('actual_video_pose_timestamps_mismatch')
    for i, frame in enumerate(frames):
        point = frame.get('position_xyz_m') or []
        timestamp = float(frame.get('timestamp_sec', -1))
        if frame.get('frame_index') != i or not math.isfinite(timestamp) or abs(timestamp - i / fps) > 0.0001:
            raise UncertainRoute('pose_video_timestamp_mismatch')
        if len(point) != 3 or not all((math.isfinite(float(v)) for v in point)):
            raise UncertainRoute('invalid_position')
        yaw_from_xyzw(frame.get('rotation_xyzw') or [])
    return (frames, fps)

def _deviation(point, a, b):
    dx, dz = (b[0] - a[0], b[2] - a[2])
    length2 = dx * dx + dz * dz
    if length2 == 0:
        return distance(point, a)
    t = max(0, min(1, ((point[0] - a[0]) * dx + (point[2] - a[2]) * dz) / length2))
    return math.hypot(point[0] - a[0] - t * dx, point[2] - a[2] - t * dz)

def _corners(points, tolerance):
    kept = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        peak = max(range(start + 1, end), key=lambda i: _deviation(points[i], points[start], points[end]))
        if _deviation(points[peak], points[start], points[end]) > tolerance:
            kept.add(peak)
            stack.extend([(start, peak), (peak, end)])
    return sorted(kept)

def _merge_minor_bends(points, indices, policy):
    """Merge RDP corners caused by small steering corrections.

    RDP is sensitive to lateral deviation on a long leg.  A candidate corner
    only becomes a route turn when the incoming and outgoing displacement
    directions have a sustained, meaningful angular difference.  Values near
    the turn threshold are retained and rejected later as ambiguous.
    """
    indices = list(indices)
    merge_below = policy.min_turn_deg - policy.boundary_margin_deg
    changed = True
    while changed and len(indices) > 2:
        changed = False
        for offset in range(1, len(indices) - 1):
            left, middle, right = indices[offset - 1:offset + 2]
            delta = abs(wrap(bearing(points[left], points[middle]) - bearing(points[middle], points[right])))
            if delta < merge_below:
                indices.pop(offset)
                changed = True
                break
    return indices

def stable_heading(yaws, indices, fps, policy):
    if len(indices) < 2 or (indices[-1] - indices[0]) / fps < policy.stable_seconds:
        raise UncertainRoute('insufficient_stable_heading_evidence')
    mean = math.degrees(math.atan2(sum((math.sin(math.radians(yaws[i])) for i in indices)), sum((math.cos(math.radians(yaws[i])) for i in indices))))
    if max((abs(wrap(yaws[i] - mean)) for i in indices)) > policy.heading_spread_deg:
        raise UncertainRoute('unstable_camera_heading')
    return mean

def turn_action(delta, policy):
    delta = wrap(delta)
    magnitude = abs(delta)
    if any((abs(magnitude - b) <= policy.boundary_margin_deg for b in (policy.min_turn_deg, policy.back_turn_deg))):
        raise UncertainRoute('turn_near_quantization_boundary')
    if magnitude < policy.min_turn_deg:
        return None
    if magnitude >= policy.back_turn_deg:
        return 'TURN_BACK'
    return 'TURN_LEFT' if delta > 0 else 'TURN_RIGHT'

def extract_route(payload, video_path=None, policy=None):
    policy = policy or Policy()
    frames, fps = validate_trace(payload, video_path)
    points = [f['position_xyz_m'] for f in frames]
    yaws = [yaw_from_xyzw(f['rotation_xyzw']) for f in frames]
    if distance(points[0], points[-1]) < policy.min_leg_m:
        raise UncertainRoute('already_at_start_or_closed_route')
    if any((distance(a, b) > policy.max_step_m for a, b in zip(points, points[1:]))):
        raise UncertainRoute('position_jump_or_sparse_poses')
    indices = _merge_minor_bends(points, _corners(points, policy.straight_tolerance_m), policy)
    legs = []
    for a, b in zip(indices, indices[1:]):
        length = distance(points[a], points[b])
        if length < policy.min_leg_m:
            raise UncertainRoute('short_or_ambiguous_travel_leg')
        support = [i for i in range(a, b + 1) if distance(points[i], points[a]) >= min(0.2, length * 0.2) and distance(points[i], points[b]) >= min(0.2, length * 0.2)]
        if len(support) < 2 or (support[-1] - support[0]) / fps < policy.min_travel_seconds:
            raise UncertainRoute('insufficient_sustained_translation')
        heading = bearing(points[a], points[b])
        dx, dz = (points[b][0] - points[a][0], points[b][2] - points[a][2])
        progress = [((points[i][0] - points[a][0]) * dx + (points[i][2] - points[a][2]) * dz) / length for i in range(a, b + 1)]
        high = progress[0]
        for value in progress:
            if high - value > policy.straight_tolerance_m:
                raise UncertainRoute('unmodeled_backtracking')
            high = max(high, value)
        if any((abs(wrap(yaws[i] - heading)) > policy.min_turn_deg for i in support)):
            raise UncertainRoute('translation_camera_direction_disagreement')
        support_count = max(2, math.ceil(policy.stable_seconds * fps) + 1)
        for subset in (support[:support_count], support[-support_count:]):
            measured = stable_heading(yaws, subset, fps, policy)
            if abs(wrap(measured - heading)) > policy.heading_spread_deg:
                raise UncertainRoute('stable_heading_path_direction_disagreement')
        legs.append(dict(start_frame=a, end_frame=b, distance_m=length, heading_deg=heading, support_frames=[support[0], support[-1]]))
    n = max(2, math.ceil(policy.stable_seconds * fps) + 1)
    if n > len(yaws):
        raise UncertainRoute('insufficient_endpoint_pose_samples')
    start_heading = stable_heading(yaws, list(range(n)), fps, policy)
    end_heading = stable_heading(yaws, list(range(len(yaws) - n, len(yaws))), fps, policy)
    if abs(wrap(yaws[0] - start_heading)) > policy.heading_spread_deg or abs(wrap(yaws[-1] - end_heading)) > policy.heading_spread_deg:
        raise UncertainRoute('unstable_endpoint_orientation')
    start_heading, end_heading = (yaws[0], yaws[-1])
    events = []

    def add_turn(before, after, frame, role, end_frame=None):
        action = turn_action(after - before, policy)
        if action:
            events.append(dict(action_type=action, heading_before_deg=before, heading_after_deg=after, heading_change_deg=wrap(after - before), video_frame_start=frame, video_frame_end=(frame if end_frame is None else end_frame) + 1, role=role))
        return action
    current = start_heading
    for j, leg in enumerate(legs):
        turn_start = 0 if j == 0 else legs[j - 1]['support_frames'][-1]
        action = add_turn(current, leg['heading_deg'], turn_start, 'initial_orientation' if j == 0 else 'path_bend', leg['support_frames'][0])
        if j and (not action):
            raise UncertainRoute('curved_or_ambiguous_path')
        events.append(dict(action_type='GO_STRAIGHT', heading_before_deg=leg['heading_deg'], heading_after_deg=leg['heading_deg'], heading_change_deg=0, video_frame_start=leg['start_frame'], video_frame_end=leg['end_frame'] + 1, distance_m=leg['distance_m'], from_waypoint_index=j, to_waypoint_index=j + 1, support_frames=leg['support_frames'], role='travel'))
        current = leg['heading_deg']
    terminal_start = max((i + 1 for i, (a, b) in enumerate(zip(points, points[1:])) if distance(a, b) > 0.02), default=len(points) - 1)
    if add_turn(current, end_heading, terminal_start, 'terminal_orientation'):
        events[-1]['video_frame_end'] = len(points)
    for i, event in enumerate(events, 1):
        event.update(action_index=i, video_start_sec=event['video_frame_start'] / fps, video_end_sec=event['video_frame_end'] / fps, pose_verified=True)
    reverse = []
    current = end_heading
    for leg in reversed(legs):
        target = wrap(leg['heading_deg'] + 180)
        action = turn_action(target - current, policy)
        if action:
            reverse.append(dict(action_type=action, heading_change_deg=wrap(target - current), heading_before_deg=current, heading_after_deg=target))
        reverse.append(dict(action_type='GO_STRAIGHT', distance_m=leg['distance_m'], retraces_frames=[leg['start_frame'], leg['end_frame']]))
        current = target
    signature = hashlib.sha256(json.dumps({'frames': frames, 'policy': asdict(policy)}, sort_keys=True).encode()).hexdigest()
    return dict(version=VERSION, policy=asdict(policy), pose_signature=signature, video_sha256=payload['camera_trajectory']['video_sha256'], start_heading_deg=start_heading, end_heading_deg=end_heading, legs=legs, canonical_actions=events, reverse_actions=reverse, forward_actions=[e['action_type'] for e in events], reverse_action_sequence=[e['action_type'] for e in reverse], boundaries=[dict(waypoint_index=j, frame_index=i, position_xyz_m=points[i]) for j, i in enumerate(indices)])
