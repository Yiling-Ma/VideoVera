"""Final data-construction component: data/displacement/geometry.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import numpy as np
from data_generator.data.displacement.camera_actions import MOVE_LABELS, action_facts, action_sequence, action_text, replay_route, replay_verbal_geometry
TASK_TYPE = 'egocentric_temporal_direction'
TASK_NAME = 'Egocentric Temporal Direction'
DIRECTIONS = ('front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left', 'same_location')

@dataclass(frozen=True)
class Policy:
    same_location_m: float = 0.2
    location_margin_m: float = 0.1
    boundary_margin_deg: float = 5.0
    min_horizontal_forward: float = 0.35
    max_speed_mps: float = 3.0
    max_angular_speed_dps: float = 240.0
    max_gap_sec: float = 0.25
    max_vertical_range_m: float = 0.8

def pose_arrays(frames):
    poses = np.asarray([f['camera_to_world'] for f in frames], dtype=float).reshape(-1, 4, 4)
    times = np.asarray([f['timestamp'] for f in frames], dtype=float)
    return (times, poses)

def horizontal_basis(pose, policy=Policy()):
    forward = np.array(pose[:3, 2], dtype=float)
    forward[2] = 0
    norm = np.linalg.norm(forward)
    if norm < policy.min_horizontal_forward:
        raise ValueError('camera_looking_nearly_vertical')
    forward /= norm
    right = np.cross(forward, [0.0, 0.0, 1.0])
    return (forward, right)

def classify_offset(front_m, right_m, policy=Policy()):
    distance = math.hypot(front_m, right_m)
    if distance < policy.same_location_m - policy.location_margin_m:
        return ('same_location', None)
    if distance <= policy.same_location_m + policy.location_margin_m:
        raise ValueError('ambiguous_same_location_boundary')
    bearing = math.degrees(math.atan2(right_m, front_m)) % 360
    sector = int(math.floor((bearing + 22.5) / 45)) % 8
    residual = abs((bearing - sector * 45 + 180) % 360 - 180)
    if 22.5 - residual < policy.boundary_margin_deg:
        raise ValueError('ambiguous_direction_boundary')
    return (DIRECTIONS[sector], bearing)

def validate_trajectory(frames, policy=Policy()):
    if len(frames) < 2 or any((f.get('camera_to_world') is None for f in frames)):
        raise ValueError('missing_pose')
    times, poses = pose_arrays(frames)
    if not np.isfinite(poses).all() or not np.isfinite(times).all():
        raise ValueError('nonfinite_pose_or_time')
    dt = np.diff(times)
    if np.any(dt <= 0) or np.max(dt) > policy.max_gap_sec:
        raise ValueError('invalid_timestamp_gap')
    rotations = poses[:, :3, :3]
    if not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=0.0001) or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=0.015) or (not np.allclose(np.linalg.det(rotations), 1, atol=0.015)):
        raise ValueError('invalid_rigid_pose')
    positions = poses[:, :3, 3]
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    relative = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    if np.max(speed) > policy.max_speed_mps:
        raise ValueError('translation_jump')
    if np.max(angles / dt) > policy.max_angular_speed_dps:
        raise ValueError('rotation_jump')
    if np.ptp(positions[:, 2]) > policy.max_vertical_range_m:
        raise ValueError('excessive_vertical_motion')
    horizontal_basis(poses[0], policy)
    horizontal_basis(poses[-1], policy)
    return (times, poses)

def direction_facts(frames, reference, policy=Policy()):
    if reference not in ('start', 'end'):
        raise ValueError('reference must be start or end')
    times, poses = validate_trajectory(frames, policy)
    ref, target = (0, -1) if reference == 'start' else (-1, 0)
    forward, right = horizontal_basis(poses[ref], policy)
    displacement = poses[target, :3, 3] - poses[ref, :3, 3]
    front_m, right_m = (float(displacement @ forward), float(displacement @ right))
    direction, bearing = classify_offset(front_m, right_m, policy)
    return {'reference': reference, 'target': 'end' if reference == 'start' else 'start', 'reference_position_world': poses[ref, :3, 3].tolist(), 'target_position_world': poses[target, :3, 3].tolist(), 'forward_world': forward.tolist(), 'right_world': right.tolist(), 'front_m': front_m, 'right_m': right_m, 'horizontal_distance_m': math.hypot(front_m, right_m), 'bearing_deg_clockwise_from_forward': bearing, 'direction': direction, 'duration_sec': float(times[-1] - times[0]), 'policy': asdict(policy)}

def motion_facts(frames, start_sec, end_sec, policy=Policy()):
    """Evaluate an arbitrary clip-relative interval, not a prescribed frame list."""
    times, poses = pose_arrays(frames)
    times = times - times[0]
    if not (math.isfinite(start_sec) and math.isfinite(end_sec) and (0 <= start_sec < end_sec <= times[-1] + 1e-06)):
        raise ValueError('invalid_motion_interval')
    a, b = (int(np.argmin(abs(times - start_sec))), int(np.argmin(abs(times - end_sec))))
    if b <= a or end_sec - start_sec < 0.5:
        raise ValueError('motion_interval_too_short')
    indexes = [a]
    for i in range(a + 1, b):
        if times[i] - times[indexes[-1]] >= 0.19:
            indexes.append(i)
    if indexes[-1] != b:
        indexes.append(b)
    sampled = poses[indexes]
    position = sampled[:, :2, 3]
    path_length = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    excursion = float(np.linalg.norm(position - position[0], axis=1).max())
    translation = 'stationary' if excursion < 0.08 else 'moving' if excursion > 0.2 else 'uncertain'
    forwards = [horizontal_basis(p, policy)[0] for p in sampled]
    headings = np.unwrap([math.atan2(f[1], f[0]) for f in forwards])
    degrees = np.degrees(headings - headings[0])
    net = float(degrees[-1])
    rotation = 'uncertain'
    if np.ptp(degrees) < 8:
        rotation = 'stable'
    elif degrees.max() > 15 and degrees.min() < -15:
        rotation = 'mixed'
    elif net > 20 and degrees.max() - net < 10 and (degrees.min() > -10):
        rotation = 'left'
    elif net < -20 and net - degrees.min() < 10 and (degrees.max() < 10):
        rotation = 'right'
    return {'start_sec': float(times[a]), 'end_sec': float(times[b]), 'translation': translation, 'rotation': rotation, 'horizontal_path_length_m': path_length, 'max_excursion_m': excursion, 'net_yaw_deg_left_positive': net}
REASONING_VERSION = 'single_action_geometry_v8_adaptive_precision'

def simplify_path(points, tolerance):
    """Keep spatial turning points, rather than cutting time into equal bins."""
    if len(points) <= 2:
        return [0, len(points) - 1]
    delta = points[-1] - points[0]
    if np.linalg.norm(delta) < 1e-08:
        distances = np.linalg.norm(points - points[0], axis=1)
    else:
        fraction = np.clip((points - points[0]) @ delta / (delta @ delta), 0, 1)
        distances = np.linalg.norm(points - (points[0] + fraction[:, None] * delta), axis=1)
    split = int(np.argmax(distances))
    if distances[split] <= tolerance:
        return [0, len(points) - 1]
    return simplify_path(points[:split + 1], tolerance)[:-1] + [split + i for i in simplify_path(points[split:], tolerance)]

def trajectory_summary(frames, policy=Policy()):
    times, poses = pose_arrays(frames)
    forward, right = horizontal_basis(poses[0], policy)
    relative = poses[:, :3, 3] - poses[0, :3, 3]
    xy = np.column_stack((relative @ forward, relative @ right))
    kept = simplify_path(xy, 0.35)
    legs = []
    for a, b in zip(kept[:-1], kept[1:]):
        delta = xy[b] - xy[a]
        if np.linalg.norm(delta) < 0.2:
            continue
        bearing = math.degrees(math.atan2(delta[1], delta[0])) % 360
        direction = DIRECTIONS[int(math.floor((bearing + 22.5) / 45)) % 8]
        if not legs or direction != legs[-1]:
            legs.append(direction)
    excursion = float(np.linalg.norm(xy, axis=1).max())
    separation = float(np.linalg.norm(xy[-1]))
    return {'directions_from_start_view': legs, 'path_kind': 'stationary' if excursion < 0.08 else 'small_motion' if excursion <= 0.2 else 'returning' if separation < policy.same_location_m - policy.location_margin_m else 'single_direction' if len(legs) == 1 else 'multi_direction', 'max_excursion_m': excursion, 'endpoint_distance_m': separation, 'spatial_tolerance_m': 0.35, 'event_times_sec': [float(times[i] - times[0]) for i in kept]}

def heading_summary(frames, policy=Policy()):
    _, poses = pose_arrays(frames)
    first, _ = horizontal_basis(poses[0], policy)
    last, _ = horizontal_basis(poses[-1], policy)
    angle = math.degrees(math.atan2(float(np.cross(first, last)[2]), float(first @ last)))
    final_facing = 'roughly_back' if abs(angle) >= 140 else 'roughly_initial' if abs(angle) < 20 else 'changed'
    try:
        headings = np.unwrap([math.atan2(f[1], f[0]) for f in (horizontal_basis(p, policy)[0] for p in poses)])
    except ValueError:
        return {'turn_pattern': 'unresolved', 'final_facing': final_facing}
    degrees = np.degrees(headings - headings[0])
    extrema = [float(degrees[0])]
    direction = 0
    extreme = float(degrees[0])
    for value in degrees[1:]:
        if direction == 0:
            if abs(value - extreme) >= 20:
                direction = 1 if value > extreme else -1
                extreme = float(value)
        elif direction * (value - extreme) >= 0:
            extreme = float(value)
        elif direction * (value - extreme) < -20:
            extrema.append(extreme)
            direction *= -1
            extreme = float(value)
    extrema.append(extreme)
    turns = ['left' if b > a else 'right' for a, b in zip(extrema[:-1], extrema[1:]) if abs(b - a) >= 20]
    pattern = 'stable' if not turns else turns[0] if len(set(turns)) == 1 else 'both_directions'
    return {'turn_pattern': pattern, 'final_facing': final_facing, 'net_yaw_deg_left_positive': float(degrees[-1])}

def motion_summary(frames, policy=Policy()):
    route = trajectory_summary(frames, policy)
    kind = route['path_kind']
    return {'translation': kind if kind in ('stationary', 'small_motion', 'returning') else 'moving', **heading_summary(frames, policy)}

def verified_evidence(value, field):
    return isinstance(value, dict) and value.get('verified') is True and isinstance(value.get(field), str) and bool(value[field].strip()) and isinstance(value.get('source'), str) and bool(value['source'].strip())

def gold_steps(frames, facts, answer_key, reference_landmark=None, visual_evidence=None):
    steps = []

    def add(kind, text, claim):
        steps.append({'step_id': f'gold_{len(steps) + 1:02d}', 'claim_type': kind, 'text': text, 'structured_claim': {'claim_type': kind, **claim}, 'evidence_source': 'camera_pose_metadata'})
    ref, target = (facts['reference'], facts['target'])
    reference_text = f'Take my viewing direction at the {ref} as forward.'
    visual_evidence = visual_evidence or {}
    if verified_evidence(reference_landmark, 'name'):
        reference_text = f"At the {ref}, I face {reference_landmark['name']}; take that direction as forward."
    add('reference_frame', reference_text, {'reference': ref, 'target': target})
    if reference_text.startswith('At the'):
        steps[-1]['evidence_source'] = 'camera_pose_metadata_and_verified_landmark'
        steps[-1]['landmark_evidence'] = dict(reference_landmark)
    policy = Policy(**facts['policy'])
    actions = action_sequence(frames, policy.min_horizontal_forward)
    replay = replay_route(actions, ref)
    if not np.allclose([replay['front_m'], replay['right_m']], [facts['front_m'], facts['right_m']], atol=1e-06):
        raise ValueError('action_route_does_not_reconstruct_endpoint')
    route_evidence = visual_evidence.get('route')
    narrated_actions = list(actions)

    def endpoint_hold(action):
        return action['move'] == 'stationary' and action['turn'] == 'none' and (action['max_excursion_m'] < 1e-06) and (abs(action['heading_change_deg_clockwise']) < 1e-06)
    while len(narrated_actions) > 1 and endpoint_hold(narrated_actions[0]):
        narrated_actions.pop(0)
    while len(narrated_actions) > 1 and endpoint_hold(narrated_actions[-1]):
        narrated_actions.pop()
    verbal_replay = None
    distance_decimals, angle_decimals = (1, 0)
    for distance_decimals, angle_decimals in ((1, 0), (2, 1), (3, 2), (4, 3)):
        verbal_replay = replay_verbal_geometry(narrated_actions, ref, distance_decimals, angle_decimals)
        if verbal_replay is None:
            break
        try:
            label = classify_offset(verbal_replay['front_m'], verbal_replay['right_m'], policy)[0]
        except ValueError:
            continue
        if label == facts['direction']:
            break
    else:
        raise ValueError('verbal_geometry_does_not_support_gold_direction')
    for index, action in enumerate(narrated_actions):
        text = action_text(action, include_geometry=True, distance_decimals=distance_decimals, angle_decimals=angle_decimals)
        add('action', text[0].upper() + text[1:] + '.', action)
        steps[-1]['action_reference'] = 'instantaneous_camera_view_not_body_heading'
        if index == 0 and verified_evidence(route_evidence, 'text'):
            steps[-1]['visual_evidence'] = dict(route_evidence)
    direction = facts['direction']
    locations = {'front': 'ahead', 'back': 'behind', 'left': 'to the left', 'right': 'to the right', 'front-left': 'ahead and to the left', 'front-right': 'ahead and to the right', 'back-left': 'behind and to the left', 'back-right': 'behind and to the right'}
    if direction == 'same_location':
        text = 'The start and end are approximately the same location, regardless of viewing direction.'
    else:
        reference = 'original' if ref == 'start' else 'final'
        target_text = f'the {target} position'
        target_evidence = visual_evidence.get(target + '_location')
        if verified_evidence(target_evidence, 'description'):
            target_text += ' ' + target_evidence['description'].strip()
        text = f'Following this route, using my {reference} viewing direction, {target_text} is {locations[direction]}.'
    add('relative_direction', text, {'reference': ref, 'direction': direction, 'same_location': direction == 'same_location'})
    steps[-1]['route_reconstruction'] = replay
    steps[-1]['verbal_geometry_reconstruction'] = verbal_replay
    steps[-1]['verbal_geometry_precision'] = {'distance_decimals': distance_decimals, 'angle_decimals': angle_decimals, 'verified': verbal_replay is not None}
    if direction != 'same_location' and verified_evidence(target_evidence, 'description'):
        steps[-1]['evidence_source'] = 'camera_pose_metadata_and_reviewed_video'
        steps[-1]['visual_evidence'] = dict(target_evidence)
    add('answer', f"Therefore, the answer is {answer_key}: {direction.replace('_', ' ')}.", {'answer_key': answer_key, 'direction': direction})
    return steps

def compare_claims(record, prediction, frames):
    """Check structured assertions only; free-form text is never silently verified."""
    facts = record['grounding']
    policy = Policy(**facts['policy'])
    results = []
    steps = []
    for step in prediction.get('reasoning_steps', []):
        claim = step.get('structured_claim') or {}
        if claim.get('claim_type') == 'action_sequence':
            actions = claim.get('actions')
            if not isinstance(actions, list) or not actions:
                results.append({'step_id': step.get('step_id'), 'claim_type': 'action_sequence', 'status': 'unknown', 'reason': 'missing_action_list'})
                continue
            for j, action in enumerate(actions):
                steps.append({'step_id': f"{step.get('step_id', 'step')}:action_{j + 1}", 'structured_claim': {**action, 'claim_type': 'action'} if isinstance(action, dict) else {'claim_type': 'action'}})
        else:
            steps.append(step)
    for i, step in enumerate(steps):
        claim = step.get('structured_claim') or {}
        kind = claim.get('claim_type')
        expected = None
        keys = ()
        try:
            allowed = {'reference': ('start', 'end'), 'target': ('start', 'end'), 'direction': DIRECTIONS, 'answer_key': tuple('ABCD'), 'translation': ('stationary', 'small_motion', 'returning', 'moving') if kind == 'motion_summary' else ('stationary', 'moving', 'uncertain'), 'rotation': ('left', 'right', 'stable', 'mixed', 'uncertain')}
            if kind == 'action':
                allowed.update(move=MOVE_LABELS + ('stationary', 'small_shift', 'mixed', 'unclear'), turn=('none', 'left', 'right', 'around_left', 'around_right', 'mixed', 'unclear'))
            if any((key in claim and claim[key] not in values for key, values in allowed.items())):
                raise ValueError('malformed_claim_value')
            if kind == 'reference_frame':
                expected, keys = (facts, ('reference', 'target'))
            elif kind == 'same_location':
                expected, keys = ({'same_location': facts['direction'] == 'same_location'}, ('same_location',))
                if type(claim.get('same_location')) is not bool:
                    raise ValueError('same_location_must_be_boolean')
            elif kind == 'relative_direction':
                expected, keys = (facts, ('reference', 'direction'))
                expected = {**facts, 'same_location': facts['direction'] == 'same_location', 'answer_key': record['answer_key']}
                keys += tuple((k for k in ('same_location', 'answer_key') if k in claim))
            elif kind == 'trajectory_summary':
                expected = trajectory_summary(frames, policy)
                keys = ('path_kind', 'directions_from_start_view')
                if any((claim.get(k) != expected[k] for k in keys)):
                    raise ValueError('different_path_summaries_require_review_not_exact_step_matching')
            elif kind == 'motion_summary':
                expected = motion_summary(frames, policy)
                keys = ('translation', 'turn_pattern', 'final_facing')
            elif kind == 'heading_change':
                expected = heading_summary(frames, policy)
                keys = ('turn_pattern', 'final_facing')
            elif kind == 'action':
                expected = action_facts(frames, float(claim['start_sec']), float(claim['end_sec']), policy.min_horizontal_forward)
                keys = ('move', 'turn')
            elif kind == 'motion':
                expected = motion_facts(frames, float(claim['start_sec']), float(claim['end_sec']), policy)
                keys = tuple((k for k in ('translation', 'rotation') if k in claim))
            elif kind == 'answer':
                expected, keys = ({'answer_key': record['answer_key'], 'direction': record['answer']}, ('answer_key', 'direction'))
            checked = [k for k in keys if expected[k] != 'uncertain' and claim.get(k) not in (None, 'uncertain') and ('unclear' not in (expected[k], claim.get(k)) or expected[k] == claim.get(k))]
            wrong = [k for k in checked if claim[k] != expected[k]]
            status = 'contradicted' if wrong else 'supported' if checked and len(checked) == len(keys) else 'unknown'
            results.append({'step_id': step.get('step_id', f'model_{i + 1}'), 'claim_type': kind, 'status': status, 'checked_fields': checked, 'contradicted_fields': wrong, 'expected': expected, 'free_text_status': 'not_automatically_verified'})
        except (KeyError, TypeError, ValueError) as exc:
            results.append({'step_id': step.get('step_id', f'model_{i + 1}'), 'claim_type': kind, 'status': 'unknown', 'reason': str(exc)})
    answer_correct = prediction.get('answer_key') == record['answer_key']
    reasoning_error = any((r['status'] == 'contradicted' and r['claim_type'] != 'answer' for r in results))
    outcome = 'answer_wrong' if not answer_correct else 'answer_correct_reasoning_error' if reasoning_error else 'answer_correct_reasoning_unverified'
    return {'answer_correct': answer_correct, 'reasoning_error_detected': reasoning_error, 'outcome': outcome, 'claim_results': results, 'missing_claim_types': sorted({'reference_frame', 'relative_direction'} - {r['claim_type'] for r in results}), 'alignment': 'semantic_claims_and_optional_time_intervals_not_fixed_frames'}
