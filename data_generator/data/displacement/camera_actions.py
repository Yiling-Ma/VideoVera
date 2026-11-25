"""Final data-construction component: data/displacement/camera_actions.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import math
import numpy as np
MOVE_LABELS = ('forward', 'forward-right', 'right', 'backward-right', 'backward', 'backward-left', 'left', 'forward-left')

def arrays(frames):
    times = np.asarray([f['timestamp'] for f in frames], dtype=float)
    poses = np.asarray([f['camera_to_world'] for f in frames], dtype=float).reshape(-1, 4, 4)
    return (times - times[0], poses)

def action_facts(frames, start_sec, end_sec, min_horizontal_forward=0.35):
    """Integrate displacement in the changing camera view, never a fixed initial view."""
    times, poses = arrays(frames)
    if not (math.isfinite(start_sec) and math.isfinite(end_sec) and (0 <= start_sec < end_sec <= times[-1] + 1e-06)):
        raise ValueError('invalid_action_interval')
    a, b = (int(np.argmin(abs(times - start_sec))), int(np.argmin(abs(times - end_sec))))
    if b <= a:
        raise ValueError('action_interval_too_short')
    xy = poses[a:b + 1, :2, 3]
    views = poses[a:b + 1, :2, 2]
    valid = bool(np.all(np.linalg.norm(views, axis=1) >= min_horizontal_forward))
    yaw = np.unwrap(np.arctan2(views[:, 1], views[:, 0]))
    excursion = float(np.linalg.norm(xy - xy[0], axis=1).max())
    kept = [0]
    for i in range(1, len(xy) - 1):
        if times[a + i] - times[a + kept[-1]] >= 0.19:
            kept.append(i)
    if kept[-1] != len(xy) - 1:
        kept.append(len(xy) - 1)
    dp = np.diff(xy[kept], axis=0)
    middle = (yaw[np.array(kept[:-1])] + yaw[np.array(kept[1:])]) / 2
    forward = float((dp[:, 0] * np.cos(middle) + dp[:, 1] * np.sin(middle)).sum())
    right = float((dp[:, 0] * np.sin(middle) - dp[:, 1] * np.cos(middle)).sum())
    length = float(np.linalg.norm(dp, axis=1).sum())
    if excursion < 0.08:
        move = 'stationary'
    elif excursion < 0.2:
        move = 'small_shift'
    elif not valid:
        move = 'unclear'
    elif math.hypot(forward, right) < 0.7 * length:
        move = 'mixed'
    else:
        bearing = math.degrees(math.atan2(right, forward)) % 360
        cardinal = int(math.floor((bearing + 45) / 90)) % 4
        residual = (bearing - cardinal * 90 + 180) % 360 - 180
        if abs(abs(residual) - 30) < 2:
            move = 'unclear'
        else:
            sector = 2 * cardinal if abs(residual) < 30 else (2 * cardinal + (1 if residual > 0 else -1)) % 8
            move = MOVE_LABELS[sector]
    degrees = np.degrees(yaw - yaw[0])
    net = float(degrees[-1])
    turn = 'unclear'
    if valid:
        reversals = np.maximum.accumulate(degrees) - degrees if net >= 0 else degrees - np.minimum.accumulate(degrees)
        if np.ptp(degrees) < 25 or (abs(net) < 25 and float(reversals.max()) < 25):
            turn = 'none'
        elif abs(net) >= 25 and float(reversals.max()) < 25:
            turn = 'left' if net > 0 else 'right'
            if 160 <= abs(net) <= 200:
                turn = 'around_' + turn
        else:
            turn = 'mixed'
    delta = xy[-1] - xy[0]
    if float(delta @ delta) > 1e-12:
        fraction = np.clip((xy - xy[0]) @ delta / (delta @ delta), 0, 1)
        line_error = float(np.linalg.norm(xy - (xy[0] + fraction[:, None] * delta), axis=1).max())
    else:
        line_error = excursion
    first_yaw = yaw[0]
    return {'start_sec': float(times[a]), 'end_sec': float(times[b]), 'move': move, 'turn': turn, 'camera_relative_forward_m': forward if valid else None, 'camera_relative_right_m': right if valid else None, 'horizontal_path_length_m': length, 'max_excursion_m': excursion, 'max_line_deviation_m': line_error, 'heading_range_deg': float(np.ptp(degrees)) if valid else None, 'heading_change_deg_clockwise': -net if valid else -math.degrees(math.atan2(math.sin(yaw[-1] - yaw[0]), math.cos(yaw[-1] - yaw[0]))), 'heading_track_valid': valid, 'offset_in_segment_start_view_m': [float(delta @ [math.cos(first_yaw), math.sin(first_yaw)]), float(delta @ [math.sin(first_yaw), -math.cos(first_yaw)])]}

def action_sequence(frames, min_horizontal_forward=0.35):
    times, poses = arrays(frames)
    xy, views = (poses[:, :2, 3], poses[:, :2, 2])
    if np.any(np.linalg.norm(views, axis=1) < min_horizontal_forward):
        return [action_facts(frames, 0, float(times[-1]), min_horizontal_forward)]
    yaw = np.degrees(np.unwrap(np.arctan2(views[:, 1], views[:, 0])))

    def errors(a, b):
        fraction = (times[a:b + 1] - times[a]) / (times[b] - times[a])
        position = np.linalg.norm(xy[a:b + 1] - (xy[a] + fraction[:, None] * (xy[b] - xy[a])), axis=1) / 0.25
        heading = abs(yaw[a:b + 1] - (yaw[a] + fraction * (yaw[b] - yaw[a]))) / 20
        return np.maximum(position, heading)
    knots = {0, len(times) - 1}
    pending = [(0, len(times) - 1)]
    while pending:
        a, b = pending.pop()
        if b - a < 2:
            continue
        error = errors(a, b)
        i = int(np.argmax(error))
        if error[i] > 1:
            k = a + i
            knots.add(k)
            pending.extend(((a, k), (k, b)))
    knots = sorted(knots)
    spans = list(zip(knots[:-1], knots[1:]))

    def facts(a, b):
        return action_facts(frames, float(times[a]), float(times[b]), min_horizontal_forward)
    merged = []
    for a, b in spans:
        current = facts(a, b)
        if merged:
            c, _, previous = merged[-1]
            little = ('stationary', 'small_shift')
            same_turn = previous['turn'] == current['turn']
            negligible = previous['move'] in little and previous['turn'] == 'none' or (current['move'] in little and current['turn'] == 'none')
            same_move = previous['move'] == current['move'] or previous['move'] in little or current['move'] in little
            if same_move and (same_turn or negligible) and (float(errors(c, b).max()) <= 1.25):
                merged[-1] = (c, b, facts(c, b))
                continue
        merged.append((a, b, current))
    return [item[2] for item in merged]

def action_text(action, *, include_geometry=False, distance_decimals=1, angle_decimals=0):
    if include_geometry:
        base = action_text(action)
        move, turn = (action['move'], action['turn'])
        distance = float(action['horizontal_path_length_m'])
        angle = float(action['heading_change_deg_clockwise'])
        if not math.isfinite(distance) or distance < 0 or (not math.isfinite(angle)):
            raise ValueError('Invalid action geometry')
        if turn == 'none' and move in ('forward', 'backward') and (distance >= 0.05):
            return f'{base} for about {distance:.{distance_decimals}f} m'
        if move == 'stationary' and turn in ('left', 'right', 'around_left', 'around_right'):
            side = 'left' if angle < 0 else 'right'
            return f'turn {side} about {abs(angle):.{angle_decimals}f} degrees in place'
        return base
    move, turn = (action['move'], action['turn'])
    movement = {'forward': 'go roughly straight', 'backward': 'move backward', 'stationary': 'stay in place', 'small_shift': 'shift slightly', 'mixed': 'change movement direction', 'unclear': 'move (direction unclear)'}.get(move, 'move ' + move)
    rotation = {'left': 'turn left', 'right': 'turn right', 'around_left': 'turn around to the left', 'around_right': 'turn around to the right', 'mixed': 'turn both ways', 'unclear': 'turning is unclear'}.get(turn)
    if turn == 'none':
        if move == 'forward' and action.get('max_line_deviation_m', 1) < 0.01 and (action.get('heading_range_deg', 180) < 1):
            return 'go straight'
        return movement
    if move == 'stationary':
        return rotation + ' in place' if turn != 'unclear' else 'stay in place; turning is unclear'
    if move == 'small_shift':
        return rotation + ' with a slight shift'
    if turn == 'unclear':
        return movement + ' (turning unclear)'
    if move in ('mixed', 'unclear'):
        return rotation + ' while moving'
    movement = 'move forward' if move == 'forward' else movement
    return movement + ' while ' + rotation.replace('turn', 'turning', 1)

def replay_route(actions, reference):
    """Replay exact local offsets, retaining lengths/angles behind the short prose."""
    position = np.zeros(2)
    heading = 0.0
    path = [position.tolist()]
    for action in actions:
        forward, right = action['offset_in_segment_start_view_m']
        angle = math.radians(heading)
        position += forward * np.array([math.sin(angle), math.cos(angle)])
        position += right * np.array([math.cos(angle), -math.sin(angle)])
        heading += action['heading_change_deg_clockwise']
        path.append(position.tolist())
    angle = math.radians(heading) if reference == 'end' else 0.0
    target = -position if reference == 'end' else position
    return {'path_right_forward_m': path, 'final_heading_deg_clockwise': heading, 'reference': reference, 'front_m': float(target @ [math.sin(angle), math.cos(angle)]), 'right_m': float(target @ [math.cos(angle), -math.sin(angle)])}

def replay_verbal_geometry(actions, reference, distance_decimals=1, angle_decimals=0):
    """Replay only geometry actually expressed by the simple action sentences."""
    rounded = []
    for action in actions:
        a = dict(action)
        if a['turn'] == 'none' and a['move'] in ('forward', 'backward'):
            distance = float(a['horizontal_path_length_m'])
            if distance < 0.05:
                return None
            sign = 1 if a['move'] == 'forward' else -1
            a['offset_in_segment_start_view_m'] = [sign * round(distance, distance_decimals), 0.0]
            a['heading_change_deg_clockwise'] = 0.0
        elif a['move'] == 'stationary' and a['turn'] in ('none', 'left', 'right', 'around_left', 'around_right'):
            a['offset_in_segment_start_view_m'] = [0.0, 0.0]
            a['heading_change_deg_clockwise'] = 0.0 if a['turn'] == 'none' else round(a['heading_change_deg_clockwise'], angle_decimals)
        else:
            return None
        rounded.append(a)
    return replay_route(rounded, reference)
