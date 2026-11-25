"""Final data-construction component: data/routes/generate.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
from videovera.data.navigation.actions import extract_route, validate_trace, file_sha256, yaw_from_xyzw, wrap
from videovera.data.navigation.questions import build_record, action_text, step, reverse_option_universe
SHAPES = ['straight line', 'L-shape', 'U-shape', 'manhattan-style piecewise linear', 'in-place orientation']
REVERSE = reverse_option_universe()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def shape_for(route):
    angles = [e['heading_change_deg'] for e in route['canonical_actions'] if e['role'] == 'path_bend']
    if not angles:
        return 'straight line'
    if any((not 60 <= abs(a) <= 120 for a in angles)):
        raise ValueError('non_orthogonal_or_ambiguous_shape')
    if len(angles) == 1:
        return 'L-shape'
    if len(angles) == 2 and angles[0] * angles[1] > 0:
        if 150 <= abs(sum(angles)) <= 210:
            return 'U-shape'
        raise ValueError('ambiguous_u_shape')
    return 'manhattan-style piecewise linear'

def crop_payload(source, start, end):
    payload = copy.deepcopy(source)
    frames = payload['camera_trajectory']['frames'][start:end]
    fps = float(payload['video']['fps'])
    for i, frame in enumerate(frames):
        frame['frame_index'], frame['timestamp_sec'] = (i, i / fps)
    payload['camera_trajectory']['frames'] = frames
    payload['video'].update(frame_count=len(frames), duration_sec=len(frames) / fps)
    for key in ('canonical_actions', 'ordered_action_sequence', 'route_plan_landmarks', 'reference_path'):
        payload.pop(key, None)
    return payload

def stationary_ranges(frames):
    begin = 0
    for i in range(1, len(frames) + 1):
        if i < len(frames):
            a, b = (frames[begin]['position_xyz_m'], frames[i]['position_xyz_m'])
            if math.dist(a, b) < 0.005:
                continue
        chunk = frames[begin:i]
        headings = [yaw_from_xyzw(f['rotation_xyzw']) for f in chunk]
        motion = sum((abs(wrap(b - a)) for a, b in zip(headings, headings[1:])))
        if len(chunk) >= 10 and motion >= 60:
            yield (begin, i)
        begin = i

def validate_stationary(payload, video=None):
    validate_trace(payload, video)
    frames = payload['camera_trajectory']['frames']
    p = frames[0]['position_xyz_m']
    if max((math.dist(p, f['position_xyz_m']) for f in frames)) > 0.01:
        raise ValueError('not_in_place')
    heads = [yaw_from_xyzw(f['rotation_xyzw']) for f in frames]
    increments = [wrap(b - a) for a, b in zip(heads, heads[1:])]
    delta = sum(increments)
    if abs(delta) < 60 or max((abs(d) for d in increments)) > 30 or (max(increments) > 0.001 and min(increments) < -0.001):
        raise ValueError('ambiguous_in_place_turn')
    return delta

def candidates(source):
    pool = defaultdict(list)
    rejected = Counter()
    for item in json.loads((source / 'manifest.json').read_text())['videos']:
        annotation, video = (source / item['action_file'], source / item['file'])
        payload = json.loads(annotation.read_text())
        try:
            route = extract_route(payload)
            if payload.get('rendering', {}).get('max_near_black_pixel_ratio', 1) > 0.1:
                raise ValueError('near_black_source')
        except (ValueError, KeyError) as exc:
            rejected[str(exc)] += 1
            continue
        frames = payload['camera_trajectory']['frames']
        n = len(frames)
        parent = route['video_sha256']
        common = dict(source_annotation=str(annotation), source_video=str(video), scene=str(payload['episode']['scene_id']), parent=parent)
        ranges = {(0, n)}
        first, last = (route['legs'][0], route['legs'][-1])
        for fraction in (0.04, 0.08, 0.16, 0.24):
            start = first['start_frame'] + int((first['end_frame'] - first['start_frame']) * fraction)
            end = last['end_frame'] + 1 - int((last['end_frame'] - last['start_frame']) * fraction)
            ranges.add((start, end))
        for start, end in sorted(ranges):
            try:
                child = extract_route(crop_payload(payload, start, end))
            except ValueError:
                continue
            answer = action_text(child['reverse_action_sequence'])
            if answer in REVERSE:
                pool['reverse_route_plan', answer].append(dict(common, start=start, end=end))
            try:
                shape = shape_for(child)
                pool['route_shape', shape].append(dict(common, start=start, end=end))
            except ValueError as exc:
                rejected[str(exc)] += 1
        for leg in route['legs']:
            start, end = (leg['start_frame'], leg['end_frame'] + 1)
            try:
                child = extract_route(crop_payload(payload, start, end))
                if shape_for(child) != 'straight line':
                    continue
            except ValueError:
                continue
            pool['route_shape', 'straight line'].append(dict(common, start=start, end=end))
        for start, end in stationary_ranges(frames):
            try:
                validate_stationary(crop_payload(payload, start, end))
            except ValueError:
                continue
            pool['route_shape', 'in-place orientation'].append(dict(common, start=start, end=end))
    return (pool, rejected)

def select(pool, count, seed):
    rng = random.Random(seed)
    selected = []
    for task, labels in [('route_shape', SHAPES), ('reverse_route_plan', REVERSE)]:
        used, scene_counts, parent_counts = (set(), Counter(), Counter())
        order = sorted(enumerate(labels), key=lambda pair: len({(c['parent'], c['start'], c['end']) for c in pool[task, pair[1]]}))
        for j, label in order:
            bucket = list(pool[task, label])
            rng.shuffle(bucket)
            needed = count // len(labels) + (j < count % len(labels))
            for _ in range(needed):
                choices = [c for c in bucket if (c['parent'], c['start'], c['end']) not in used and parent_counts[c['parent']] < 5]
                if not choices:
                    raise ValueError('insufficient_real_clips:%s:%s:%s' % (task, label, needed))
                c = min(choices, key=lambda x: (parent_counts[x['parent']], scene_counts[x['scene']]))
                used.add((c['parent'], c['start'], c['end']))
                scene_counts[c['scene']] += 1
                parent_counts[c['parent']] += 1
                selected.append(dict(c, task=task, answer=label))
    for task in ('route_shape', 'reverse_route_plan'):
        group = [c for c in selected if c['task'] == task]
        rng.shuffle(group)
        for i, c in enumerate(group):
            c['letter'] = ('ABCDE' if task == 'route_shape' else 'ABCD')[i % (5 if task == 'route_shape' else 4)]
    for i, c in enumerate(selected):
        c['index'] = i
        c['id'] = 'route_repair_v2_%s_%04d_%s' % (c['task'], i, c['parent'][:12])
    return selected

def options_for(c, seed):
    rng = random.Random('%s:%s' % (seed, c['id']))
    universe = SHAPES if c['task'] == 'route_shape' else REVERSE
    distractors = [s for s in universe if s != c['answer']]
    rng.shuffle(distractors)
    keys = 'ABCDE' if c['task'] == 'route_shape' else 'ABCD'
    choices = dict(zip([k for k in keys if k != c['letter']], distractors[:len(keys) - 1]))
    choices[c['letter']] = c['answer']
    return {k: choices[k] for k in keys}

def replace_answer(row, choices, key):
    row.update(choices=choices, answer_key=key, answer=choices[key])
    trace = row['gold_trace'][:-1]
    trace.append(step('Therefore, the correct answer is choice %s: %s' % (key, choices[key]), 'answer', key, {'choices': choices}))
    for i, s in enumerate(trace, 1):
        s['step_id'] = i
    row.update(gold_trace=trace, verifier_reasoning=trace, gold_reasoning=[s['claim'] for s in trace], gold_reasoning_original=[s['claim'] for s in trace])

def render_one(c, output, seed):
    complete = output / 'completed' / (c['id'] + '.json')
    if complete.exists():
        row = json.loads(complete.read_text())
        if file_sha256(Path(row['video_path'])) != row['metadata']['video_sha256']:
            raise ValueError('corrupted_completed_video')
        return row
    parent = json.loads(Path(c['source_annotation']).read_text())
    if parent['camera_trajectory']['video_sha256'] != c['parent']:
        raise ValueError('source_annotation_changed_since_selection')
    payload = crop_payload(parent, c['start'], c['end'])
    video, annotation = (output / 'videos' / (c['id'] + '.mp4'), output / 'annotations' / (c['id'] + '.json'))
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-threads', '1', '-i', c['source_video'], '-an', '-vf', 'trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS' % (c['start'], c['end']), '-vsync', '0', '-c:v', 'libx264', '-threads', '1', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(video)], check=True, capture_output=True)
    payload['camera_trajectory']['video_sha256'] = file_sha256(video)
    payload['video']['file'] = video.name
    payload['source'] = dict(parent.get('source', {}), training_repair_crop=c)
    dump(annotation, payload)
    choices = options_for(c, seed)
    if c['task'] == 'reverse_route_plan':
        row = build_record(payload, video, annotation, c['scene'], video.stem, c['task'], c['id'])
        if row['answer'] != c['answer']:
            raise ValueError('rendered_reverse_gold_changed')
    else:
        if c['answer'] == 'in-place orientation':
            delta = validate_stationary(payload, video)
            trace = [step('Turn %s in place.' % ('left' if delta > 0 else 'right'), 'route_action', delta, {'frame_start': 0, 'frame_end': payload['video']['frame_count'] - 1}), step('My position stays fixed; only my viewing direction changes.', 'route_shape', c['answer'], {'max_displacement_m': 0.01})]
            route = None
        else:
            route = extract_route(payload, video)
            if shape_for(route) != c['answer']:
                raise ValueError('rendered_shape_gold_changed')
            trace = [step(action_text([e['action_type']]), 'route_action', e['action_type'], e) for e in route['canonical_actions']]
            description = {'straight line': 'The movement follows one straight leg.', 'L-shape': 'Two straight legs meet at one approximately right-angle bend.', 'U-shape': 'Two bends in the same direction make a U-shaped path.', 'manhattan-style piecewise linear': 'The path follows several straight legs with a zigzag or multiple bends.'}
            trace.append(step(description[c['answer']], 'route_shape', c['answer'], {'pose_route': route}))
        trace.append(step('', 'answer', '', {}))
        row = dict(id=c['id'], dataset_source='r2r', scene_id=c['scene'], video_id=video.stem, video_path=str(video), annotation_path=str(annotation), task_type=c['task'], question='What is the shape of my actual moving trajectory?', gold_trace=trace, grounding={'pose_route': route}, metadata={})
    replace_answer(row, choices, c['letter'])
    row['metadata'].update(training_repair_version='v2_measured_balanced_routes', parent_video_sha256=c['parent'], parent_video_path=c['source_video'], parent_frame_range=[c['start'], c['end']], video_sha256=payload['camera_trajectory']['video_sha256'], source_scene_id=c['scene'], human_review_required=True, evaluation_valid=False, choice_policy='balanced_semantic_universe_not_gold_centered', split_warning='Crops share parent routes; split only by physical scene.')
    dump(complete, row)
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count-per-task', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.count_per_task < 6:
        raise ValueError('At least six examples per task are required to cover every label')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(source=str(args.source.resolve()), count=args.count_per_task, seed=args.seed, version=2, max_clips_per_parent=5)
    path = args.output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('configuration_changed_use_new_directory')
    dump(path, config)
    plan_path = args.output / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        pool, rejected = candidates(args.source)
        dump(args.output / 'pool_summary.json', {'counts': {str(k): len(v) for k, v in pool.items()}, 'unique_parents': {str(k): len({c['parent'] for c in v}) for k, v in pool.items()}, 'unique_clips': {str(k): len({(c['parent'], c['start'], c['end']) for c in v}) for k, v in pool.items()}, 'rejected': dict(rejected)})
        plan = select(pool, args.count_per_task, args.seed)
        dump(plan_path, plan)
    print(json.dumps({'planned': len(plan), 'task_counts': dict(Counter((c['task'] for c in plan))), 'unique_parent_videos': len({c['parent'] for c in plan})}), flush=True)
    if args.plan_only:
        return
    for path, expected in {c['source_video']: c['parent'] for c in plan}.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError('source_video_hash_mismatch:' + path)
    for name in ('videos', 'annotations', 'completed'):
        (args.output / name).mkdir(exist_ok=True)
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_one, c, args.output, args.seed) for c in plan]
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result())
            if len(rows) % 100 == 0:
                print('verified_clips=%d/%d' % (len(rows), len(plan)), flush=True)
    from videovera.data.routes.finalize import finalize
    summary = finalize(args.source, args.output, args.seed, args.count_per_task)
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
