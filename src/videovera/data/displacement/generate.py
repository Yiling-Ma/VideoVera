"""Final data-construction component: data/displacement/generate.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from videovera.io import preserve
import argparse
import fcntl
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from videovera.data.displacement.geometry import DIRECTIONS, TASK_NAME, TASK_TYPE, REASONING_VERSION, classify_offset, direction_facts, gold_steps
from videovera.data.displacement.route_policy import route_policy
from videovera.data.displacement.source import normalized_frames, direct_endpoint_components
REPO = Path(__file__).resolve().parents[2]
from videovera.data.navigation.actions import bearing, wrap, yaw_from_xyzw, file_sha256

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def slots_for(count):
    if count < 36:
        raise ValueError('at_least_36_records_for_joint_balancing')
    slots = []
    seen = Counter()
    for i in range(count):
        direction = DIRECTIONS[(i + 8) % 9]
        j = seen[direction]
        reference = ('start', 'end')[(j + DIRECTIONS.index(direction)) % 2]
        seen[direction] += 1
        slots.append({'index': i, 'direction': direction, 'reference': reference, 'letter': 'ABCD'[i % 4], 'required_mode': None})
    for ref, direction in (('start', 'front'), ('end', 'back')):
        eligible = [s for s in slots if s['reference'] == ref and s['direction'] == direction]
        for slot in eligible[:min(25, max(1, count // 40))]:
            slot['required_mode'] = 'straight'
    return slots

def plan_label(points, start_heading, end_heading, reference):
    delta = np.asarray(points[-1]) - np.asarray(points[0])
    heading = start_heading
    if reference == 'end':
        delta, heading = (-delta, end_heading)
    radians = math.radians(heading)
    front = float(delta @ [-math.sin(radians), 0, -math.cos(radians)])
    right = float(delta @ [math.cos(radians), 0, -math.sin(radians)])
    return classify_offset(front, right)[0]

def balanced_letters(slots, seed=20261002):
    """Avoid a direction/template-dependent option-position shortcut."""
    rng = random.Random(seed)
    result, cursor = ({}, 0)
    for reference in ('start', 'end'):
        for direction in DIRECTIONS:
            indexes = [s['index'] for s in slots if s['reference'] == reference and s['direction'] == direction]
            rng.shuffle(indexes)
            for i, index in enumerate(indexes):
                result[index] = 'ABCD'[(cursor + i) % 4]
            cursor += len(indexes)
    return result

def variants(payload):
    points = payload['navmesh_route']['simplified_agent_points_xyz_m']
    source_frames = payload['camera_trajectory']['frames']
    initial = yaw_from_xyzw(source_frames[0]['rotation_xyzw'])
    final = yaw_from_xyzw(source_frames[-1]['rotation_xyzw'])
    yield ('original', points, initial, final)
    first = bearing(points[0], points[1])
    last = bearing(points[-2], points[-1])
    yield ('straight', points[:2], first, first)
    yield ('return_one_leg', [points[0], points[1], points[0]], first, wrap(first + 180))
    if len(points) >= 3:
        yield ('return_two_legs', points[:3] + [points[1], points[0]], first, wrap(first + 180))
    for offset in (-180, -135, -90, -45, 45, 90, 135):
        yield ('initial_turn_%s' % offset, points, wrap(first + offset), last)
        yield ('final_turn_%s' % offset, points, first, wrap(last + offset))

def path_key(scene, points):
    rounded = np.asarray(points).round(3).tolist()
    return digest([scene, min(rounded, list(reversed(rounded)))])

def build_pool(source):
    pool, rejected = ([], Counter())
    manifest = json.loads((source / 'manifest.json').read_text())
    for item in manifest['videos']:
        annotation = source / item['action_file']
        payload = json.loads(annotation.read_text())
        if payload.get('pose_route_status') != 'navmesh_macro_pose_verified':
            rejected['unverified_source'] += 1
            continue
        ratio = float(payload.get('rendering', {}).get('max_near_black_pixel_ratio', 1))
        if not math.isfinite(ratio) or ratio > 0.1:
            rejected['source_near_black'] += 1
            continue
        scene = str(payload['episode']['scene_id'])
        parent_points = payload['navmesh_route']['simplified_agent_points_xyz_m']
        parent = path_key(scene, parent_points)
        for mode, points, start, end in variants(payload):
            positions = np.asarray(points)
            distances = np.linalg.norm(np.diff(positions[:, [0, 2]], axis=0), axis=1)
            headings = [bearing(a, b) for a, b in zip(points, points[1:])]
            turns = [wrap(b - a) for a, b in zip([start] + headings, headings + [end])]
            turns = [t for t in turns if abs(t) > 1]
            duration = (10 + sum((max(4, math.ceil(float(d) / 0.125)) for d in distances)) + sum((math.ceil(abs(t) / 7.5) for t in turns))) / 8
            if len(turns) > 3 or min(distances) < 0.8 or duration > 40 or (np.ptp(positions[:, 1]) > 0.8) or any((abs(t) < 30 for t in turns)):
                rejected['route_policy'] += 1
                continue
            fingerprint = digest([scene, positions.round(3).tolist(), round(wrap(start), 2), round(wrap(end), 2)])
            for reference in ('start', 'end'):
                try:
                    label = plan_label(points, start, end, reference)
                except ValueError:
                    rejected['ambiguous_direction'] += 1
                    continue
                pool.append({'source_annotation': str(annotation), 'source_video': str(source / item['file']), 'scene': scene, 'parent_key': parent, 'route_fingerprint': fingerprint, 'mode': mode, 'points': points, 'start_heading': start, 'end_heading': end, 'direction': label, 'reference': reference, 'turn_count': len(turns), 'candidate_id': digest([annotation.name, mode, reference])})
    return (pool, dict(rejected))

def select_batch(slots, pool, completed, failed, seed):
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    used_parents = {r['metadata']['parent_route_key'] for r in completed.values()}
    used_routes = {r['metadata']['route_fingerprint'] for r in completed.values()}
    scene_counts = Counter((r['scene_id'] for r in completed.values()))
    mode_counts = Counter((r['metadata']['augmentation'] for r in completed.values()))
    failed_routes = {c['route_fingerprint'] for c in pool if c['candidate_id'] in failed}
    buckets = {}
    for candidate in shuffled:
        if candidate['candidate_id'] not in failed and candidate['route_fingerprint'] not in failed_routes:
            buckets.setdefault((candidate['reference'], candidate['direction']), []).append(candidate)
    pending = [s for s in slots if s['index'] not in completed]
    pending.sort(key=lambda s: (s['required_mode'] is None, len(buckets.get((s['reference'], s['direction']), []))))
    batch = []
    for slot in pending:
        choices = [c for c in buckets.get((slot['reference'], slot['direction']), []) if c['parent_key'] not in used_parents and c['route_fingerprint'] not in used_routes and (slot['required_mode'] is None or c['mode'] == slot['required_mode'])]
        if not choices:
            raise ValueError('insufficient_unique_routes_for_slot:%s' % slot)

        def score(c):
            return (scene_counts[c['scene']] // 4, c['mode'] != 'original', mode_counts[c['mode']] if c['direction'] == 'same_location' else 0, scene_counts[c['scene']])
        candidate = min(choices, key=score)
        batch.append((slot, candidate))
        used_parents.add(candidate['parent_key'])
        used_routes.add(candidate['route_fingerprint'])
        scene_counts[candidate['scene']] += 1
        mode_counts[candidate['mode']] += 1
    return sorted(batch, key=lambda pair: (pair[1]['scene'], pair[0]['index']))

def render_variant(candidate, source_payload, video, sim):
    from videovera.data.navigation.navmesh import emit_pose
    from videovera.data.navigation.renderer import VideoWriter, SENSOR_HEIGHT_M
    points = np.asarray(candidate['points'], dtype=float)
    for a, b in zip(points, points[1:]):
        for fraction in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.05) + 1)):
            if not sim.pathfinder.is_navigable(a + fraction * (b - a)):
                raise ValueError('augmented_path_off_navmesh')
    camera_points = points + [0, SENSOR_HEIGHT_M, 0]
    writer = VideoWriter(video)
    agent = sim.initialize_agent(0)
    position = camera_points[0]
    heading = candidate['start_heading']
    try:
        for _ in range(4):
            emit_pose(sim, agent, writer, position.tolist(), heading)

        def rotate(target):
            nonlocal heading
            delta = wrap(target - heading)
            n = int(math.ceil(abs(delta) / 7.5)) if abs(delta) > 1e-05 else 0
            for i in range(1, n + 1):
                emit_pose(sim, agent, writer, position.tolist(), heading + delta * i / n)
            heading = target
        for a, b in zip(camera_points, camera_points[1:]):
            rotate(bearing(a, b))
            n = max(4, int(math.ceil(np.linalg.norm((b - a)[[0, 2]]) / 0.125)))
            for i in range(1, n + 1):
                emit_pose(sim, agent, writer, (a + (b - a) * i / n).tolist(), heading)
            position = b
        rotate(candidate['end_heading'])
        for _ in range(6):
            emit_pose(sim, agent, writer, position.tolist(), heading)
    finally:
        writer.close()
    if writer.max_near_black_ratio > 0.1:
        raise ValueError('augmented_video_near_black')
    for frame in writer.camera_frames:
        p = np.asarray(frame['position_xyz_m']) - [0, SENSOR_HEIGHT_M, 0]
        if not sim.pathfinder.is_navigable(p):
            raise ValueError('measured_pose_off_navmesh')
    return {'schema_version': 'r2r-temporal-augmentation-v1', 'episode': {k: v for k, v in source_payload['episode'].items() if k in ('scene_id', 'episode_id', 'trajectory_id')}, 'source': {'parent_sidecar': candidate['source_annotation'], 'parent_source': source_payload.get('source', {}), 'augmentation': candidate['mode'], 'original_instruction_not_applicable': True}, 'video': {'file': video.name, 'fps': 8, 'frame_count': len(writer.camera_frames), 'duration_sec': len(writer.camera_frames) / 8}, 'camera_trajectory': {'basis': 'measured_camera_pose', 'capture_source': 'Habitat sensor state for each rendered frame', 'coordinate_system': 'habitat_y_up_camera_minus_z_xyzw', 'video_sha256': file_sha256(video), 'frames': writer.camera_frames}, 'rendering': {'max_near_black_pixel_ratio': writer.max_near_black_ratio, 'speed_scale': 0.5}, 'pose_route_status': 'navmesh_macro_pose_verified', 'navmesh_route': {'simplified_agent_points_xyz_m': candidate['points'], 'all_interpolated_and_measured_points_navigable': True}}

def make_record(slot, candidate, payload, video, annotation, output, seed):
    frames, fps = normalized_frames(payload, video)
    facts = direction_facts(frames, slot['reference'])
    if facts['direction'] != slot['direction']:
        raise ValueError('rendered_answer_differs_from_plan')
    if not np.allclose(direct_endpoint_components(payload, slot['reference']), [facts['front_m'], facts['right_m']], atol=1e-06):
        raise ValueError('native_coordinate_disagreement')
    key = slot['letter']
    rng = random.Random(seed + slot['index'])
    distractors = rng.sample([d for d in DIRECTIONS if d != facts['direction']], 3)
    choices = dict(zip([x for x in 'ABCD' if x != key], distractors))
    choices[key] = facts['direction']
    choices = {k: choices[k] for k in 'ABCD'}
    steps = gold_steps(frames, facts, key)
    policy = route_policy(steps, 0, 3)
    if policy['turn_count'] != candidate['turn_count']:
        raise ValueError('measured_turn_count_disagrees_with_plan')
    if slot['required_mode'] == 'straight' and policy['turn_count'] != 0:
        raise ValueError('straight_route_contains_turn')
    if facts['direction'] == 'same_location':
        positions = np.asarray([f['camera_to_world'] for f in frames]).reshape(-1, 4, 4)[:, :2, 3]
        if np.linalg.norm(positions - positions[0], axis=1).max() < 0.8:
            raise ValueError('same_location_requires_actual_out_and_back')
    identity = video.stem + '_' + slot['reference']
    trajectory = output / 'trajectories' / (identity + '.json')
    dump(trajectory, {'frames': frames, 'coordinate_system': 'z_up_camera_right_down_forward', 'video_sha256': payload['camera_trajectory']['video_sha256']})
    ref, target = (facts['reference'], facts['target'])
    return {'id': identity, 'instance_id': identity, 'task_name': TASK_NAME, 'level_1_type': TASK_NAME, 'level_2_category': 'Cross-Time Relative Position', 'task_type': TASK_TYPE, 'dataset_source': 'r2r', 'scene_id': candidate['scene'], 'video_id': video.stem, 'video_path': str(video), 'question': f"From my viewing position and direction at the {ref} of the video, where is my position at the {target} of the video? Judge direction on the horizontal ground plane. 'Same location' means approximately the same horizontal position, regardless of viewing direction.", 'choices': choices, 'answer_key': key, 'answer': facts['direction'], 'grounding': facts, 'gold_trace': steps, 'gold_reasoning_steps': steps, 'gold_reasoning': [s['text'] for s in steps], 'llm_reasoning_steps': [], 'step_reward_labels': [], 'reasoning_alignment': {'mode': 'claim_level', 'fixed_frames_required': False, 'one_action_per_step': True, 'claim_types': ['reference_frame', 'action', 'relative_direction', 'answer'], 'free_text_policy': 'not_automatically_verified'}, 'metadata': {'physical_scene_id': candidate['scene'], 'split_group': candidate['scene'], 'trajectory_path': str(trajectory), 'annotation_path': str(annotation), 'source_video': candidate['source_video'], 'source_annotation': candidate['source_annotation'], 'source_episode': payload['episode'], 'augmentation': candidate['mode'], 'source_provenance': payload.get('source', {}), 'parent_route_key': candidate['parent_key'], 'route_fingerprint': candidate['route_fingerprint'], 'candidate_id': candidate['candidate_id'], 'slot_index': slot['index'], 'video_sha256': payload['camera_trajectory']['video_sha256'], 'annotation_sha256': file_sha256(annotation), 'source_fps': fps, 'video_origin': 'Habitat rendering of an R2R-derived Matterport3D route; not live-action video', 'gold_source': 'frame_aligned_rendered_camera_poses', 'reasoning_version': REASONING_VERSION, 'route_policy': policy, 'human_review_required': True, 'human_review_status': 'not_individually_reviewed', 'training_ready': False, 'evaluation_valid': False, 'pilot_only': False, 'outcome': 'pending_full_automated_audit'}}

def export(output, completed, count):
    if len(completed) != count:
        raise ValueError('incomplete_dataset')
    rows = [completed[i] for i in range(count)]
    for row, slot in zip(rows, slots_for(count)):
        if (row['answer'], row['grounding']['reference'], row['answer_key']) != (slot['direction'], slot['reference'], slot['letter']):
            raise ValueError('checkpoint_violates_balance_plan')
    if len({r['metadata']['parent_route_key'] for r in rows}) != count:
        raise ValueError('duplicate_parent_route')
    if len({r['metadata']['route_fingerprint'] for r in rows}) != count:
        raise ValueError('duplicate_rendered_route')
    keys = balanced_letters(slots_for(count))
    for row in rows:
        frames = json.loads(Path(row['metadata']['trajectory_path']).read_text())['frames']
        if row['gold_reasoning_steps'] != gold_steps(frames, row['grounding'], row['answer_key']) or row['choices'][row['answer_key']] != row['answer']:
            raise ValueError('invalid_checkpoint_before_option_rebalancing')
        key = keys[row['metadata']['slot_index']]
        distractors = [v for k, v in row['choices'].items() if k != row['answer_key']]
        row['choices'] = dict(zip([k for k in 'ABCD' if k != key], distractors))
        row['choices'][key] = row['answer']
        row['choices'] = {k: row['choices'][k] for k in 'ABCD'}
        row['answer_key'] = key
        steps = gold_steps(frames, row['grounding'], key)
        row.update(gold_trace=steps, gold_reasoning_steps=steps, gold_reasoning=[s['text'] for s in steps])
        row['metadata']['answer_position_policy'] = 'balanced_within_each_reference_direction_cell'
    path = output / 'qa_candidates.jsonl'
    path.write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in rows)))
    from videovera.data.displacement.validation import audit
    report = audit(path)
    dump(output / 'audit_report.json', report)
    if not report['geometry_and_asset_checks_passed']:
        raise ValueError('final_audit_failed')
    if len({r['metadata']['video_sha256'] for r in rows}) != count:
        raise ValueError('duplicate_video_bytes')
    sft = []
    for row in rows:
        row['metadata'].update(training_ready=True, human_review_required=False, outcome='automated_audit_passed_pilot_design_human_accepted', quality_gate='automated_geometry_assets_actions; five-example pilot reviewed, not every training row')
        prompt = row['question'] + '\n\n' + '\n'.join((f"{k}. {v.replace('_', ' ')}" for k, v in row['choices'].items()))
        answer = '\n'.join((f'{i}. {text}' for i, text in enumerate(row['gold_reasoning'], 1)))
        sft.append({'id': row['id'], 'videos': [row['video_path']], 'split_group': row['scene_id'], 'messages': [{'role': 'user', 'content': '<video>\n' + prompt}, {'role': 'assistant', 'content': answer}]})
    path.write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in rows)))
    (output / 'train_sft.jsonl').write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in sft)))
    report.update(jsonl_sha256=file_sha256(path), sft_jsonl_sha256=file_sha256(output / 'train_sft.jsonl'))
    dump(output / 'audit_report.json', report)
    summary = {'records': len(rows), 'unique_videos': len({r['video_path'] for r in rows}), 'reference_counts': dict(Counter((r['grounding']['reference'] for r in rows))), 'answer_counts': dict(Counter((r['answer'] for r in rows))), 'answer_key_counts': dict(Counter((r['answer_key'] for r in rows))), 'joint_reference_direction_counts': dict(Counter((r['grounding']['reference'] + ':' + r['answer'] for r in rows))), 'joint_reference_direction_letter_counts': dict(Counter((r['grounding']['reference'] + ':' + r['answer'] + ':' + r['answer_key'] for r in rows))), 'scene_counts': dict(Counter((r['scene_id'] for r in rows))), 'turn_counts': dict(Counter((r['metadata']['route_policy']['turn_count'] for r in rows))), 'augmentation_counts': dict(Counter((r['metadata']['augmentation'] for r in rows))), 'video_total_bytes': sum((Path(r['video_path']).stat().st_size for r in rows)), 'qa_jsonl': str(path), 'sft_jsonl': str(output / 'train_sft.jsonl'), 'all_rows_automatically_audited': True, 'all_rows_human_reviewed': False, 'split_warning': 'Training only. Split by physical scene, never randomly by row; R2R routes may overlap other task datasets.'}
    dump(output / 'summary.json', summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenes-dir', type=Path, required=True)
    parser.add_argument('--pilot-labels', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3200)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--gpu-device-id', type=int, default=0)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--limit-new', type=int)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--worker-count', type=int, default=4)
    args = parser.parse_args()
    source, output = (args.source_dir.resolve(), args.output_dir.resolve())
    labels = [json.loads(s) for s in args.pilot_labels.read_text().splitlines() if s.strip()]
    if len({r['id'] for r in labels}) < 5 or not all((r.get('decision') == 'accept' and r.get('gold_answer_correct') == 'yes' and (r.get('gold_reasoning_correct') == 'yes') for r in labels)):
        raise ValueError('five_accepted_pilot_reviews_required')
    config = {'source': str(source), 'count': args.count, 'seed': args.seed, 'pilot_reviews_sha256': file_sha256(args.pilot_labels), 'schema_version': 1}
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('w')
    worker = args.worker_index is not None
    fcntl.flock(lock, (fcntl.LOCK_SH if worker else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    if worker:
        if not 0 <= args.worker_index < args.worker_count:
            raise ValueError('invalid_worker_index')
        worker_lock = (output / ('.worker_%d.lock' % args.worker_index)).open('w')
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'config.json').exists() and json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('resume_configuration_mismatch')
    if not worker:
        dump(output / 'config.json', config)
        shutil.copy2(args.pilot_labels, output / 'pilot_human_labels.jsonl')
    for name in ('r2r_clips', 'annotations', 'trajectories', 'completed'):
        (output / name).mkdir(exist_ok=True)
    pool_path = output / 'candidate_pool_eighth_turns.json'
    if pool_path.exists():
        pool = json.loads(pool_path.read_text())
    else:
        pool, rejected = build_pool(source)
        dump(pool_path, pool)
        dump(output / 'pool_rejections.json', rejected)
    slots = slots_for(args.count)
    completed = {int(p.stem): json.loads(p.read_text()) for p in (output / 'completed').glob('*.json')}
    failures_path = output / 'failures.json'
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    if not worker:
        for path in output.glob('failures.worker_*.json'):
            failures.extend(json.loads(path.read_text()))
    else:
        failures_path = output / ('failures.worker_%d.json' % args.worker_index)
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    failed = {r['candidate_id'] for r in failures}
    sim, loaded_scene, new = (None, None, 0)
    try:
        while len(completed) < args.count:
            if worker:
                plan = json.loads((output / 'pending_plan.json').read_text())
                scenes = sorted({item['candidate']['scene'] for item in plan})
                assigned = set(scenes[args.worker_index::args.worker_count])
                batch = [(item['slot'], item['candidate']) for item in plan if item['candidate']['scene'] in assigned and item['slot']['index'] not in completed]
            else:
                batch = select_batch(slots, pool, completed, failed, args.seed)
                dump(output / 'pending_plan.json', [{'slot': s, 'candidate': c} for s, c in batch])
            print(json.dumps({'completed': len(completed), 'remaining': len(batch), 'planned_modes': dict(Counter((c['mode'] for _, c in batch))), 'planned_scenes': len({c['scene'] for _, c in batch})}), flush=True)
            if args.plan_only:
                return
            for slot, candidate in batch:
                stem = 'etd_%04d_%s_%s' % (slot['index'], candidate['scene'], candidate['candidate_id'][:10])
                video = output / 'r2r_clips' / (stem + '.mp4')
                annotation = output / 'annotations' / (stem + '.actions.json')
                try:
                    payload = json.loads(Path(candidate['source_annotation']).read_text())
                    if candidate['mode'] == 'original':
                        shutil.copy2(candidate['source_video'], video)
                    else:
                        import habitat_sim
                        from videovera.data.navigation.renderer import make_simulator, scene_file
                        if loaded_scene != candidate['scene']:
                            if sim is not None:
                                sim.close()
                            sim = None
                            scene_path = scene_file(args.scenes_dir, candidate['scene'])
                            sim = make_simulator(habitat_sim, scene_path, args.gpu_device_id)
                            if not sim.pathfinder.is_loaded:
                                navmesh = scene_path.resolve().with_suffix('.navmesh')
                                if not navmesh.is_file() or not sim.pathfinder.load_nav_mesh(str(navmesh)):
                                    raise RuntimeError('missing_navigation_mesh:%s' % navmesh)
                            loaded_scene = candidate['scene']
                        payload = render_variant(candidate, payload, video, sim)
                    dump(annotation, payload)
                    row = make_record(slot, candidate, payload, video, annotation, output, args.seed)
                    dump(output / 'completed' / (str(slot['index']) + '.json'), row)
                    completed[slot['index']] = row
                    new += 1
                    print(json.dumps({'accepted': len(completed), 'id': row['id'], 'answer': row['answer'], 'reference': slot['reference'], 'mode': candidate['mode']}), flush=True)
                except (ValueError, RuntimeError, OSError) as exc:
                    failures.append({'candidate_id': candidate['candidate_id'], 'reason': str(exc), 'slot': slot['index']})
                    failed.add(candidate['candidate_id'])
                    dump(failures_path, failures)
                    for path in (video, annotation):
                        if path.exists():
                            preserve(path)
                    print(json.dumps({'rejected': candidate['candidate_id'], 'reason': str(exc)}), flush=True)
                if args.limit_new and new >= args.limit_new:
                    return
            if worker:
                return
        if worker:
            return
        print(json.dumps(export(output, completed, args.count), indent=2), flush=True)
    finally:
        if sim is not None:
            sim.close()
if __name__ == '__main__':
    main()
