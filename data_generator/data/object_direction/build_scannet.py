"""Final data-construction component: data/object_direction/build_scannet.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(''.join((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)))

def physical_scene(row):
    scene = str(row.get('scene_id') or '')
    return scene.rsplit('_', 1)[0] if row.get('dataset_source') == 'scannet' else scene

def scene_order(refs, excluded, seed, min_unique_categories=8):
    by_scene = defaultdict(lambda: defaultdict(set))
    for ref in refs:
        scene = ref['scene_id']
        if scene not in excluded:
            by_scene[scene][ref['object_name'].strip().lower()].add(str(ref['object_id']))
    candidates = []
    for scene, categories in by_scene.items():
        unique = sum((len(ids) == 1 for ids in categories.values()))
        if unique >= min_unique_categories:
            candidates.append((scene, unique))
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [scene for scene, _ in candidates]

def pool(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'pool_state.json'
    state = json.loads(manifest.read_text()) if manifest.exists() else {'scenes': {}, 'seed': args.seed}
    if state['seed'] != args.seed:
        raise ValueError('Resume seed differs from existing pool state')
    benchmark = read_jsonl(args.benchmark)
    excluded = {r['scene_id'] for r in benchmark if r.get('dataset_source') == 'scannet'}
    excluded |= {r['scene_id'] for r in read_jsonl(args.dev) if r.get('dataset_source') == 'scannet'} if args.dev else set()
    refs = json.loads((args.root / 'scanrefer' / 'ScanRefer_filtered_train.json').read_text())
    order = scene_order(refs, excluded, args.seed, args.min_unique_categories)
    total = sum((s['count'] for s in state['scenes'].values() if s['status'] == 'ok'))
    for scene in order:
        if total >= args.target:
            break
        if len(state['scenes']) >= args.max_scenes and scene not in state['scenes']:
            continue
        if scene in state['scenes'] and (state['scenes'][scene]['status'] == 'ok' or not args.retry_failed):
            continue
        scene_out = out / 'scenes' / scene
        command = [sys.executable, '-m', 'data_generator.data.object_direction.scannet', '--root', str(args.root), '--scene', scene, '--count', str(args.per_scene), '--seed', str(args.seed), '--fallback-source-fps', str(args.source_fps), '--output-dir', str(scene_out)]
        if args.allow_partial_count:
            command.append('--allow-partial-count')
        if args.preferred_answers:
            command += ['--preferred-answers', *args.preferred_answers]
        print(f"[{len(state['scenes']) + 1}] {scene}: preparing {args.per_scene} cases", flush=True)
        with (out / 'generation.log').open('a') as log:
            log.write(f'\nSCENE {scene}\n')
            log.flush()
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode and 'Non-monotonic color timestamps' in (out / 'generation.log').read_text().split(f'SCENE {scene}')[-1]:
            command.append('--force-frame-time')
            with (out / 'generation.log').open('a') as log:
                log.write('RETRY with explicit frame-time assumption\n')
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        qa_path = scene_out / 'scannet_relative_direction_review.jsonl'
        count = len(read_jsonl(qa_path)) if result.returncode == 0 and qa_path.exists() else 0
        accepted_count = count > 0 if args.allow_partial_count else count == args.per_scene
        state['scenes'][scene] = {'status': 'ok' if accepted_count else 'failed', 'count': count, 'path': str(qa_path), 'exit_code': result.returncode}
        total += count
        manifest.write_text(json.dumps(state, indent=2) + '\n')
        print(f'{scene}: {count} cases, total {total}/{args.target}', flush=True)
    if total < args.target:
        raise RuntimeError(f'Only {total}/{args.target} generated; see {manifest} and generation.log')
    rows = []
    for scene, item in state['scenes'].items():
        if item['status'] == 'ok':
            rows.extend(read_jsonl(Path(item['path'])))
    rows = rows[:args.target]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ScanNet IDs')
    if len({r['video_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ScanNet video paths')
    if {r['id'] for r in rows} & {r['id'] for r in benchmark}:
        raise ValueError('Benchmark ID overlap')
    write_jsonl(out / 'scannet_spatial_candidates.jsonl', rows)
    (out / 'pool_summary.json').write_text(json.dumps({'target': args.target, 'generated': len(rows), 'physical_scenes': len({physical_scene(r) for r in rows}), 'answer_counts': dict(Counter((r['answer_key'] for r in rows))), 'training_ready': False, 'requires_human_review': len(rows), 'source_sha256': hashlib.sha256(args.benchmark.read_bytes()).hexdigest()}, indent=2) + '\n')

def replace(args):
    old = read_jsonl(args.input)
    spatial = read_jsonl(args.candidates)
    labels = {r['id']: r for r in read_jsonl(args.labels)}
    benchmark = read_jsonl(args.benchmark)
    benchmark_ids = {r['id'] for r in benchmark}
    benchmark_videos = {str(Path(r['video_path']).resolve()) for r in benchmark}
    benchmark_scenes = {physical_scene(r) for r in benchmark if r.get('dataset_source') == 'scannet'}
    if any((r.get('dataset_source') != 'scannet' or r.get('task_type') != 'relative_direction' for r in spatial)):
        raise ValueError('Candidate pool contains non-ScanNet spatial rows')
    accepted = []
    for row in spatial:
        label = labels.get(row['id'], {})
        if label.get('decision') == 'accept' and label.get('video_quality') == 'pass' and (label.get('question_unique') == 'yes') and (label.get('gold_answer_correct') == 'yes') and (label.get('gold_reasoning_correct') == 'yes'):
            accepted.append(row)
    needed = sum((r['task_type'] == 'relative_direction' for r in old))
    if len(accepted) < needed:
        raise RuntimeError(f'{len(accepted)}/{needed} ScanNet cases fully accepted; final training export blocked')
    selected = accepted[:needed]
    if any((r['id'] in benchmark_ids or str(Path(r['video_path']).resolve()) in benchmark_videos or physical_scene(r) in benchmark_scenes for r in selected)):
        raise ValueError('Benchmark overlap')
    if len({r['id'] for r in selected}) != needed:
        raise ValueError('Duplicate accepted ScanNet IDs')
    if len({r['video_path'] for r in selected}) != needed:
        raise ValueError('Duplicate accepted video paths')
    scene_counts = Counter((physical_scene(r) for r in selected))
    if max(scene_counts.values()) > 10:
        raise ValueError('More than ten spatial cases from one physical scene')
    missing_videos = [r['video_path'] for r in selected if not Path(r['video_path']).is_file()]
    if missing_videos:
        raise FileNotFoundError(f'{len(missing_videos)} accepted videos missing, first: {missing_videos[0]}')
    if any((r['choices'][r['answer_key']] != r['answer'] for r in selected)):
        raise ValueError('Accepted ScanNet answer key/text mismatch')
    for row in selected:
        row['metadata'].update(human_review_required=False, evaluation_valid=True, outcome='human_verified_reference', rebuild_status='review_accepted')
        for reward in row['step_reward_labels']:
            reward.update(reward=1.0, label='correct', scorable=True, reward_basis='human_verified_gold_reference')
        row['reward_data']['step_reward_labels'] = row['step_reward_labels']
    merged = [r for r in old if r['task_type'] != 'relative_direction'] + selected
    if len(merged) != len(old) or len({r['id'] for r in merged}) != len(merged):
        raise ValueError('Final count or uniqueness mismatch')
    write_jsonl(args.output, merged)
    args.output.with_suffix('.summary.json').write_text(json.dumps({'total': len(merged), 'replaced_spatial': len(selected), 'source': str(args.input), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'task_counts': dict(Counter((r['task_type'] for r in merged))), 'spatial_scenes': len({physical_scene(r) for r in selected})}, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    generate = sub.add_parser('pool')
    generate.add_argument('--root', type=Path, required=True)
    generate.add_argument('--benchmark', type=Path, required=True)
    generate.add_argument('--dev', type=Path)
    generate.add_argument('--output-dir', type=Path, required=True)
    generate.add_argument('--target', type=int, default=240)
    generate.add_argument('--per-scene', type=int, default=10)
    generate.add_argument('--min-unique-categories', type=int, choices=range(3, 101), default=3, help='Scene search breadth only; per-question quality gates are unchanged')
    generate.add_argument('--max-scenes', type=int, default=80)
    generate.add_argument('--seed', type=int, default=20260916)
    generate.add_argument('--source-fps', type=float, default=30)
    generate.add_argument('--retry-failed', action='store_true', help='Revisit failed scenes, resuming partial official downloads')
    generate.add_argument('--allow-partial-count', action='store_true', help='Keep non-empty scene output without weakening quality thresholds')
    generate.add_argument('--preferred-answers', nargs='*', default=[], help='Keep only generated records with these semantic answers')
    export = sub.add_parser('replace')
    export.add_argument('--input', type=Path, required=True)
    export.add_argument('--candidates', type=Path, required=True)
    export.add_argument('--labels', type=Path, required=True)
    export.add_argument('--benchmark', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (pool if args.mode == 'pool' else replace)(args)
if __name__ == '__main__':
    main()
