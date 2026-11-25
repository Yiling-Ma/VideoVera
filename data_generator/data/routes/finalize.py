"""Final data-construction component: data/routes/finalize.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from data_generator.io import preserve
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import tempfile
from data_generator.data.routes.generate import candidates, dump, render_one, file_sha256

def publish(path, rows):
    text = ''.join((json.dumps(r, allow_nan=False) + '\n' for r in rows))
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError('Refusing to change published snapshot: ' + str(path))
    else:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.publish-')
        try:
            with os.fdopen(fd, 'w') as handle:
                handle.write(text)
            os.link(temporary, path)
        finally:
            preserve(temporary)

def finalize(source, output, seed, count):
    plan = json.loads((output / 'plan.json').read_text())
    map_path = output / 'duplicate_replacements.json'
    replacements = json.loads(map_path.read_text()) if map_path.exists() else {}
    slots = {c['id']: json.loads((output / 'completed' / (replacements.get(c['id'], c['id']) + '.json')).read_text()) for c in plan}
    original_counts = Counter(((r['task_type'], r['answer'], r['answer_key']) for r in slots.values()))
    seen, pending = ({}, [])
    kept = []
    for identity, row in slots.items():
        digest = row['metadata']['video_sha256']
        if file_sha256(Path(row['video_path'])) != digest:
            raise ValueError('Corrupted completed media:' + row['id'])
        key = (row['task_type'], digest)
        if key in seen:
            if seen[key] != row['answer']:
                raise ValueError('Identical video with conflicting gold labels')
            pending.append(identity)
        else:
            seen[key] = row['answer']
            kept.append(row)
    print('duplicate_clips_to_replace=%d' % len(pending), flush=True)
    if pending:
        pool, _ = candidates(source)
        rng = random.Random(seed)
        for bucket in pool.values():
            rng.shuffle(bucket)
        attempted = set()
        for path in (output / 'completed').glob('*.json'):
            r = json.loads(path.read_text())
            attempted.add((r['task_type'], r['metadata']['parent_video_sha256'], *r['metadata']['parent_frame_range']))
        parents = Counter(((r['task_type'], r['metadata']['parent_video_sha256']) for r in kept))
        scenes = Counter(((r['task_type'], r['scene_id']) for r in kept))
        next_index = len(plan) + len(attempted)
        for identity in pending:
            old = slots[identity]
            task, answer = (old['task_type'], old['answer'])
            for attempt in range(100):
                eligible = [c for c in pool[task, answer] if (task, c['parent'], c['start'], c['end']) not in attempted and parents[task, c['parent']] < 5]
                if not eligible:
                    raise ValueError('No genuinely new clips for ' + task + ':' + answer)
                c = dict(min(eligible, key=lambda x: (parents[task, x['parent']], scenes[task, x['scene']])))
                c.update(task=task, answer=answer, letter=old['answer_key'], index=next_index, id='route_repair_v2_%s_%04d_%s' % (task, next_index, c['parent'][:12]))
                next_index += 1
                attempted.add((task, c['parent'], c['start'], c['end']))
                if file_sha256(Path(c['source_video'])) != c['parent']:
                    raise ValueError('Replacement parent hash mismatch')
                row = render_one(c, output, seed)
                key = (task, row['metadata']['video_sha256'])
                if key in seen:
                    if seen[key] != answer:
                        raise ValueError('Replacement conflicts with existing gold')
                    continue
                seen[key] = answer
                slots[identity] = row
                replacements[identity] = row['id']
                dump(map_path, replacements)
                parents[task, c['parent']] += 1
                scenes[task, c['scene']] += 1
                print('replaced_duplicate=' + identity, flush=True)
                break
            else:
                raise ValueError('Too many duplicate candidates for ' + identity)
    rows = list(slots.values())
    if Counter(((r['task_type'], r['answer'], r['answer_key']) for r in rows)) != original_counts:
        raise ValueError('Deduplication changed class/answer-position quotas')
    summary = {'status': 'DATA_REBUILT_NOT_TRAINED', 'scope': 'training_only', 'duplicate_replacements': len(replacements), 'tasks': {}}
    for task in ('route_shape', 'reverse_route_plan'):
        group = sorted([r for r in rows if r['task_type'] == task], key=lambda r: r['id'])
        if len(group) != count or len({r['metadata']['video_sha256'] for r in group}) != count:
            raise ValueError('Wrong count or duplicate media in final ' + task)
        path = output / (task + '.jsonl')
        publish(path, group)
        summary['tasks'][task] = dict(records=count, gold_counts=dict(Counter((r['answer'] for r in group))), answer_key_counts=dict(Counter((r['answer_key'] for r in group))), unique_parent_videos=len({r['metadata']['parent_video_sha256'] for r in group}), scene_count=len({r['scene_id'] for r in group}), sha256=file_sha256(path), max_clips_per_parent=max(Counter((r['metadata']['parent_video_sha256'] for r in group)).values()))
    dump(output / 'summary.json', summary)
    return summary
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    config = json.loads((args.output / 'config.json').read_text())
    print(json.dumps(finalize(Path(config['source']), args.output, config['seed'], config['count']), indent=2))
