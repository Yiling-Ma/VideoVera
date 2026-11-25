"""Accept reviewed candidates and export separated policy/reference artifacts."""
import argparse
import collections
import copy
import json
import re
import shutil
from pathlib import Path
from videovera.io import atomic_json, read_jsonl, write_jsonl, sha256
from videovera.formats import targets, canonical
from videovera.formats.messages import convert_row
from videovera.formats.raw import build_sft_row
from videovera.formats.normalization import convert_relative_candidate, normalize_training_row, shape_single
from videovera.data.landmark_routes.selection import full_row, validate_pool_row
from videovera.data.object_direction.quality import target_extent_check
from videovera.validation.media import decoded
from videovera.validation.reference import gold_completion, score_completion, RewardWeights
from videovera.validation.trace import validate_trace

TASKS = ('relative_direction', 'egocentric_temporal_direction', 'route_plan',
         'object_grounded_route_planning', 'route_shape', 'reverse_route_plan',
         'event_order', 'before_or_after_action')


def family(row):
    if row['task_type'] in ('event_order', 'before_or_after_action'):
        return 'epic'
    if row['task_type'] in ('egocentric_temporal_direction', 'route_plan', 'route_shape', 'reverse_route_plan'):
        return 'r2r'
    return 'scannet' if 'scannet' in str(row.get('dataset_source', '')).lower() else 'arkitscenes'


def group(row, registry, require_train=True):
    metadata = row.get('metadata', {})
    source = str(metadata.get('source_scene_id') or row.get('scene_id') or row.get('video_id') or '')
    kind = family(row)
    item = registry.get(kind, {}).get(source)
    if not item or not item.get('physical_scene_id'):
        raise ValueError(f'Missing official physical source identity: {kind}/{source}')
    if require_train and item.get('split') not in ('train', 'Training'):
        raise ValueError(f'Non-Training source: {kind}/{source}')
    return kind, str(item['physical_scene_id'])


def question(row):
    if row.get('question'):
        return re.sub(r'\s+', ' ', row['question']).strip()
    text = '\n'.join(p['text'] for m in row.get('messages', []) if m['role'] == 'user'
                     for p in m['content'] if p.get('type') == 'text')
    return re.sub(r'\s+', ' ', text.split('\n\nOptions:', 1)[0]).strip()


def semantic(row, identity):
    task = row['task_type']; meta = row.get('metadata', {}); ground = row.get('grounding') or meta.get('grounding') or row.get('structured_gold', {}).get('grounding', {})
    if task == 'relative_direction':
        roles = [ground.get(key, {}) for key in ('observer_object', 'facing_object', 'target_object')]
        if not all(r.get('object_key') or r.get('key') for r in roles):
            raise ValueError('Missing spatial referent identities')
        return (task, identity, tuple(sorted(str(r.get('object_key') or r['key']) for r in roles)))
    if task == 'object_grounded_route_planning':
        return (task, identity, tuple(meta['route_object_keys']), str(meta['facing_object_key']))
    if task == 'event_order':
        return (task, tuple(s['segment_uid'] for s in ground['ordered_segments']))
    if task == 'before_or_after_action':
        return (task, ground['anchor']['segment_uid'], ground['neighbor']['segment_uid'], ground['relation'])
    pose = ground.get('pose_route', {}) or {}
    signature = meta.get('pose_signature') or pose.get('pose_signature') or ground.get('pose_signature')
    if not signature and meta.get('parent_video_sha256') and meta.get('parent_frame_range'):
        signature = (meta['parent_video_sha256'], tuple(meta['parent_frame_range']))
    if not signature and task == 'route_shape' and ground.get('in_place_evidence'):
        signature = ('validated_stationary_clip', sha256(row['video_path']))
    if not signature:
        # Measured trajectory identity is required; renamed IDs are insufficient.
        path = meta.get('trajectory_path')
        if not path:
            raise ValueError('Missing measured route / trajectory identity')
        signature = sha256(path)
    return task, identity, signature, question(row)


def reviewed(label):
    return (label.get('decision') == 'accept' and label.get('video_quality') == 'pass'
            and label.get('question_unique') == 'yes' and label.get('gold_answer_correct') == 'yes'
            and label.get('gold_reasoning_correct') == 'yes')



def approved_protocol(approval, source_family, task, labels, policy_hash):
    samples = approval.get('reviewed_sample_ids', [])
    return (bool(policy_hash) and approval.get('protocol_sha256') == policy_hash
            and approval.get('family') == source_family and approval.get('task_type') == task
            and approval.get('decision') == 'approve_pipeline'
            and approval.get('allow_bulk_generation') is True
            and approval.get('allow_admission_after_checks') is True
            and bool(approval.get('reviewer')) and bool(approval.get('statement'))
            and bool(samples) and all(reviewed(labels.get(str(i), {})) for i in samples))

def convert_reference(raw, source):
    task = raw['task_type']; raw = copy.deepcopy(raw)
    if task == 'relative_direction':
        if 'grounding' not in raw.get('metadata', {}):
            raw.setdefault('metadata', {})['grounding'] = raw['grounding']
        ground = raw['metadata']['grounding']
        if not target_extent_check(ground, .1)['passed']:
            raise ValueError('Target crosses a spatial decision boundary')
        row = convert_relative_candidate(build_sft_row(raw))
    elif task == 'object_grounded_route_planning':
        validate_pool_row(raw, set()); row = full_row(raw)
    else:
        row = convert_row(raw, fps=1., max_pixels=151200, allow_unreviewed=True,
                          review_override_note='Explicit candidate or reviewed source-protocol approval verified by packaging CLI')
    if task == 'route_shape':
        row = shape_single(row, raw['answer_key'], sha256(source))
    normalize_training_row(row, 'videovera')
    builder = {'relative_direction': targets.relative_gold, 'event_order': targets.temporal_gold,
        'before_or_after_action': targets.temporal_gold, 'object_grounded_route_planning': targets.object_route_gold,
        'route_plan': targets.route_plan_gold, 'reverse_route_plan': targets.reverse_route_gold,
        'route_shape': targets.route_shape_gold, 'egocentric_temporal_direction': targets.etd_gold}[task]
    gold = canonical.update_gold(builder(row, raw, source), task)
    answer = gold_completion(gold)
    errors = validate_trace(json.loads(answer), task, set(gold['choices']))
    if errors:
        raise ValueError('Reference schema validation failed: ' + str(errors))
    score = score_completion(answer, {'task_type':task, 'structured_gold':gold}, weights=RewardWeights(answer=.40, process=.45, joint=.10))
    if not score.valid_json_schema or not score.final_answer_correct or not score.trace_exact or abs(score.total-1.) > 1e-8:
        raise ValueError('Reference fails original reward-one validation')
    row['structured_gold'] = gold
    row.setdefault('metadata', {})['human_review_complete'] = False
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, action='append', required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--source-registry', type=Path, required=True)
    parser.add_argument('--review-labels', type=Path, required=True)
    parser.add_argument('--source-approvals', type=Path, help='Explicit reviewed source-protocol approvals for bulk admission')
    parser.add_argument('--quality-policy', type=Path, help='Policy file whose SHA256 is bound by source approvals')
    parser.add_argument('--existing', type=Path)
    parser.add_argument('--existing-targets', type=Path)
    parser.add_argument('--warmup', type=Path, required=True, help='Frozen warm-up JSONL; copied byte-for-byte')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new export directory; previous artifacts are immutable')
    labels = {str(x['id']): x for x in read_jsonl(args.review_labels)}
    registry = json.loads(args.source_registry.read_text())
    approvals = json.loads(args.source_approvals.read_text()) if args.source_approvals else []
    policy_hash = sha256(args.quality_policy) if args.quality_policy else None
    base = read_jsonl(args.existing) if args.existing else []
    golds = {str(x['id']): x.get('structured_gold', x.get('target', x)) for x in read_jsonl(args.existing_targets)} if args.existing_targets else {}
    for row in base:
        if not row.get('structured_gold'):
            if str(row['id']) not in golds:
                raise ValueError('Existing records require their original reference targets')
            row['structured_gold'] = golds[str(row['id'])]
    bench = read_jsonl(args.benchmark)
    if not bench:
        raise ValueError('A non-empty frozen benchmark is required')
    forbidden = {group(r, registry, False) for r in bench}
    cache = {}
    def media(path):
        if path not in cache:
            cache[path] = decoded(path)
        return cache[path]
    benchmark_hashes = {media(r['video_path'])['decoded_sha256'] for r in bench}
    parent_counts = collections.Counter()
    counts = collections.Counter(); seen_ids = set(); seen_slots = set(); seen_semantics = set()
    for row in base:
        identity = group(row, registry)
        if identity in forbidden:
            raise ValueError('Existing source overlaps benchmark')
        if str(row['id']) in seen_ids:
            raise ValueError('Duplicate existing ID')
        seen_ids.add(str(row['id'])); counts[row['task_type'], identity] += 1
        parent = row.get('metadata', {}).get('parent_video_sha256')
        if parent:
            parent_counts[row['task_type'], parent] += 1
        digest = media(row['video_path'])['decoded_sha256']
        if digest in benchmark_hashes:
            raise ValueError('Existing decoded video content overlaps benchmark')
        seen_slots.add((row['task_type'], digest, question(row)))
        seen_semantics.add(semantic(row, identity))
    args.output_dir.mkdir(parents=True)
    accepted = []; rejected = []
    for source in args.candidates:
        for raw in read_jsonl(source):
            try:
                if raw.get('task_type') not in TASKS:
                    raise ValueError('Unsupported task')
                individual = reviewed(labels.get(str(raw['id']), {}))
                protocol_approval = next((a for a in approvals if approved_protocol(a, family(raw), raw['task_type'], labels, policy_hash)), None)
                if not individual and protocol_approval is None:
                    raise ValueError('Human review or explicit reviewed source-protocol approval incomplete')
                identity = group(raw, registry)
                if identity in forbidden:
                    raise ValueError('Physical source overlaps benchmark')
                if str(raw['id']) in seen_ids:
                    raise ValueError('Duplicate ID')
                row = convert_reference(raw, source)
                row['metadata']['human_review_complete'] = individual
                row['metadata']['human_review_status'] = 'per_record_accepted' if individual else 'source_protocol_accepted_not_individually_reviewed'
                if not individual:
                    row['metadata']['source_protocol_approval_sha256'] = sha256(args.source_approvals)
                digest = media(row['video_path'])['decoded_sha256']
                if digest in benchmark_hashes:
                    raise ValueError('Decoded content overlaps benchmark')
                slot = row['task_type'], digest, question(row)
                sem = semantic(row, identity)
                if slot in seen_slots or sem in seen_semantics:
                    raise ValueError('Duplicate source trajectory, semantics or decoded QA slot')
                cap = {'relative_direction':10, 'object_grounded_route_planning':14,
                       'event_order':45, 'before_or_after_action':50}.get(row['task_type'])
                if cap is not None and counts[row['task_type'], identity] >= cap:
                    raise ValueError('Combined existing/new physical-source cap reached')
                parent = row.get('metadata', {}).get('parent_video_sha256')
                if row['task_type'] in ('route_shape', 'reverse_route_plan') and parent and parent_counts[row['task_type'], parent] >= 5:
                    raise ValueError('Combined existing/new parent clip cap reached')
                accepted.append(row); counts[row['task_type'],identity] += 1
                if parent:
                    parent_counts[row['task_type'], parent] += 1
                seen_ids.add(str(row['id'])); seen_slots.add(slot); seen_semantics.add(sem)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                rejected.append(dict(id=raw.get('id'), source=str(source), reason=str(exc)))
            atomic_json(args.output_dir/'acceptance_progress.json', dict(accepted=len(accepted), rejected=rejected))
    full = base + accepted; by_id = {str(r['id']):r for r in full}
    warm = read_jsonl(args.warmup); warm_ids = {str(r['id']) for r in warm}
    if len(warm_ids) != len(warm) or not warm_ids <= by_id.keys():
        raise ValueError('Frozen warm-up is not a unique subset of the corpus')
    for row in warm:
        other = by_id[str(row['id'])]
        for key in ('answer', 'answer_key', 'task_type', 'messages'):
            if row.get(key) != other.get(key):
                raise ValueError('Frozen warm-up record content changed: '+str(row['id']))
    policy = []; hidden = []
    for row in full:
        if str(row['id']) in warm_ids:
            continue
        p = canonical.update_policy(targets.policy_row(row))
        policy.append(p); hidden.append(targets.hidden_target(row, row['structured_gold']))
    targets.no_policy_leak(policy)
    write_jsonl(args.output_dir/'accepted_new.jsonl', accepted)
    # Preserve the original existing JSONL prefix and warm-up bytes exactly.
    master = args.output_dir/'corpus.jsonl'
    with master.open('xb') as stream:
        if args.existing:
            payload = args.existing.read_bytes(); stream.write(payload)
            if payload and not payload.endswith(b'\n'):
                stream.write(b'\n')
        for row in accepted:
            stream.write((json.dumps(row, ensure_ascii=False)+'\n').encode())
    shutil.copyfile(args.warmup, args.output_dir/'warmup.jsonl')
    assert sha256(args.warmup) == sha256(args.output_dir/'warmup.jsonl')
    write_jsonl(args.output_dir/'policy.jsonl', policy)
    write_jsonl(args.output_dir/'references.jsonl', hidden)
    atomic_json(args.output_dir/'rejections.json', rejected)
    atomic_json(args.output_dir/'manifest.json', dict(total=len(full), warmup=len(warm), rl=len(policy),
        per_task=dict(collections.Counter(r['task_type'] for r in full)),
        accepted_new=len(accepted), rejected=len(rejected), frozen_warmup_sha256=sha256(args.warmup),
        benchmark_sha256=sha256(args.benchmark), source_registry_sha256=sha256(args.source_registry),
        input_files={str(p):sha256(p) for p in args.candidates}, full_decode_and_uniform_content_checks=True))

if __name__ == '__main__':
    main()
