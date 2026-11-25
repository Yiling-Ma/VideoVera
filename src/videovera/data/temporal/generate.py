"""Generate temporal candidates from locally licensed official Training videos."""
import argparse
import collections
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from videovera.io import atomic_json, read_jsonl, write_jsonl, sha256
from videovera.sources.epic.annotations import load_segments
from videovera.sources.epic.generators import maybe_make_conditional_adjacent_record
from videovera.data.temporal import event_order as event
from videovera.formats.messages import convert_row
from videovera.formats.normalization import (required_segments, tight_crop_bounds,
    render_temporal_crop, support_counts, update_video, probe_video)


def adjacent(manifests, annotations):
    known = set()
    for manifest in manifests:
        segments = load_segments(annotations, manifest['video_id'])
        local = event.local_segments(manifest)
        local_ids = {s.uid for s in local}
        for i, anchor in enumerate(segments):
            for direction, delta in [('next', 1), ('previous', -1)]:
                j = i + delta
                if not 0 <= j < len(segments) or anchor.uid not in local_ids:
                    continue
                neighbor = segments[j]
                key = (anchor.uid, neighbor.uid, 'after' if delta == 1 else 'before')
                if neighbor.uid not in local_ids or key in known:
                    continue
                known.add(key)
                yield dict(manifest=manifest, segments=segments, local=local,
                           idx=i, direction=direction, key=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['event_order', 'before_or_after_action'])
    parser.add_argument('--annotations', type=Path, required=True, help='Official EPIC_100_train.csv')
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--existing', type=Path, help='Previously generated raw candidates to exclude')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    config = dict(task=args.task, annotations_sha256=sha256(args.annotations),
                  benchmark_sha256=sha256(args.benchmark), video_root=str(args.video_root.resolve()), count=args.count)
    cfg = out / 'config.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError('Configuration changed; use a new output directory')
    atomic_json(cfg, config)
    excluded = {str(r.get('scene_id') or r.get('video_id')) for r in read_jsonl(args.benchmark)
                if r['task_type'] in ('event_order', 'before_or_after_action')}
    existing = read_jsonl(args.existing) if args.existing else []
    manifests = []
    for path in sorted(args.video_root.rglob('*')):
        if path.suffix.lower() != '.mp4' or path.stem in excluded:
            continue
        vid = path.stem; segments = load_segments(args.annotations, vid)
        if not segments:
            continue
        probe = probe_video(path)
        manifests.append(dict(video_id=vid, clip_id=vid, clip_path=str(path), start_sec=0.,
                              end_sec=probe['duration_sec'], segments=[dataclasses.asdict(s) for s in segments]))
    event.local_clip_path = lambda manifest, clips_root: Path(manifest['clip_path'])
    used = event.used_event_windows(existing)
    candidates = event.enumerate_candidates(manifests, args.video_root, used) if args.task == 'event_order' else list(adjacent(manifests, args.annotations))
    previous_pairs = set()
    for row in existing:
        g = row.get('grounding', {})
        if row.get('task_type') == 'before_or_after_action' and g.get('anchor') and g.get('neighbor'):
            previous_pairs.add((g['anchor']['segment_uid'], g['neighbor']['segment_uid'], g['relation']))
    candidates.sort(key=lambda c: hashlib.sha256(str(c.get('window_key', c.get('key'))).encode()).hexdigest())
    counts = collections.Counter(); rows = []; failures = []
    for item in candidates:
        vid = item['video_id'] if args.task == 'event_order' else item['manifest']['video_id']
        cap = 45 if args.task == 'event_order' else 50
        if counts[vid] >= cap:
            continue
        identifier = event.make_id(item) if args.task == 'event_order' else vid + '_before_after_' + hashlib.sha256(str(item['key']).encode()).hexdigest()[:16]
        folder = out / 'items' / identifier; folder.mkdir(parents=True, exist_ok=True)
        done = folder / 'raw.json'
        if done.exists():
            raw = json.loads(done.read_text())
        else:
            if (folder / 'rejected.json').exists():
                continue
            try:
                if args.task == 'event_order':
                    raw, converted = event.build_row(item, len(rows)+1, args.annotations, folder)
                    raw['video_path'] = converted['video_path']
                    raw.setdefault('metadata', {})['continuous_source'] = converted['metadata']['v6_6400_event_order_supplement']
                else:
                    if item['key'] in previous_pairs:
                        continue
                    source = Path(item['manifest']['clip_path'])
                    raw = maybe_make_conditional_adjacent_record(idx=item['idx'], direction=item['direction'],
                        segments=item['segments'], local_segments=item['local'], video_path=source,
                        annotations_path=args.annotations, clips_dir=folder/'scratch', frames_dir=folder/'support_frames',
                        pad_before=0., pad_after=0., clip_duration_limit=18., multi_cache={}, existing_clip_path=source,
                        clip_start_sec=0., clip_end_sec=item['manifest']['end_sec'], clip_id=identifier,
                        source_video_path=source, sample_idx=len(rows)+1, record_id=identifier)
                    if raw is None:
                        raise ValueError('Original adjacent-action generator rejected candidate')
                    needed = required_segments(raw); bounds = tight_crop_bounds(needed)
                    if bounds is None:
                        raise ValueError('No continuous crop with unchanged uniform-frame support')
                    video = folder / 'video.mp4'; probe = render_temporal_crop(source, video, *bounds)
                    hits = support_counts(needed, bounds[0], bounds[0]+probe['duration_sec'], probe['fps'])
                    if min(hits) < 2:
                        raise ValueError('Fewer than two uniform model samples for a required action')
                    raw['video_path'] = str(video)
                    raw.setdefault('metadata', {})['continuous_source'] = dict(video=str(source), start=bounds[0], end=bounds[1], uniform_support=hits)
                raw.setdefault('metadata', {})['source_split'] = 'Training'
                raw['human_review_required'] = True
                atomic_json(done, raw)
            except ValueError as exc:
                failure = dict(id=identifier, reason=str(exc), partial_outputs_retained=True)
                atomic_json(folder/'rejected.json', failure); failures.append(failure)
                continue
        rows.append(raw); counts[vid] += 1
        if len(rows) >= args.count:
            break
    final = out / 'candidates.jsonl'
    if final.exists():
        if read_jsonl(final) != rows:
            raise ValueError('Final output differs; previous output retained')
    else:
        write_jsonl(final, rows)
    atomic_json(out/'summary.json', dict(candidate_count=len(rows), requested=args.count,
                per_recording=dict(counts), rejected=failures, accepted_count=0, human_review_required=True))

if __name__ == '__main__':
    main()
