"""Final data-construction component: data/temporal/event_order.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from data_generator.io import preserve
import argparse
import collections
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from data_generator.formats.normalization import normalize_training_row, probe_video, render_temporal_crop, support_counts, tight_crop_bounds, write_json, write_jsonl
from data_generator.formats.messages import convert_row
from data_generator.sources.epic.generators import distinct_labels, has_ambiguous_overlap, is_local_window, make_event_order_record, segment_from_dict
from data_generator.formats.options import shuffle_group, stable_hash
from data_generator.formats.validation import validate_messages
SEED = 20261005
TASK = 'event_order'
TARGET = 384
ATTEMPT_BUFFER = 256

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]

def local_clip_path(manifest: dict[str, Any], clips_root: Path) -> Path:
    return clips_root / str(manifest['video_id']) / Path(str(manifest['clip_path'])).name

def local_segments(manifest: dict[str, Any]) -> list[Any]:
    origin = float(manifest['start_sec'])
    result = []
    for raw in manifest.get('segments', []):
        row = copy.deepcopy(raw)
        row['start_sec'] = float(row['start_sec']) - origin
        row['stop_sec'] = float(row['stop_sec']) - origin
        result.append(segment_from_dict(row))
    return result

def used_event_windows(rows: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {tuple((str(segment['segment_uid']) for segment in row['grounding']['ordered_segments'])) for row in rows if row.get('task_type') == TASK}

def enumerate_candidates(manifests: list[dict[str, Any]], clips_root: Path, used: set[tuple[str, ...]]) -> list[dict[str, Any]]:
    candidates = []
    seen = set(used)
    for manifest in manifests:
        clip_path = local_clip_path(manifest, clips_root)
        if not clip_path.is_file():
            continue
        segments = local_segments(manifest)
        for start_index in range(max(0, len(segments) - 3)):
            window = segments[start_index:start_index + 4]
            key = tuple((segment.uid for segment in window))
            if key in seen:
                continue
            if not distinct_labels(window) or has_ambiguous_overlap(window) or (not is_local_window(window, 18.0)):
                continue
            bounds = tight_crop_bounds([{'start_sec': segment.start_sec, 'stop_sec': segment.stop_sec} for segment in window])
            if bounds is None:
                continue
            seen.add(key)
            candidates.append({'video_id': str(manifest['video_id']), 'clip_id': str(manifest['clip_id']), 'clip_path': clip_path, 'window': window, 'window_key': key, 'bounds': bounds})
    return candidates

def select_diverse(candidates: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected = []
    video_counts = collections.Counter()
    clip_ids = set()
    while len(selected) < target:
        eligible = [item for item in remaining if item['clip_id'] not in clip_ids]
        if not eligible:
            raise RuntimeError(f'Only {len(selected)}/{target} distinct clips available')
        chosen = min(eligible, key=lambda item: (video_counts[item['video_id']], stable_hash(SEED, 'new-event', item['clip_id'], *item['window_key'])))
        remaining.remove(chosen)
        selected.append(chosen)
        video_counts[chosen['video_id']] += 1
        clip_ids.add(chosen['clip_id'])
    return selected

def make_id(item: dict[str, Any]) -> str:
    digest = hashlib.sha256('\x00'.join(item['window_key']).encode()).hexdigest()[:12]
    return f"v6supp_{item['video_id']}_event_order_{digest}"

def build_row(item: dict[str, Any], index: int, annotations: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    clip_probe = probe_video(item['clip_path'])
    record_id = make_id(item)
    raw = make_event_order_record(window=item['window'], video_path=item['clip_path'], annotations_path=annotations, clips_dir=item['clip_path'].parent, frames_dir=output_dir / 'support_frames', frames_per_segment=2, pad_before=0.0, pad_after=0.0, clip_duration_limit=18.0, multi_cache={}, sample_idx=index, record_id=record_id, clip_id=item['clip_id'], existing_clip_path=item['clip_path'], clip_start_sec=0.0, clip_end_sec=clip_probe['duration_sec'], source_video_path=item['clip_path'])
    if raw is None:
        raise ValueError(f'Frozen generator rejected selected window: {record_id}')
    converted = convert_row(copy.deepcopy(raw), fps=1.0, max_pixels=151200, allow_unreviewed=True, review_override_note='Metadata-derived EPIC action annotations with independent temporal gates.')
    output_video = output_dir / 'videos' / f'{record_id}.mp4'
    probe = render_temporal_crop(item['clip_path'], output_video, *item['bounds'])
    actual_end = item['bounds'][0] + probe['duration_sec']
    segments = [{'start_sec': segment.start_sec, 'stop_sec': segment.stop_sec} for segment in item['window']]
    support = support_counts(segments, item['bounds'][0], actual_end, probe['fps'])
    if min(support) < 2:
        raise ValueError(f'Post-render action support failed for {record_id}: {support}')
    normalize_training_row(converted, 'v6_6400_balanced_v1')
    from data_generator.formats.normalization import update_video
    update_video(converted, output_video)
    converted['metadata']['v6_6400_event_order_supplement'] = {'version': 'v6_6400_balanced_v1', 'source_clip_id': item['clip_id'], 'source_clip_path': str(item['clip_path']), 'source_window_segment_uids': list(item['window_key']), 'source_window_previously_used': False, 'crop_start_sec': item['bounds'][0], 'crop_end_sec': actual_end, 'uniform_24_support_counts': support, 'media_probe': probe, 'generator': 'epic_temporal_grounding_qa.generators.make_event_order_record'}
    return (raw, converted)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--existing-qa', type=Path, required=True)
    parser.add_argument('--clips-root', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'status').write_text('BUILDING\n')
    existing = read_jsonl(args.existing_qa)
    candidates = enumerate_candidates(read_jsonl(args.manifest), args.clips_root, used_event_windows(existing))
    distinct_clips = len({item['clip_id'] for item in candidates})
    attempted = select_diverse(candidates, min(distinct_clips, TARGET + ATTEMPT_BUFFER))
    raw_rows = []
    converted_rows = []
    selected = []
    rejected = []
    for attempt_index, item in enumerate(attempted, 1):
        try:
            raw, converted = build_row(item, attempt_index, args.annotations, args.output_dir)
        except (ValueError, subprocess.CalledProcessError) as exc:
            rejected.append({'clip_id': item['clip_id'], 'window_key': list(item['window_key']), 'reason': repr(exc)})
            output_video = args.output_dir / 'videos' / f'{make_id(item)}.mp4'
            preserve(output_video)
            continue
        raw_rows.append(raw)
        converted_rows.append(converted)
        selected.append(item)
        if len(converted_rows) % 20 == 0 or len(converted_rows) == TARGET:
            print(f'built {len(converted_rows)}/{TARGET}', flush=True)
        if len(converted_rows) == TARGET:
            break
    if len(converted_rows) != TARGET:
        raise RuntimeError(f'Only {len(converted_rows)}/{TARGET} rows passed post-render quality checks')
    shuffle_group(converted_rows, labels=list('ABCD'), seed=SEED)
    validate_messages(converted_rows, {TASK: TARGET})
    answer_positions = collections.Counter((row['answer_key'] for row in converted_rows))
    if answer_positions != collections.Counter(dict.fromkeys('ABCD', TARGET // 4)):
        raise ValueError(f'Answer positions are not balanced: {answer_positions}')
    video_counts = collections.Counter((item['video_id'] for item in selected))
    report = {'valid': True, 'version': 'v6_event_order_supplement_v1', 'candidate_windows': len(candidates), 'attempted_windows': len(selected) + len(rejected), 'post_render_rejections': len(rejected), 'selected': len(converted_rows), 'unique_ids': len({row['id'] for row in converted_rows}), 'unique_source_clips': len({item['clip_id'] for item in selected}), 'source_video_count': len(video_counts), 'max_rows_per_source_video': max(video_counts.values()), 'answer_positions': dict(answer_positions), 'uniform_24_min_support': min((min(row['metadata']['v6_6400_event_order_supplement']['uniform_24_support_counts']) for row in converted_rows)), 'all_videos_exist': all((Path(row['video_path']).is_file() for row in converted_rows)), 'benchmark_qa_used': False}
    write_jsonl(args.output_dir / 'raw_candidates.jsonl', raw_rows)
    write_jsonl(args.output_dir / 'selected_candidates.jsonl', converted_rows)
    write_jsonl(args.output_dir / 'post_render_rejections.jsonl', rejected)
    write_json(args.output_dir / 'summary.json', report)
    (args.output_dir / 'status').write_text('COMPLETED\n')
    print(json.dumps(report, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
