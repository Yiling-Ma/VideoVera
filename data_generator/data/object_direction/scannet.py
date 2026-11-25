"""Final data-construction component: data/object_direction/scannet.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import argparse
import copy
import io
import itertools
import json
import random
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
from data_generator.sources.spatial_geometry.scannet import load_scannet_objects
from data_generator.sources.spatial_geometry.generators import generate_object_relative_direction
from data_generator.formats.raw import build_sft_row
from data_generator.data.object_direction.descriptions import audit
from data_generator.data.object_direction.quality import ROLE_KEYS, context_metrics, role_signature, target_extent_check
BASE = 'http://kaldir.vc.cit.tum.de/scannet/'
VISUAL_EXCLUSIONS = {'scene0000_00': {'16': 'Several pillows are visible; category-count uniqueness is insufficient.', '34': 'Annotated coffee table appears upholstered; ordinary referent is uncertain.', '54': 'Door reference is ambiguous across the interior views.', '56': 'Shelf annotation spans a grouped bookcase/cabinet assembly.'}, 'scene0477_00': {'15': 'Human review could not locate the bathtub in the generated video.'}, 'scene0235_00': {'13': 'Human review could not locate the cabinet in the generated video.'}, 'scene0640_00': {'11': 'Human review found the kitchen-cabinet annotation distributed across multiple locations.'}, 'scene0604_00': {'11': 'Human review could not identify this coffee-table instance in the raw RGB clip.'}}

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    print('Downloading', url, flush=True)
    for attempt in range(12):
        result = subprocess.run(['curl', '-fL', '-C', '-', '--retry', '2', '--connect-timeout', '20', '--speed-time', '120', '--speed-limit', '1024', url, '-o', str(part)])
        if result.returncode == 0:
            break
        print(f'Download interrupted ({result.returncode}); resuming from {(part.stat().st_size if part.exists() else 0)} bytes (attempt {attempt + 1}/12)', flush=True)
    else:
        raise RuntimeError(f'Download failed after resumable attempts: {url}')
    part.replace(path)

def prepare(root, scene):
    directory = root / 'scans' / scene
    for suffix in ('.aggregation.json', '_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '_2d-instance-filt.zip', '.txt'):
        fetch(BASE + f'v2/scans/{scene}/{scene}{suffix}', directory / (scene + suffix))

def read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError('Truncated ScanNet sensor file')
    return data

def unpack(f, fmt):
    return struct.unpack('<' + fmt, read_exact(f, struct.calcsize('<' + fmt)))

def sensor_index(path, fallback_source_fps=None, force_frame_time=False):
    """Index compressed JPEGs without loading the multi-GB stream into RAM."""
    result = []
    with path.open('rb') as f:
        if unpack(f, 'I')[0] != 4:
            raise ValueError('Only ScanNet .sens version 4 is supported')
        read_exact(f, unpack(f, 'Q')[0])
        matrices = [unpack(f, '16f') for _ in range(4)]
        color, depth = unpack(f, 'ii')
        width, height, dw, dh = unpack(f, '4I')
        shift = unpack(f, 'f')[0]
        count = unpack(f, 'Q')[0]
        if color != 2:
            raise ValueError('Expected JPEG color compression')
        for i in range(count):
            pose = unpack(f, '16f')
            stamp, depth_stamp, nc, nd = unpack(f, '4Q')
            result.append(dict(frame_id=i, timestamp_us=stamp, offset=f.tell(), size=nc, camera_to_world=list(pose) if all(np.isfinite(pose)) else None))
            f.seek(nc + nd, 1)
        if f.tell() > path.stat().st_size:
            raise ValueError('Sensor payload exceeds file size')
    stamps = [r['timestamp_us'] for r in result]
    if len(stamps) < 2:
        raise ValueError('Insufficient sensor frames')
    missing = all((t == 0 for t in stamps))
    if force_frame_time and (fallback_source_fps is None or not np.isfinite(fallback_source_fps) or fallback_source_fps <= 0):
        raise ValueError('Forced frame time requires a finite positive source FPS')
    if (missing or force_frame_time) and fallback_source_fps is not None and (fallback_source_fps > 0):
        for r in result:
            r['timestamp'] = r['frame_id'] / fallback_source_fps
            reason = 'timestamps_explicitly_ignored' if force_frame_time else 'timestamps_missing'
            r['time_basis'] = f'assumed_source_fps_{fallback_source_fps:g}_{reason}'
        return result
    if any((b <= a for a, b in zip(stamps, stamps[1:]))):
        raise ValueError('Non-monotonic color timestamps; explicitly set --fallback-source-fps only for all-zero timestamps')
    for r in result:
        r['timestamp'] = (r['timestamp_us'] - stamps[0]) / 1000000.0
        r['time_basis'] = 'sensor_color_timestamp_microseconds'
    return result

def visibility(zip_path, frames, objects, stride=15, min_area=0.02):
    """Use official filtered instance masks; pixel instance ID = objectId + 1."""
    wanted = {int(o.uid) + 1: o.object_key for o in objects}
    visible = {o.object_key: [] for o in objects}
    with zipfile.ZipFile(zip_path) as z:
        names = sorted((n for n in z.namelist() if n.endswith('.png') and Path(n).stem.isdigit()), key=lambda n: int(Path(n).stem))
        for n in names:
            i = int(Path(n).stem)
            if i % stride or i >= len(frames):
                continue
            arr = np.array(Image.open(io.BytesIO(z.read(n))))
            for instance, key in wanted.items():
                y, x = np.where(arr == instance)
                ratio = len(x) / arr.size
                if ratio < min_area:
                    continue
                visible[key].append(dict(frame_id=i, timestamp=frames[i]['timestamp'], pixel_area_ratio=ratio, bbox_area_ratio=ratio, bbox_2d=[int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)], mask_size=[arr.shape[1], arr.shape[0]], mask_file=n, source=str(zip_path), instance_pixel_id=instance))
    return visible

def expand_clip_window(start, end, duration, min_seconds=8, padding=1):
    if duration + 1e-06 < min_seconds or start < 0 or end < start or (end > duration + 1e-06):
        return None
    clip_start = max(0, start - padding)
    clip_end = min(duration, end + padding)
    missing = min_seconds - (clip_end - clip_start)
    if missing > 0:
        add_left = min(clip_start, missing / 2)
        clip_start -= add_left
        missing -= add_left
        add_right = min(duration - clip_end, missing)
        clip_end += add_right
        missing -= add_right
        if missing > 0:
            add_left = min(clip_start, missing)
            clip_start -= add_left
            missing -= add_left
    if clip_end - clip_start + 1e-06 < min_seconds:
        return None
    return (clip_start, clip_end)

def common_window(role_hits, duration, max_seconds=30, min_seconds=8, min_hits=3, min_peak_area=0.03):
    times = sorted({h['timestamp'] for hits in role_hits for h in hits})
    best = None
    best_rank = None
    for start in times:
        following = [sorted((h['timestamp'] for h in hits if h['timestamp'] >= start)) for hits in role_hits]
        if any((len(t) < min_hits for t in following)):
            continue
        end = max((t[min_hits - 1] for t in following))
        if end - start > max_seconds:
            continue
        clip = expand_clip_window(start, end, duration, min_seconds)
        if clip is None:
            continue
        clip_start, clip_end = clip
        subsets = [[h for h in hits if clip_start <= h['timestamp'] <= clip_end] for hits in role_hits]
        if any((len(hits) < min_hits for hits in subsets)):
            continue
        evidence = [max(hits, key=lambda h: h['pixel_area_ratio']) for hits in subsets]
        score = min((h['pixel_area_ratio'] for h in evidence))
        if score < min_peak_area:
            continue
        context = context_metrics(subsets)
        if not context['connected']:
            continue
        rank = (context['triple_shared_frame_count'], context['co_visible_pair_count'], min((len(hits) for hits in subsets)), score, -(end - start))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = (score, clip_start, clip_end, subsets)
    return best

def render_clip(sensor, frames, start, end, output, cache, min_duration=8):
    selected = []
    last = -1000000000.0
    for f in frames:
        if start <= f['timestamp'] <= end and f['timestamp'] - last >= 0.095:
            selected.append(f)
            last = f['timestamp']
    if len(selected) < 2:
        raise ValueError('Insufficient video frames')
    cache.mkdir(parents=True, exist_ok=True)
    with sensor.open('rb') as source:
        for f in selected:
            path = cache / f"{f['frame_id']:06d}.jpg"
            if not path.exists():
                source.seek(f['offset'])
                path.write_bytes(read_exact(source, f['size']))
    listing = output.with_suffix('.ffconcat')
    lines = ['ffconcat version 1.0']
    for i, f in enumerate(selected):
        duration = selected[i + 1]['timestamp'] - f['timestamp'] if i + 1 < len(selected) else 0.1
        lines += [f"file '{cache / (str(f['frame_id']).zfill(6) + '.jpg')}'", f'duration {duration:.6f}']
    lines.append(lines[-2])
    listing.write_text('\n'.join(lines) + '\n')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-safe', '0', '-i', str(listing), '-vf', 'scale=960:-2', '-r', '10', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_name,width,height', '-of', 'json', str(output)]))
    expected = selected[-1]['timestamp'] - selected[0]['timestamp'] + 0.1
    if abs(float(probe['format']['duration']) - expected) > 0.5:
        raise ValueError('Encoded duration differs from sensor timestamps')
    if float(probe['format']['duration']) + 0.15 < min_duration:
        raise ValueError(f'Encoded clip is shorter than {min_duration:g} seconds')
    return dict(source_start_sec=selected[0]['timestamp'], source_end_sec=selected[-1]['timestamp'], duration_sec=float(probe['format']['duration']), source_frame_ids=[f['frame_id'] for f in selected])

def align_evidence(value, clip_start):
    if isinstance(value, dict):
        if 'frame_id' in value and 'timestamp' in value:
            value['source_timestamp_sec'] = value['timestamp']
            value['clip_timestamp_sec'] = value['timestamp'] - clip_start
        for child in value.values():
            align_evidence(child, clip_start)
    elif isinstance(value, list):
        for child in value:
            align_evidence(child, clip_start)

def build(args):
    directory = args.root / 'scans' / args.scene
    out = args.output_dir or args.root / 'outputs' / 'sample10_quality_v2'
    out.mkdir(parents=True, exist_ok=True)
    final = out / 'scannet_relative_direction_review.jsonl'
    if final.exists():
        raise FileExistsError(f'Review file already exists; refusing to overwrite: {final}')
    agg = directory / f'{args.scene}.aggregation.json'
    refs = json.loads((args.root / 'scanrefer' / 'ScanRefer_filtered_train.json').read_text())
    refs = [r for r in refs if r['scene_id'] == args.scene]
    if not refs:
        raise ValueError('Scene not in ScanRefer training split')
    report = audit(refs, json.loads(agg.read_text()))
    candidates = {r['object_id']: r for r in report['unique_category_candidates']}
    exclusions = VISUAL_EXCLUSIONS.get(args.scene, {})
    objects = [o for o in load_scannet_objects(agg) if o.uid in candidates and o.uid not in exclusions]
    sensor = directory / f'{args.scene}.sens'
    frames = sensor_index(sensor, args.fallback_source_fps, args.force_frame_time)
    write_json(out / 'sensor_index.json', frames)
    hits = visibility(directory / f'{args.scene}_2d-instance-filt.zip', frames, objects)
    write_json(out / 'visibility.json', hits)
    objects = [o for o in objects if len(hits[o.object_key]) >= 2]
    names = {}
    for obj in objects:
        reference = candidates[obj.uid]['default_reference']
        names[obj.object_key] = reference[4:] if reference.startswith('the ') else reference
    rejected_roles = set()
    if bool(args.review_sample) != bool(args.review_labels):
        raise ValueError('--review-sample and --review-labels must be supplied together')
    if args.review_sample:
        reviewed = {r['id']: r for r in (json.loads(s) for s in args.review_sample.read_text().splitlines() if s.strip())}
        for line in args.review_labels.read_text().splitlines():
            if not line.strip():
                continue
            label = json.loads(line)
            if label['id'] not in reviewed:
                raise ValueError('Review label has no matching source sample: ' + label['id'])
            if label.get('decision') in ('drop', 'revise'):
                rejected_roles.add(role_signature(reviewed[label['id']]))
    policy = dict(version='scannet_quality_v4', target_extent_margin_m=args.target_extent_margin, max_evidence_span_sec=args.max_evidence_span, context_selection='co_visibility_then_compact_window', minimum_clip_duration_sec=args.min_clip_seconds, minimum_visible_samples_per_role=args.min_visible_samples, minimum_peak_mask_area_ratio_per_role=args.min_peak_area, require_connected_role_coview_graph=True, object_uniqueness='canonicalized scene category; grouped labels and conflicting colors excluded', review_sample=str(args.review_sample) if args.review_sample else None, review_labels=str(args.review_labels) if args.review_labels else None)
    rejected = []
    eligible = []
    for triple in itertools.combinations(objects, 3):
        window = common_window([hits[o.object_key] for o in triple], frames[-1]['timestamp'], args.max_evidence_span, args.min_clip_seconds, args.min_visible_samples, args.min_peak_area)
        if window:
            eligible.append((triple, window))
        else:
            rejected.append(dict(objects=[o.uid for o in triple], reason='no_compact_visibility_window'))
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    counts = {k: 0 for k in 'ABCD'}
    rows = []
    ranked = []
    for triple, window in eligible:
        _, start, end, subsets = window
        evidence = {}
        for obj, seen in zip(triple, subsets):
            best = max(seen, key=lambda h: h['pixel_area_ratio'])
            evidence[obj.object_key] = dict(best, num_visible_frames=len(seen), visible_frame_samples=seen, frame_name=f"{best['frame_id']:06d}.jpg")
        qa = generate_object_relative_direction(args.scene, agg, triple, 6, rng, 0.25, display_names=names, frame_evidence=evidence, min_role_bbox_area_ratio=0.02)
        options = []
        for record in qa:
            extent = target_extent_check(record['grounding'], args.target_extent_margin)
            reason = 'human_rejected_role_combination' if role_signature(record) in rejected_roles else 'target_extent_crosses_or_nears_boundary' if not extent['passed'] else None
            if reason:
                rejected.append(dict(id=record['id'], role_signature=role_signature(record), reason=reason, extent=extent))
                continue
            roles = [record['grounding'][k]['object_key'] for k in ROLE_KEYS]
            context = context_metrics([evidence[k]['visible_frame_samples'] for k in roles])
            record['grounding']['target_extent_check'] = extent
            record['grounding']['visual_context'] = context
            options.append(record)
        if options:
            context = context_metrics(subsets)
            ranked.append((context['connected'], context['co_visible_pair_count'], -(end - start), options, window))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    write_json(out / 'quality_filter_report.json', dict(policy=policy, rejected=rejected, reason_counts=dict(Counter((r['reason'] for r in rejected))), eligible_object_triples=len(ranked)))
    if len(ranked) < args.count and (not args.allow_partial_count):
        raise ValueError(f'Only {len(ranked)} quality-filtered triples for {args.count} requested; thresholds are not relaxed')
    requested = min(args.count, len(ranked)) if args.allow_partial_count else args.count
    semantic_counts = Counter()
    for _, _, _, options, window in ranked[:requested]:
        _, start, end, _ = window
        preferred = [record for record in options if record['answer'] in args.preferred_answers]
        if args.preferred_answers and (not preferred):
            continue
        choices = preferred or options
        record = min(choices, key=lambda r: (semantic_counts[r['answer']], not r['grounding']['visual_context']['observer_facing_co_visible'], counts[r['answer_key']]))
        record = copy.deepcopy(record)
        video = out / f'clip_{len(rows) + 1:02d}.mp4'
        timing = render_clip(sensor, frames, start, end, video, out / 'rgb', args.min_clip_seconds)
        align_evidence(record, timing['source_start_sec'])
        record['dataset_source'] = 'scannet'
        record['video_path'] = str(video)
        record['grounding']['note'] = 'ScanNet mesh bounding-box centers in the +Z-up world frame; x-y horizontal plane.'
        roles = [record['grounding'][k] for k in ('observer_object', 'facing_object', 'target_object')]
        a, b, c = [names[o['object_key']] for o in roles]
        record['question'] = f'Imagine standing at the center of the {a}, facing the center of the {b}. Where is the {c} relative to you?'
        record['metadata'] = dict(source_clip=timing, physical_scene_id=args.scene.rsplit('_', 1)[0], quality_policy=policy, video_time_basis=frames[0]['time_basis'], object_descriptions={o['uid']: candidates[o['uid']] for o in roles}, visibility_basis='ScanNet filtered instance masks, objectId + 1; sampled every 15 frames', description_policy='canonical category unique; stable ScanRefer appearance used when non-conflicting')
        row = build_sft_row(record)
        row['metadata'].update(human_review_required=True, evaluation_valid=False, outcome='reference_pending_review', rebuild_status='rebuilt_pending_visual_review', sft_source='scannet_pilot')
        for label in row['step_reward_labels']:
            label.update(reward=None, label=None, scorable=False, reward_basis='reference_pending_review')
        row['reward_data']['step_reward_labels'] = copy.deepcopy(row['step_reward_labels'])
        rows.append(row)
        counts[row['answer_key']] += 1
        semantic_counts[row['answer']] += 1
        print('Built', len(rows), row['question'], row['answer'], flush=True)
    if len(rows) != args.count and (not args.allow_partial_count):
        write_json(out / 'insufficient_candidates.json', dict(required=args.count, available=len(rows), eligible_triples=len(eligible)))
        raise ValueError(f'Only {len(rows)} suitable triples; do not weaken visibility/uniqueness filters')
    if not rows:
        write_json(out / 'insufficient_candidates.json', dict(required=args.count, available=0, eligible_triples=len(eligible)))
        raise ValueError('No suitable triples passed the unchanged quality filters')
    assert len({r['id'] for r in rows}) == len(rows)
    temp = final.with_suffix('.tmp')
    temp.write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in rows)))
    temp.replace(final)
    write_json(out / 'summary.json', dict(total=len(rows), scenes=[args.scene], answer_counts=counts, semantic_answer_counts=semantic_counts, requested_count=args.count, partial_count_allowed=args.allow_partial_count, preferred_answers=args.preferred_answers, unique_object_triples=len(rows), human_review_required=True, training_ready=False, output=str(final), visual_exclusions=exclusions, quality_policy=policy, filter_reason_counts=dict(Counter((r['reason'] for r in rejected)))))
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scene', default='scene0000_00')
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--allow-partial-count', action='store_true', help='Keep non-empty partial scene output without weakening quality thresholds')
    p.add_argument('--preferred-answers', nargs='*', default=[], help='Keep only generated records with these semantic answers')
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--fallback-source-fps', type=float)
    p.add_argument('--force-frame-time', action='store_true', help='Explicitly ignore unreliable sensor times; requires --fallback-source-fps')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--target-extent-margin', type=float, default=0.1)
    p.add_argument('--max-evidence-span', type=float, default=30)
    p.add_argument('--min-clip-seconds', type=float, default=8)
    p.add_argument('--min-visible-samples', type=int, default=3)
    p.add_argument('--min-peak-area', type=float, default=0.03)
    p.add_argument('--review-sample', type=Path)
    p.add_argument('--review-labels', type=Path)
    args = p.parse_args()
    if args.count < 1 or not np.isfinite(args.max_evidence_span) or args.max_evidence_span <= 0:
        p.error('count and max-evidence-span must be positive')
    if not np.isfinite(args.target_extent_margin) or args.target_extent_margin < 0:
        p.error('target-extent-margin must be finite and non-negative')
    if not np.isfinite(args.min_clip_seconds) or args.min_clip_seconds <= 0 or args.min_visible_samples < 2 or (not np.isfinite(args.min_peak_area)) or (not 0 < args.min_peak_area < 1):
        p.error('clip/visibility thresholds are invalid')
    prepare(args.root, args.scene)
    if not args.prepare_only:
        fetch(BASE + f'v1/scans/{args.scene}/{args.scene}.sens', args.root / 'scans' / args.scene / (args.scene + '.sens'))
        build(args)
