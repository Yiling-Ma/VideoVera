"""Final data-construction component: formats/normalization.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from data_generator.io import preserve
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from data_generator.formats.messages import convert_row

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            preserve(temporary)
        except FileNotFoundError:
            pass
        raise

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_atomic(path, ''.join((json.dumps(row, ensure_ascii=False) + '\n' for row in rows)))

def write_json(path: Path, value: Any) -> None:
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def update_video(row: dict[str, Any], path: Path, nframes: int=24) -> None:
    row['video_path'] = str(path)
    videos = [item for message in row.get('messages', []) if message.get('role') == 'user' for item in message.get('content', []) if isinstance(item, dict) and item.get('type') == 'video']
    if len(videos) != 1:
        raise ValueError(f"Expected one chat video: {row.get('id')}")
    videos[0]['video'] = str(path)
    videos[0]['nframes'] = nframes
    for key in ('fps', 'min_frames', 'max_frames'):
        videos[0].pop(key, None)

def required_segments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    grounding = raw['grounding']
    if raw['task_type'] == 'event_order':
        return list(grounding['ordered_segments'])
    if grounding['query_direction'] == 'before':
        return [grounding['adjacent_segment'], grounding['anchor_segment']]
    return [grounding['anchor_segment'], grounding['adjacent_segment']]

def sampled_times(start: float, end: float, fps: float=30.0, count: int=24) -> list[float]:
    frame_count = max(2, int(math.floor((end - start) * fps)) + 1)
    indices = [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]
    return [start + frame_index / fps for frame_index in indices]

def support_counts(segments: list[dict[str, Any]], start: float, end: float, fps: float=30.0) -> list[int]:
    margin = 1.01 / fps
    times = sampled_times(start, end, fps=fps)
    return [sum((float(segment['start_sec']) + margin <= time <= float(segment['stop_sec']) - margin for time in times)) for segment in segments]

def tight_crop_bounds(segments: list[dict[str, Any]], fps: float=30.0) -> tuple[float, float] | None:
    first = min((float(segment['start_sec']) for segment in segments))
    last = max((float(segment['stop_sec']) for segment in segments))
    for padding_frames in range(1, 10):
        padding = padding_frames / fps
        start = max(0.0, first - padding)
        end = last + padding
        if min(support_counts(segments, start, end, fps)) >= 2:
            return (start, end)
    return None

def probe_video(path: Path) -> dict[str, Any]:
    command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate,nb_read_frames,width,height:format=duration', '-of', 'json', str(path)]
    data = json.loads(subprocess.check_output(command))
    stream = data['streams'][0]
    numerator, denominator = (int(value) for value in stream['avg_frame_rate'].split('/'))
    return {'duration_sec': float(data['format']['duration']), 'fps': numerator / denominator, 'frames': int(stream['nb_read_frames']), 'width': int(stream['width']), 'height': int(stream['height'])}

def render_temporal_crop(source: Path, output: Path, start: float, end: float) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.mp4')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{start:.6f}', '-i', str(source), '-t', f'{end - start:.6f}', '-vf', 'fps=30,scale=960:-2', '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    subprocess.run(command, check=True)
    probe = probe_video(temporary)
    if probe['frames'] < 24 or probe['duration_sec'] <= 0:
        raise ValueError(f'Invalid rendered crop: {output}')
    temporary.replace(output)
    return probe

def normalize_training_row(row: dict[str, Any], version: str) -> None:
    row['sample_weight'] = 1.0
    update_video(row, Path(row['video_path']))
    row.setdefault('metadata', {})['v6_6400_supplement_training'] = {'version': version, 'loss_mode': 'standard_native_token_ce', 'sample_weight': 1.0, 'nframes': 24, 'uniform_sampling': True}

def convert_relative_candidate(row: dict[str, Any]) -> dict[str, Any]:
    converted = convert_row(copy.deepcopy(row), fps=1.0, max_pixels=151200, allow_unreviewed=True, review_override_note='Provisional V6 6400 supplement; generated with the frozen ScanNet quality pipeline and blocked from training pending human review.')
    normalize_training_row(converted, 'v6_6400_balanced_v1')
    converted['metadata']['v6_6400_relative_review'] = {'status': 'pending_human_review', 'training_ready': False, 'source_version': 'v6_relative_direction_supplement_pools_v4_v5_v6'}
    return converted

def shape_single(row, key, source_hash):
    from data_generator.formats import shape_labels as shape
    row = copy.deepcopy(row)
    label = shape.normalized_label(row['answer'])
    reasoning = shape.source_reasoning_steps(row, label)
    opts = shape.option_mapping(row['id'], label, key, 20261004)
    for msg in row['messages']:
        for part in msg['content']:
            if part.get('type') == 'text':
                part['text'] = shape.prompt_text(opts) if msg['role'] == 'user' else shape.assistant_text(reasoning, key)
    row.update(answer=label, answer_key=key, question="What's the shape of my moving trajectory?", choices=opts, gold_reasoning=reasoning, sample_weight=1.0)
    row.setdefault('metadata', {})['route_shape_redesign'] = {'version': shape.VERSION, 'source_id': row['id'], 'source_sha256': source_hash, 'seed': 20261004, 'ontology': list(shape.LABELS), 'all_five_options_present': True, 'deterministic_option_permutation': True, 'assistant_target': 'concise_grounded_steps_then_answer', 'benchmark_qa_used': False, 'answer_position_policy': 'preserve_removed_slot_position; combined quotas checked at V5 audit'}
    return row
