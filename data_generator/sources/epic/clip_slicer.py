from __future__ import annotations
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from .annotations import ActionSegment, load_segments
from .dataset import list_available_videos, resolve_epic_paths
from .generators import humanize_action, is_local_window
from .media import extract_clip_range

@dataclass
class ClipSpec:
    clip_id: str
    video_id: str
    source_video_path: str
    clip_path: str
    clip_type: str
    start_sec: float
    end_sec: float
    pad_before: float
    pad_after: float
    segment_uids: list[str]
    segments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def segment_to_dict(seg: ActionSegment) -> dict[str, Any]:
    return {'uid': seg.uid, 'video_id': seg.video_id, 'participant_id': seg.participant_id, 'narration': seg.narration, 'verb': seg.verb, 'verb_class': seg.verb_class, 'noun': seg.noun, 'noun_class': seg.noun_class, 'start_frame': seg.start_frame, 'stop_frame': seg.stop_frame, 'start_timestamp': seg.start_timestamp, 'stop_timestamp': seg.stop_timestamp, 'start_sec': round(seg.start_sec, 3), 'stop_sec': round(seg.stop_sec, 3), 'label': humanize_action(seg.verb, seg.noun)}

def propose_clips_for_video(*, video_id: str, video_path: Path, segments: list[ActionSegment], clips_dir: Path, pad_before: float, pad_after: float, max_span_sec: float=16.0, min_segments: int=2, max_segments: int=6, max_clips: int | None=None) -> list[ClipSpec]:
    """Propose task-agnostic local clips by packing consecutive actions."""
    specs: list[ClipSpec] = []
    idx = 0
    while idx < len(segments):
        if max_clips is not None and len(specs) >= max_clips:
            break
        end = idx
        while end + 1 < len(segments) and end + 1 - idx + 1 <= max_segments and is_local_window(segments[idx:end + 2], max_span_sec):
            end += 1
        window = segments[idx:end + 1]
        if len(window) >= min_segments and is_local_window(window, max_span_sec):
            specs.append(_make_spec(video_id=video_id, video_path=video_path, clips_dir=clips_dir, window=window, sample_idx=len(specs) + 1, pad_before=pad_before, pad_after=pad_after, duration_limit=max_span_sec + pad_before + pad_after))
            idx = end + 1
        else:
            idx += 1
    specs = _make_non_overlapping(specs)
    if video_path.is_file():
        duration = _probe_video_duration(video_path)
        for spec in specs:
            spec.end_sec = round(min(spec.end_sec, duration), 3)
    return specs if max_clips is None else specs[:max_clips]

def _make_non_overlapping(specs: list[ClipSpec]) -> list[ClipSpec]:
    """Trim only shared padding while preserving every retained action span."""
    retained: list[ClipSpec] = []
    for spec in sorted(specs, key=lambda row: (row.start_sec, row.end_sec, row.clip_id)):
        if not retained:
            retained.append(spec)
            continue
        previous = retained[-1]
        if previous.end_sec <= spec.start_sec:
            retained.append(spec)
            continue
        previous_action_end = max((float(segment['stop_sec']) for segment in previous.segments))
        current_action_start = min((float(segment['start_sec']) for segment in spec.segments))
        if current_action_start < previous_action_end:
            continue
        boundary = (previous_action_end + current_action_start) / 2.0
        previous.end_sec = round(min(previous.end_sec, boundary), 3)
        spec.start_sec = round(max(spec.start_sec, boundary), 3)
        retained.append(spec)
    return retained

def _probe_video_duration(video_path: Path) -> float:
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)], check=True, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f'Invalid source video duration: {video_path}')
    return duration

def _make_spec(*, video_id: str, video_path: Path, clips_dir: Path, window: list[ActionSegment], sample_idx: int, pad_before: float, pad_after: float, duration_limit: float) -> ClipSpec:
    ordered_window = sorted(window, key=lambda seg: (seg.start_frame, seg.start_sec, seg.uid))
    if not ordered_window:
        raise ValueError('Cannot build clip spec for empty window')
    start_seg = ordered_window[0]
    end_seg = ordered_window[-1]
    if start_seg.start_frame > end_seg.stop_frame:
        raise AssertionError(f'clip frame order invalid for {video_id}_local_{sample_idx:04d}: {start_seg.uid} start_frame={start_seg.start_frame} > {end_seg.uid} stop_frame={end_seg.stop_frame}')
    start_sec = max(0.0, start_seg.start_sec - pad_before)
    end_sec = end_seg.stop_sec + pad_after
    if end_sec - start_sec > duration_limit:
        end_sec = start_sec + duration_limit
    clip_id = f'{video_id}_local_{sample_idx:04d}_{start_seg.uid}_{end_seg.uid}'
    clip_path = clips_dir / video_id / f'{clip_id}.mp4'
    return ClipSpec(clip_id=clip_id, video_id=video_id, source_video_path=str(video_path), clip_path=str(clip_path), clip_type='local_window', start_sec=round(start_sec, 3), end_sec=round(end_sec, 3), pad_before=pad_before, pad_after=pad_after, segment_uids=[seg.uid for seg in ordered_window], segments=[segment_to_dict(seg) for seg in ordered_window])

def materialize_clip(spec: ClipSpec, *, overwrite: bool=False) -> ClipSpec:
    out_path = Path(spec.clip_path)
    if out_path.exists() and out_path.stat().st_size > 0 and (not overwrite):
        return spec
    extract_clip_range(video_path=Path(spec.source_video_path), start_sec=spec.start_sec, end_sec=spec.end_sec, out_path=out_path)
    return spec

def slice_epic_dataset(*, data_root: Path, clips_dir: Path, manifest_path: Path, pad_before: float=2.0, pad_after: float=3.0, max_span_sec: float=16.0, min_segments: int=2, max_segments: int=6, max_videos: int | None=None, max_clips_per_video: int | None=None, video_ids: list[str] | None=None, overwrite: bool=False) -> dict[str, Any]:
    train_csv, _info_csv, _videos_root = resolve_epic_paths(data_root)
    videos = list_available_videos(data_root, max_videos=max_videos, video_ids=video_ids)
    if not videos:
        raise RuntimeError(f'No usable EPIC videos found under {data_root}')
    all_specs: list[ClipSpec] = []
    video_summaries: list[dict[str, Any]] = []
    for video_id, video_path in videos:
        segments = load_segments(train_csv, video_id)
        specs = propose_clips_for_video(video_id=video_id, video_path=video_path, segments=segments, clips_dir=clips_dir, pad_before=pad_before, pad_after=pad_after, max_span_sec=max_span_sec, min_segments=min_segments, max_segments=max_segments, max_clips=max_clips_per_video)
        for spec in specs:
            materialize_clip(spec, overwrite=overwrite)
        all_specs.extend(specs)
        video_summaries.append({'video_id': video_id, 'source_video_path': str(video_path), 'num_segments': len(segments), 'num_clips': len(specs), 'clip_type_counts': _counts(specs)})
    write_manifest(all_specs, manifest_path)
    return {'data_root': str(data_root), 'clips_dir': str(clips_dir), 'manifest': str(manifest_path), 'num_videos': len(videos), 'num_clips': len(all_specs), 'clip_type_counts': _counts(all_specs), 'slice_params': {'max_span_sec': max_span_sec, 'min_segments': min_segments, 'max_segments': max_segments, 'pad_before': pad_before, 'pad_after': pad_after}, 'videos': video_summaries}

def write_manifest(specs: Iterable[ClipSpec], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w') as f:
        for spec in specs:
            f.write(json.dumps(spec.to_dict(), ensure_ascii=False) + '\n')
            count += 1
    return count

def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def _counts(specs: Iterable[ClipSpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in specs:
        out[spec.clip_type] = out.get(spec.clip_type, 0) + 1
    return out
