from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def load_visibility_evidence(visibility_root: Path | None, scene_id: str) -> tuple[dict[str, dict[str, Any]], Path | None]:
    if visibility_root is None:
        return ({}, None)
    path = _find_visibility_json(visibility_root, scene_id)
    if path is None:
        return ({}, None)
    data = json.loads(path.read_text())
    if data.get('visibility_reliable') is not True:
        return ({}, path)
    frame_ids = _build_frame_ids(data)
    evidence = {}
    for obj in data.get('objects', []):
        frames = obj.get('visible_frames', [])
        if not frames:
            continue
        best = max(frames, key=lambda row: row.get('bbox_area_ratio', 0.0))
        frame_name = best.get('frame_name')
        sampled_frames = _sample_visible_frames(frames, frame_ids)
        evidence[obj['object_key']] = {'source': str(path), 'object_key': obj['object_key'], 'label': obj.get('label'), 'frame_name': frame_name, 'frame_id': frame_ids.get(frame_name), 'timestamp': best.get('timestamp'), 'bbox_2d': best.get('bbox_2d'), 'bbox_area_ratio': best.get('bbox_area_ratio'), 'center_2d': best.get('center_2d'), 'num_visible_frames': obj.get('num_visible_frames'), 'first_visible_frame': obj.get('first_visible_frame'), 'last_visible_frame': obj.get('last_visible_frame'), 'visible_frame_samples': sampled_frames}
    return (evidence, path)

def _sample_visible_frames(frames: list[dict[str, Any]], frame_ids: dict[str, int], max_samples: int=3) -> list[dict[str, Any]]:
    if not frames:
        return []
    sorted_frames = sorted(frames, key=lambda row: row.get('timestamp', 0.0))
    best = max(sorted_frames, key=lambda row: row.get('bbox_area_ratio', 0.0))
    candidates = [sorted_frames[0], best, sorted_frames[-1]]
    samples = []
    seen = set()
    for row in candidates:
        frame_name = row.get('frame_name')
        if not frame_name or frame_name in seen:
            continue
        seen.add(frame_name)
        samples.append({'frame_name': frame_name, 'frame_id': frame_ids.get(frame_name), 'timestamp': row.get('timestamp'), 'bbox_area_ratio': row.get('bbox_area_ratio'), 'center_2d': row.get('center_2d')})
        if len(samples) >= max_samples:
            break
    return samples

def _find_visibility_json(root: Path, scene_id: str) -> Path | None:
    if root.is_file():
        return root
    patterns = [f'visibility_{scene_id}_w2c_row_filtered_v2.json', f'visibility_{scene_id}_w2c_row_filtered.json', f'visibility_{scene_id}*.json']
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None

def _build_frame_ids(data: dict[str, Any]) -> dict[str, int]:
    rows = []
    for obj in data.get('objects', []):
        for frame in obj.get('visible_frames', []):
            frame_name = frame.get('frame_name')
            timestamp = frame.get('timestamp')
            if frame_name is not None and timestamp is not None:
                rows.append((float(timestamp), frame_name))
    frame_ids = {}
    for idx, (_, frame_name) in enumerate(sorted(set(rows)), start=1):
        frame_ids[frame_name] = idx
    return frame_ids
