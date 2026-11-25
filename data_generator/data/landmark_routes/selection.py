"""Final data-construction component: data/landmark_routes/selection.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
TASK = 'object_grounded_route_planning'

def validate_pool_row(row: dict[str, Any], benchmark_scenes: set[str]) -> None:
    metadata = row.get('metadata', {})
    quality = metadata.get('video_quality', {})
    if row.get('task_type') != TASK:
        raise ValueError(f"{row.get('id')}: wrong task")
    if row.get('scene_id') in benchmark_scenes:
        raise ValueError(f"{row['id']}: benchmark scene overlap")
    route_count = len(metadata.get('route_object_keys', []))
    if route_count not in (3, 4):
        raise ValueError(f"{row['id']}: expected three or four route landmarks")
    if len(metadata.get('turn_sequence', [])) != route_count - 1:
        raise ValueError(f"{row['id']}: turn count does not match route length")
    if not quality.get('valid'):
        raise ValueError(f"{row['id']}: continuous-clip quality is invalid")
    if quality.get('continuous_clip') is not True:
        raise ValueError(f"{row['id']}: media is not marked as one continuous clip")
    if quality.get('discrete_evidence_frame_insertion') is not False:
        raise ValueError(f"{row['id']}: discrete evidence frames were inserted")
    if int(quality.get('rendered_frame_count', 0)) <= 24:
        raise ValueError(f"{row['id']}: continuous video does not retain source frames")
    sampled_indices = quality.get('sampled_source_frame_indices', [])
    if len(sampled_indices) != 24 or sampled_indices != sorted(sampled_indices):
        raise ValueError(f"{row['id']}: invalid uniform 24-frame model view")
    sampled = quality.get('sampled_anchor_visibility', [])
    if len(sampled) != route_count + 1 or any((int(item.get('sampled_visible_frame_count', 0)) < 1 for item in sampled)):
        raise ValueError(f"{row['id']}: not every named object survives model sampling")
    source_visibility = metadata.get('anchor_visibility', [])
    if len(source_visibility) != route_count + 1 or any((int(item.get('visible_frame_count', 0)) < 3 for item in source_visibility)):
        raise ValueError(f"{row['id']}: insufficient strict source visibility")
    if not Path(row['video_path']).is_file():
        raise FileNotFoundError(f"{row['id']}: missing video {row['video_path']}")

def move_answer_to_key(raw: dict[str, Any], target: str) -> dict[str, Any]:
    row = copy.deepcopy(raw)
    current = row['answer_key']
    if current != target:
        row['choices'][current], row['choices'][target] = (row['choices'][target], row['choices'][current])
        row['answer_key'] = target
    if row['choices'][target] != row['answer']:
        raise ValueError(f"{row['id']}: answer mapping broke during rebalance")
    return row

def full_row(raw: dict[str, Any]) -> dict[str, Any]:
    options = '\n'.join((f"{key}. {raw['choices'][key]}" for key in sorted(raw['choices'])))
    user_text = f"Infer the ordered navigation actions using the specified object landmarks in the video.\n\nQuestion: {raw['question']}\n\nOptions:\n{options}\n\nThink step by step using only visible evidence from the video.\nEnd with exactly one final line in this format: Answer: <option letter>"
    assistant_text = '\n'.join([f'{index}. {text}' for index, text in enumerate(raw['gold_reasoning'], 1)] + [f"Answer: {raw['answer_key']}"])
    metadata = copy.deepcopy(raw['metadata'])
    metadata.update({'source_id': raw['id'], 'source_video_id': raw['video_id'], 'source_scene_id': raw['scene_id'], 'source_dataset': raw['dataset_source'], 'source_evaluation_valid': True, 'source_human_review_required': True, 'source_human_review_status': 'pipeline_requires_final_human_review', 'source_rebuild_status': 'continuous_clip_uniform_24_frame_audit_passed', 'canonical_v4_object_route': {'version': 'object_grounded_route_contiguous_final_v4', 'only_changed_task': TASK, 'model_frame_budget': 24, 'minimum_strict_source_frames_per_named_object': 3, 'minimum_uniform_model_samples_per_named_object': 1, 'single_continuous_source_interval': True, 'discrete_frame_insertion': False, 'other_task_types_frozen': True}})
    return {'answer': raw['answer'], 'answer_key': raw['answer_key'], 'dataset_source': raw['dataset_source'], 'id': raw['id'], 'messages': [{'role': 'user', 'content': [{'type': 'video', 'video': raw['video_path'], 'nframes': 24, 'max_pixels': 151200}, {'type': 'text', 'text': user_text}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_text}]}], 'metadata': metadata, 'sample_weight': 1.0, 'task_type': TASK, 'video_path': raw['video_path']}
