"""Final data-construction component: formats/raw.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from typing import Any
from videovera.formats.route_claims import canonical_gold_trace_claim_type as egocentric_claim_type
from videovera.formats.spatial_claims import canonical_gold_trace_claim_type as spatial_claim_type
from videovera.formats.temporal_claims import canonical_gold_trace_claim_type as temporal_claim_type
from videovera.formats.steps import build_step_reward_labels, extract_gold_reasoning_steps, resolve_video_path

def build_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    mapper = claim_type_mapper(record)
    gold_steps = extract_gold_reasoning_steps(record, claim_type_mapper=mapper)
    if not gold_steps:
        raise ValueError('missing_gold_reasoning')
    compact_steps = [compact_step(step) for step in gold_steps]
    reward_labels = build_step_reward_labels(gold_steps)
    row = {'id': record.get('id'), 'instance_id': record.get('id'), 'scene_id': record.get('scene_id'), 'video_id': record.get('video_id'), 'task_type': record.get('task_type'), 'dataset_source': record.get('dataset_source'), 'video_path': resolve_video_path(record), 'question': record.get('question'), 'choices': record.get('choices'), 'answer_key': record.get('answer_key'), 'answer': record.get('answer'), 'gold_reasoning': [step['text'] for step in compact_steps], 'gold_reasoning_steps': compact_steps, 'gold_reasoning_visual_enriched': [], 'gold_reasoning_visual_enriched_steps': [], 'qwen_reasoning': [], 'qwen_reasoning_steps': [], 'step_reward_labels': reward_labels, 'reward_data': {'format': 'video_prm_sft_gold_reference_v1', 'target_reasoning_steps': compact_steps, 'step_reward_labels': reward_labels, 'candidate_reasoning_steps': [], 'candidate_answer_key': None, 'gold_answer_key': record.get('answer_key'), 'gold_answer': record.get('answer'), 'reward_note': 'Gold-only SFT reference; candidate/model rewards are produced in the later RL/evaluation stage.'}, 'metadata': public_metadata(record)}
    validate_sft_row(row)
    return row

def claim_type_mapper(record: dict[str, Any]):
    source = str(record.get('dataset_source') or '')
    task_type = str(record.get('task_type') or '')
    if source == 'r2r' or task_type in {'route_plan', 'reverse_route_plan', 'route_shape'}:
        return egocentric_claim_type
    if source == 'arkitscenes' or task_type == 'relative_direction':
        return spatial_claim_type
    if source == 'epic_kitchens_100' or task_type in {'event_order', 'before_or_after_action'}:
        return temporal_claim_type
    return lambda claim_type: claim_type or 'other'

def compact_step(step: dict[str, Any]) -> dict[str, Any]:
    view = {'step_id': step['step_id'], 'claim_type': step['claim_type'], 'text': step['text']}
    for key in ('structured_claim', 'comparison_targets', 'evidence_summary', 'verifier_rule', 'expected'):
        value = step.get(key)
        if value not in (None, '', [], {}):
            view[key] = value
    return view

def public_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get('metadata') or {})
    metadata.update({'sft_source': 'nonoverlap_raw_qa_pool', 'pool_name': record.get('_pool_name'), 'pool_path': record.get('_pool_path'), 'source_key': source_key(record), 'evaluation_valid': True, 'outcome': 'gold_sft_reference'})
    for key in ('difficulty', 'quality_score', 'quality_notes', 'grounding', 'gold_trace', 'annotation_path'):
        value = record.get(key)
        if value not in (None, '', [], {}):
            metadata[key] = value
    return metadata

def validate_sft_row(row: dict[str, Any]) -> None:
    for field in ('id', 'video_path', 'question', 'choices', 'answer_key', 'answer', 'gold_reasoning_steps'):
        if row.get(field) in (None, '', [], {}):
            raise ValueError(f'missing_{field}')
    choices = row.get('choices')
    if not isinstance(choices, dict):
        raise ValueError('choices_not_dict')
    if str(row['answer_key']) not in choices:
        raise ValueError('answer_key_not_in_choices')

def source_key(row: dict[str, Any]) -> str:
    task_type = str(row.get('task_type') or '')
    source = str(row.get('dataset_source') or row.get('_pool_name') or 'unknown')
    if task_type in {'event_order', 'before_or_after_action'}:
        identifier = str((row.get('grounding') or {}).get('clip_id') or row.get('id') or '')
    else:
        identifier = str(row.get('video_id') or row.get('scene_id') or '')
    return f'{source}:{identifier}' if identifier else ''
