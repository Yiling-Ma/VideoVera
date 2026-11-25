from __future__ import annotations
import re
from typing import Any, Callable
ClaimTypeMapper = Callable[[str], str]

def identity_claim_type(claim_type: str) -> str:
    return claim_type or 'other'

def extract_gold_reasoning_steps(record: dict[str, Any], *, claim_type_mapper: ClaimTypeMapper=identity_claim_type) -> list[dict[str, Any]]:
    alignment = record.get('reasoning_alignment') or {}
    top_level_steps = normalize_reasoning_steps(record.get('gold_reasoning_steps'), source='gold_reasoning_steps', claim_type_mapper=claim_type_mapper)
    if top_level_steps:
        return top_level_steps
    for key in ('annotation_reasoning', 'gold_reasoning_generation'):
        value = record.get(key)
        if not isinstance(value, dict) or value.get('status') != 'ok':
            continue
        normalized = normalize_reasoning_steps(value.get('reasoning_steps'), source=key, claim_type_mapper=claim_type_mapper)
        if normalized:
            return normalized
    gold_reasoning = record.get('gold_reasoning')
    if isinstance(gold_reasoning, list):
        normalized = []
        claim_types = [claim_type_mapper(str(step.get('claim_type') or '')) for step in record.get('gold_trace') or [] if isinstance(step, dict)]
        if not claim_types:
            claim_types = [str(step.get('claim_type') or 'other') for step in alignment.get('gold_claims') or [] if isinstance(step, dict)]
        for idx, text in enumerate(gold_reasoning, start=1):
            text_value = str(text or '').strip()
            if not text_value:
                continue
            claim_type = claim_types[idx - 1] if idx - 1 < len(claim_types) else 'other'
            normalized.append({'step_id': idx, 'claim_type': claim_type_mapper(claim_type), 'text': text_value, 'source': 'gold_reasoning', **step_metadata_from_alignment(alignment, idx)})
        if normalized:
            return normalized
    gold_trace = record.get('gold_trace')
    if isinstance(gold_trace, list):
        normalized = []
        for idx, step in enumerate(gold_trace, start=1):
            if not isinstance(step, dict):
                continue
            text = str(step.get('model_facing_step') or step.get('claim') or '').strip()
            if not text:
                continue
            normalized.append({'step_id': int(step.get('step_id') or idx), 'claim_type': claim_type_mapper(str(step.get('claim_type') or '')), 'text': text, 'source': 'gold_trace', **normalize_step_metadata(step)})
        if normalized:
            return normalized
    gold_claims = alignment.get('gold_claims')
    if isinstance(gold_claims, list):
        normalized = []
        for idx, step in enumerate(gold_claims, start=1):
            if not isinstance(step, dict):
                continue
            text = str(step.get('gold_text') or '').strip()
            if not text:
                continue
            normalized.append({'step_id': int(step.get('step_id') or idx), 'claim_type': str(step.get('claim_type') or 'other'), 'text': text, 'source': 'reasoning_alignment.gold_claims', **normalize_alignment_step_metadata(step)})
        if normalized:
            return normalized
    return []

def normalize_reasoning_steps(steps: Any, *, source: str='annotation_reasoning', claim_type_mapper: ClaimTypeMapper=identity_claim_type) -> list[dict[str, Any]]:
    if not isinstance(steps, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        text = clean_training_text(step.get('text'))
        if not text:
            continue
        normalized.append({'step_id': int(step.get('step_id') or idx), 'claim_type': claim_type_mapper(str(step.get('claim_type') or 'other')), 'text': text, 'source': source, **normalize_step_metadata(step)})
    return normalized

def step_metadata_from_alignment(alignment: dict[str, Any], index: int) -> dict[str, Any]:
    gold_claims = alignment.get('gold_claims') or []
    if not isinstance(gold_claims, list) or index - 1 >= len(gold_claims):
        return {}
    step = gold_claims[index - 1]
    if not isinstance(step, dict):
        return {}
    return normalize_alignment_step_metadata(step)

def normalize_alignment_step_metadata(step: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source_key, target_key in (('structured_claim', 'structured_claim'), ('claim', 'structured_claim'), ('comparison_targets', 'comparison_targets'), ('evidence_summary', 'evidence_summary'), ('verifier_step', 'verifier_step'), ('verifier_rule', 'verifier_rule'), ('expected', 'expected'), ('source_claim_type', 'source_claim_type')):
        value = step.get(source_key)
        if value is not None:
            metadata[target_key] = value
    if 'structured_claim' not in metadata:
        synthesized = synthesize_structured_claim(step)
        if synthesized:
            metadata['structured_claim'] = synthesized
    return metadata

def normalize_step_metadata(step: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source_key, target_key in (('claim', 'structured_claim'), ('structured_claim', 'structured_claim'), ('comparison_targets', 'comparison_targets'), ('evidence_summary', 'evidence_summary'), ('verifier_step', 'verifier_step'), ('verifier_rule', 'verifier_rule'), ('expected', 'expected'), ('source_claim_type', 'source_claim_type')):
        value = step.get(source_key)
        if value is not None:
            metadata[target_key] = value
    verifier = step.get('verifier')
    if isinstance(verifier, dict):
        if verifier.get('rule') is not None and 'verifier_rule' not in metadata:
            metadata['verifier_rule'] = verifier.get('rule')
        if verifier.get('expected') is not None and 'expected' not in metadata:
            metadata['expected'] = verifier.get('expected')
    if 'structured_claim' not in metadata:
        synthesized = synthesize_structured_claim(step)
        if synthesized:
            metadata['structured_claim'] = synthesized
    return metadata

def synthesize_structured_claim(step: dict[str, Any]) -> dict[str, Any]:
    comparison_targets = step.get('comparison_targets')
    evidence_summary = step.get('evidence_summary')
    synthesized: dict[str, Any] = {}
    if isinstance(comparison_targets, dict):
        synthesized.update({key: value for key, value in comparison_targets.items() if value not in (None, '', [], {})})
    if isinstance(evidence_summary, dict):
        for key in ('descriptor', 'frame_ids', 'frame_names', 'supporting_visible_frames', 'screen_region', 'category', 'count', 'labels', 'object_keys'):
            value = evidence_summary.get(key)
            if value not in (None, '', [], {}) and key not in synthesized:
                synthesized[key] = value
    return synthesized

def build_step_reward_labels(gold_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [reward_for_step(step) for step in gold_steps]

def reward_for_step(step: dict[str, Any]) -> dict[str, Any]:
    label = {'step_id': step.get('step_id'), 'claim_type': step.get('claim_type') or 'other', 'reward': 1, 'label': 1, 'scorable': True, 'reward_basis': 'gold_verified_step'}
    for source_key, target_key in (('structured_claim', 'structured_claim'), ('comparison_targets', 'comparison_targets'), ('evidence_summary', 'evidence_summary'), ('verifier_step', 'verifier_step'), ('verifier_rule', 'verifier_rule'), ('expected', 'expected'), ('source_claim_type', 'source_claim_type')):
        value = step.get(source_key)
        if value not in (None, '', [], {}):
            label[target_key] = value
    return label

def clean_training_text(text: Any) -> str:
    cleaned = ' '.join(str(text or '').split())
    if not cleaned:
        return ''
    cleaned = re.sub(',?\\s*visible\\s+(?:around|at)\\s+frame\\s+\\d+\\b', '', cleaned, flags=re.I)
    cleaned = re.sub(',?\\s*seen\\s+(?:around|at)\\s+frame\\s+\\d+\\b', '', cleaned, flags=re.I)
    cleaned = re.sub('\\bframe\\s+\\d+\\b', '', cleaned, flags=re.I)
    replacements = [('\\bdishwasheras\\b', 'dishwasher as'), ('\\bwasheras\\b', 'washer as'), ('\\bbedas\\b', 'bed as'), ('\\bsofaas\\b', 'sofa as'), ('\\bcabinetas\\b', 'cabinet as'), ('\\btableas\\b', 'table as'), ('\\bandface\\b', 'and face'), ('\\btargetobject\\b', 'target object'), ('\\bfacingfrom\\b', 'facing from'), ('\\binfront\\b', 'in front'), ('\\binthe\\b', 'in the'), ('\\banswer([A-D])\\b', 'answer \\1'), ('\\bwhichis\\b', 'which is'), ('\\bcorrespondsto\\b', 'corresponds to'), ('\\bfalls on the right side\\b', 'is on the right side'), ('\\bfalls on your right side\\b', 'is on your right side'), ('\\bfalls on my right side\\b', 'is on my right side'), ('\\bfalls on the left side\\b', 'is on the left side'), ('\\bfalls on your left side\\b', 'is on your left side'), ('\\bfalls on my left side\\b', 'is on my left side'), ('\\bto appear\\b', 'seen in the scan'), ('\\bon the right side of the frame\\b', 'on the right side'), ('\\bon the left side of the frame\\b', 'on the left side'), ('\\btoward the top of the frame\\b', 'toward the top'), ('\\btoward the bottom of the frame\\b', 'toward the bottom')]
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
    cleaned = re.sub('\\s+\\)', ')', cleaned)
    cleaned = re.sub('\\(\\s+', '(', cleaned)
    cleaned = re.sub(',\\s*,', ', ', cleaned)
    cleaned = re.sub(',([A-Za-z])', ', \\1', cleaned)
    cleaned = re.sub('\\.([A-Za-z])', '. \\1', cleaned)
    cleaned = re.sub('\\s+([,.;:])', '\\1', cleaned)
    cleaned = re.sub('\\s{2,}', ' ', cleaned)
    return cleaned.strip()

def resolve_video_path(record: dict[str, Any]) -> str | None:
    video_judge = record.get('video_judge') or {}
    for value in (video_judge.get('video_path'), record.get('video_path')):
        if value:
            return str(value)
    return None
