from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

def write_jsonl(records: Iterable[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open('w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            count += 1
    return count

def make_step(step_id: int, claim_type: str, claim: str, evidence: dict[str, Any], verifier_rule: str, expected: Any, *, model_facing_step: str | None=None, verifier_step: str | None=None) -> dict[str, Any]:
    return {'step_id': step_id, 'claim_type': claim_type, 'claim': claim, 'model_facing_step': model_facing_step if model_facing_step is not None else claim, 'verifier_step': verifier_step if verifier_step is not None else claim, 'evidence': evidence, 'verifier': {'rule': verifier_rule, 'expected': expected}}

def make_record(*, record_id: str, video_id: str, annotation_path: Path, task_type: str, question: str, choices: dict[str, str], answer_key: str, answer: str, gold_trace: list[dict[str, Any]], grounding: dict[str, Any], quality_score: float, quality_notes: list[str], difficulty: str='medium') -> dict[str, Any]:
    record = {'id': record_id, 'dataset_source': 'epic_kitchens_100', 'scene_id': video_id, 'video_id': video_id, 'annotation_path': str(annotation_path), 'task_type': task_type, 'difficulty': difficulty, 'question': question, 'choices': choices, 'answer_key': answer_key, 'answer': answer, 'gold_trace': gold_trace, 'verifier_reasoning': gold_trace, 'gold_reasoning': [step['claim'] for step in gold_trace], 'grounding': grounding, 'quality_score': round(float(quality_score), 4), 'quality_notes': quality_notes}
    clip_path = grounding.get('clip_path')
    if clip_path:
        record['video_path'] = str(clip_path)
    return record
