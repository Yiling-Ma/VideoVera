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

def make_step(step_id: int, claim_type: str, claim: str, evidence: dict[str, Any], verifier_rule: str, expected: Any) -> dict[str, Any]:
    return {'step_id': step_id, 'claim_type': claim_type, 'claim': claim, 'evidence': evidence, 'verifier': {'rule': verifier_rule, 'expected': expected}}

def make_aligned_step(step_id: int, claim_type: str, model_facing_step: str, verifier_step: str, evidence: dict[str, Any], verifier_rule: str, expected: Any) -> dict[str, Any]:
    return {'step_id': step_id, 'claim_type': claim_type, 'claim': model_facing_step, 'model_facing_step': model_facing_step, 'verifier_step': verifier_step, 'evidence': evidence, 'verifier': {'rule': verifier_rule, 'expected': expected}}

def make_record(*, record_id: str, scene_id: str, annotation_path: Path, task_type: str, question: str, choices: dict[str, str] | None, answer_key: str | None, answer: str, gold_trace: list[dict[str, Any]], gold_reasoning: list[str], grounding: dict[str, Any], video_path: Path | None=None, difficulty: str='medium') -> dict[str, Any]:
    return {'id': record_id, 'dataset_source': 'arkitscenes', 'scene_id': scene_id, 'video_id': scene_id, 'video_path': str(video_path) if video_path is not None else None, 'annotation_path': str(annotation_path), 'task_type': task_type, 'difficulty': difficulty, 'question': question, 'choices': choices, 'answer_key': answer_key, 'answer': answer, 'gold_trace': gold_trace, 'verifier_reasoning': gold_trace, 'gold_reasoning': gold_reasoning, 'grounding': grounding}
