"""Candidate V2 deterministic process reward for the weight-selection pilot.

This module is intentionally isolated from the production V1 reward.  The V2
reward replaces claim-type weights with an ordered, slot-aware F1 score.  It
therefore penalizes missing, surplus, incorrect, duplicated, and reordered
claims while keeping every response reward in ``[0, 1]``.
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from videovera.validation.trace import parse_strict_json, validate_trace
REWARD_VERSION = 'video_rl_reward_v2_candidate'
FORMAT_WEIGHT = 0.05

@dataclass(frozen=True)
class RewardWeights:
    answer: float
    process: float
    joint: float
    format: float = FORMAT_WEIGHT

    def __post_init__(self) -> None:
        values = (self.format, self.answer, self.process, self.joint)
        if any((not math.isfinite(value) or value < 0 for value in values)):
            raise ValueError(f'reward weights must be finite and nonnegative: {values}')
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f'reward weights must sum to 1.0: {values}')
DEFAULT_WEIGHTS = RewardWeights(answer=0.45, process=0.4, joint=0.1)

@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    valid_json_schema: bool
    format_reward: float
    outcome_reward: float
    process_reward: float
    joint_reward: float
    final_answer_correct: bool
    correct_process_weight: float
    total_process_weight: float
    correct_claims: int
    expected_claims: int
    predicted_claims: int
    process_precision: float
    process_recall: float
    process_f1: float
    trace_exact: bool
    errors: tuple[str, ...]

def normalize_label(value: str) -> str:
    return re.sub('\\s+', ' ', value.strip().lower().rstrip('.'))

def canonical_claim(claim_type: str, claim: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(claim, ensure_ascii=False))
    for key in ('label', 'action', 'to_label', 'facing_label'):
        if key in value:
            value[key] = normalize_label(str(value[key]))
    return {'claim_type': claim_type, 'claim': value}

def slot_key(item: dict[str, Any]) -> tuple[Any, ...]:
    claim_type = item['claim_type']
    claim = item['claim']
    if 'sequence_index' in claim:
        return (claim_type, claim.get('phase'), claim['sequence_index'])
    if 'order_index' in claim:
        return (claim_type, claim['order_index'])
    if 'turn_index' in claim:
        return (claim_type, claim['turn_index'])
    if 'travel_index' in claim:
        return (claim_type, claim['travel_index'])
    if claim_type == 'landmark':
        return (claim_type, claim['role'])
    return (claim_type,)

def expected_trace(structured_gold: dict[str, Any]) -> list[dict[str, Any]]:
    task = structured_gold['reward_spec']['task']
    steps = structured_gold['reasoning_steps']
    result: list[dict[str, Any]] = []
    action_index = 0
    travel_index = 0
    for step in steps:
        claim_type = step['claim_type']
        source = step['structured_claim']
        item: dict[str, Any] | None = None
        if task == 'relative_direction':
            if claim_type == 'anchor_object':
                item = {'claim_type': 'anchor_object', 'claim': {'label': source['label']}}
            elif claim_type == 'facing_anchor':
                item = {'claim_type': 'facing_anchor', 'claim': {'label': source['label']}}
            elif claim_type == 'target_object':
                item = {'claim_type': 'target_object', 'claim': {'label': source['label']}}
            elif claim_type == 'longitudinal_relation':
                item = {'claim_type': claim_type, 'claim': {'relation': source['relation']}}
            elif claim_type == 'lateral_relation':
                item = {'claim_type': claim_type, 'claim': {'relation': source['relation']}}
            elif claim_type == 'relative_direction':
                item = {'claim_type': claim_type, 'claim': {'direction': source['direction']}}
        elif task == 'before_or_after_action':
            if claim_type == 'event_observation':
                item = {'claim_type': claim_type, 'claim': {'order_index': source['order_index'], 'action': source['event']['label']}}
            elif claim_type == 'temporal_relation':
                answer_uid = source['answer_segment_uid']
                grounding = structured_gold['grounding']
                answer_label = grounding['adjacent_segment']['label']
                if answer_uid != grounding['adjacent_segment']['segment_uid']:
                    raise ValueError('Temporal answer segment mismatch')
                item = {'claim_type': claim_type, 'claim': {'relation': source['relation'], 'action': answer_label}}
        elif task == 'event_order' and claim_type == 'ordered_event':
            item = {'claim_type': claim_type, 'claim': {'order_index': source['order_index'], 'action': source['event']['label']}}
        elif task == 'object_grounded_route_planning':
            grounding = structured_gold['grounding']
            if claim_type == 'landmark':
                item = {'claim_type': claim_type, 'claim': {'role': 'start', 'label': source['label']}}
            elif claim_type == 'initial_orientation':
                item = {'claim_type': claim_type, 'claim': {'facing_label': source['facing_label']}}
            elif claim_type == 'turn':
                item = {'claim_type': claim_type, 'claim': {'turn_index': source['turn_index'], 'action_type': source['action_type']}}
            elif claim_type == 'travel_to_landmark':
                item = {'claim_type': claim_type, 'claim': {'travel_index': travel_index, 'to_label': source['to_label']}}
                travel_index += 1
            elif claim_type == 'turn_sequence':
                item = {'claim_type': claim_type, 'claim': {'actions': source['actions']}}
        elif task in {'route_plan', 'reverse_route_plan'}:
            if claim_type == 'route_action':
                item = {'claim_type': claim_type, 'claim': {'phase': source['phase'], 'sequence_index': source['sequence_index'], 'action_type': source['action_type']}}
            elif claim_type == 'reverse_plan':
                item = {'claim_type': claim_type, 'claim': {'actions': source['actions']}}
        elif task == 'route_shape':
            if claim_type in {'route_action', 'orientation_action'}:
                role = source.get('role', 'orientation_only')
                item = {'claim_type': 'route_action', 'claim': {'phase': 'forward', 'sequence_index': source.get('sequence_index', action_index), 'action_type': source['action_type'], 'role': role}}
                action_index += 1
            elif claim_type == 'stationarity_evidence':
                item = {'claim_type': claim_type, 'claim': {'position_fixed': source['position_fixed']}}
            elif claim_type == 'route_shape':
                item = {'claim_type': claim_type, 'claim': {'shape': source['shape']}}
        elif task == 'egocentric_temporal_direction':
            if claim_type == 'reference_frame':
                item = {'claim_type': claim_type, 'claim': {'reference': source['reference'], 'target': source['target']}}
            elif claim_type == 'action':
                item = {'claim_type': claim_type, 'claim': {'sequence_index': action_index, 'move': source['move'], 'turn': source['turn']}}
                action_index += 1
            elif claim_type == 'relative_direction':
                item = {'claim_type': claim_type, 'claim': {'direction': source['direction']}}
        if item is not None:
            result.append(canonical_claim(item['claim_type'], item['claim']))
    slots = [slot_key(item) for item in result]
    if not result or len(slots) != len(set(slots)):
        raise ValueError(f"Invalid expected trace for {structured_gold['provenance']['source_id']}")
    return result

def gold_completion(structured_gold: dict[str, Any]) -> str:
    steps = expected_trace(structured_gold)
    payload = {'schema_version': 'video_rl_trace_v1', 'task_type': structured_gold['reward_spec']['task'], 'steps': [{'step_id': index, 'claim_type': item['claim_type'], 'claim': item['claim']} for index, item in enumerate(steps, 1)], 'final_answer': {'option': structured_gold['answer']['key']}}
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

def invalid_breakdown(*errors: str) -> RewardBreakdown:
    return RewardBreakdown(total=0.0, valid_json_schema=False, format_reward=0.0, outcome_reward=0.0, process_reward=0.0, joint_reward=0.0, final_answer_correct=False, correct_process_weight=0.0, total_process_weight=0.0, correct_claims=0, expected_claims=0, predicted_claims=0, process_precision=0.0, process_recall=0.0, process_f1=0.0, trace_exact=False, errors=tuple(errors))

def ordered_match_count(candidate: list[dict[str, Any]], expected: list[dict[str, Any]]) -> int:
    """Return exact-claim LCS length, preserving canonical trace order.

    LCS gives credit to correct later claims after one omission, but a swapped
    pair cannot both match.  Since claim values include their typed slot index,
    matching is both position-aware and source-grounded.
    """
    previous = [0] * (len(expected) + 1)
    for predicted in candidate:
        current = [0]
        for index, wanted in enumerate(expected, 1):
            if predicted == wanted:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]

def score_completion(text: str, target: dict[str, Any], mode: str='structured', *, weights: RewardWeights=DEFAULT_WEIGHTS) -> RewardBreakdown:
    structured_gold = target.get('structured_gold', target)
    task = structured_gold['reward_spec']['task']
    choices = set(structured_gold['choices'])
    try:
        parsed = parse_strict_json(text)
    except Exception as exc:
        return invalid_breakdown(f'json_parse:{type(exc).__name__}')
    errors = validate_trace(parsed, task, choices)
    if errors:
        return invalid_breakdown(*errors)
    final_correct = parsed['final_answer']['option'] == structured_gold['answer']['key']
    if mode == 'outcome':
        value = 1.0 if final_correct else 0.0
        return RewardBreakdown(total=value, valid_json_schema=True, format_reward=0.0, outcome_reward=value, process_reward=0.0, joint_reward=0.0, final_answer_correct=final_correct, correct_process_weight=0.0, total_process_weight=0.0, correct_claims=0, expected_claims=0, predicted_claims=0, process_precision=0.0, process_recall=0.0, process_f1=0.0, trace_exact=False, errors=())
    if mode != 'structured':
        raise ValueError(f'Unknown reward mode: {mode}')
    expected = expected_trace(structured_gold)
    candidate = [canonical_claim(step['claim_type'], step['claim']) for step in parsed['steps']]
    matched = ordered_match_count(candidate, expected)
    precision = matched / len(candidate) if candidate else 0.0
    recall = matched / len(expected) if expected else 0.0
    process_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    trace_exact = candidate == expected
    outcome = weights.answer if final_correct else 0.0
    process = weights.process * process_f1
    joint = weights.joint if final_correct and trace_exact else 0.0
    total = weights.format + outcome + process + joint
    if not math.isfinite(total) or total < 0 or total > 1 + 1e-12:
        raise ArithmeticError(f'Invalid reward {total}')
    return RewardBreakdown(total=min(1.0, total), valid_json_schema=True, format_reward=weights.format, outcome_reward=outcome, process_reward=process, joint_reward=joint, final_answer_correct=final_correct, correct_process_weight=float(matched), total_process_weight=float(len(expected)), correct_claims=matched, expected_claims=len(expected), predicted_claims=len(candidate), process_precision=precision, process_recall=recall, process_f1=process_f1, trace_exact=trace_exact, errors=())

class RewardIndex:

    def __init__(self, hidden_targets: str | Path, *, weights: RewardWeights=DEFAULT_WEIGHTS):
        self.weights = weights
        self.targets: dict[str, dict[str, Any]] = {}
        with Path(hidden_targets).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row['id'])
                if row_id in self.targets:
                    raise ValueError(f'Duplicate reward target: {row_id}')
                self.targets[row_id] = row

    def score(self, row_id: str, completion: str, mode: str='structured') -> RewardBreakdown:
        return score_completion(completion, self.targets[row_id], mode=mode, weights=self.weights)

    def scalar(self, row_id: str, completion: str, mode: str='structured') -> float:
        return self.score(row_id, completion, mode=mode).total

def breakdown_dict(value: RewardBreakdown) -> dict[str, Any]:
    return asdict(value)
