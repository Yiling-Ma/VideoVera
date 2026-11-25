"""Final data-construction component: formats/canonical.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import copy
from typing import Any
VERSION = 'video_rl_structured_gold_canonical_v3'
ABOUT_FACE_THRESHOLD_DEG = 134.5
PROMPT_MARKER = '\n\nReturn exactly one standard JSON object.'
TARGET_TASKS = {'egocentric_temporal_direction', 'route_shape'}
HUMAN_OVERRIDES = {'etd_0646_E9uDoFAP3SH_b78cb336f3_end': [{'kind': 'forward'}, {'kind': 'turn', 'direction': 'left', 'source_direction': 'left'}, {'kind': 'turn', 'direction': 'back', 'source_direction': 'left'}]}

def policy_text_part(row: dict[str, Any]) -> dict[str, Any]:
    parts = [part for message in row['messages'] for part in message['content'] if part.get('type') == 'text']
    if len(parts) != 1:
        raise ValueError(f"{row['id']}: expected exactly one policy text part")
    return parts[0]

def source_action_specs_from_gold(gold: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in gold['reasoning_steps']:
        if step['claim_type'] != 'action':
            continue
        claim = step['structured_claim']
        if claim['move'] == 'forward' and claim['turn'] == 'none':
            actions.append({'kind': 'forward'})
        elif claim['move'] == 'stationary' and claim['turn'] in {'left', 'right', 'around_left', 'around_right'}:
            direction = 'left' if 'left' in claim['turn'] else 'right'
            actions.append({'kind': 'turn', 'direction': direction, 'source_direction': direction, 'angle_deg': abs(float(claim.get('heading_change_deg_clockwise') or 0.0))})
    return actions

def canonical_etd_actions(row_id: str, source_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if row_id in HUMAN_OVERRIDES:
        return copy.deepcopy(HUMAN_OVERRIDES[row_id])
    result: list[dict[str, Any]] = []
    kept_forward = False
    for source in source_actions:
        if source['kind'] == 'forward':
            if not kept_forward:
                result.append({'kind': 'forward'})
                kept_forward = True
            continue
        direction = source['direction']
        angle = source.get('angle_deg')
        if direction == 'back' or (angle is not None and angle >= ABOUT_FACE_THRESHOLD_DEG):
            direction = 'back'
        result.append({'kind': 'turn', 'direction': direction, 'source_direction': source.get('source_direction', source.get('direction', 'left'))})
    if not kept_forward:
        raise ValueError(f'{row_id}: ETD source contains no forward motion')
    return result

def direction_phrase(direction: str) -> str:
    return {'front': 'ahead', 'front-right': 'ahead and to the right', 'right': 'to the right', 'back-right': 'behind and to the right', 'back': 'behind', 'back-left': 'behind and to the left', 'left': 'to the left', 'front-left': 'ahead and to the left', 'same_location': 'at approximately the same horizontal location'}[direction]

def action_label(action: dict[str, Any]) -> str:
    if action['kind'] == 'forward':
        return 'GO_STRAIGHT'
    return {'left': 'TURN_LEFT', 'right': 'TURN_RIGHT', 'back': 'TURN_BACK'}[action['direction']]

def action_text(action: dict[str, Any]) -> str:
    return {'GO_STRAIGHT': 'Go straight.', 'TURN_LEFT': 'Turn left.', 'TURN_RIGHT': 'Turn right.', 'TURN_BACK': 'Turn back.'}[action_label(action)]

def common_instruction(task: str, task_rules: str) -> str:
    return 'Return exactly one standard JSON object. The first character must be { and the last character must be }. Do not output Markdown, analysis, a numbered natural-language list, or any text before or after the JSON. The outer object must contain exactly schema_version, task_type, steps, and final_answer. schema_version must be "video_rl_trace_v1" and task_type must be "' + task + '". final_answer must be exactly {"option":"<chosen option letter>"}; replace the angle-bracket description with one letter that exists in the question options. step_id must start at 1 and be consecutive. Never add an unrequested claim. ' + task_rules
CANONICAL_INSTRUCTIONS = {'egocentric_temporal_direction': common_instruction('egocentric_temporal_direction', 'First output reference_frame {reference, target}. Then summarize the visible route topology in temporal order with action claims. Keep the first forward motion as {sequence_index, move:"forward", turn:"none"}; omit later forward legs. Keep every discrete turn as {sequence_index, move:"stationary", turn}, where turn is left, right, around_left, or around_right and rotations of about 135 degrees or more are around turns. Do not output distances, angles, times, or frame indices. Finish with relative_direction {direction}.'), 'route_shape': common_instruction('route_shape', 'Output the complete ordered topological route. For every visible travel leg or turn, output route_action {phase, sequence_index, action_type, role}; phase is forward, action_type is GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, or TURN_BACK, and role is travel, path_bend, initial_orientation, or orientation_only. Preserve every GO_STRAIGHT and every turn in temporal order, but never output distances, angles, times, or frame indices. For a fixed-position rotation also output stationarity_evidence {position_fixed}. Finish with route_shape {shape}.')}

def reset_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, step in enumerate(steps, 1):
        item = copy.deepcopy(step)
        item['step_id'] = f'claim_{index:02d}'
        result.append(item)
    return result

def minimal_etd_steps(gold: dict[str, Any]) -> list[dict[str, Any]]:
    row_id = str(gold['provenance']['source_id'])
    references = [step for step in gold['reasoning_steps'] if step['claim_type'] == 'reference_frame']
    directions = [step for step in gold['reasoning_steps'] if step['claim_type'] == 'relative_direction']
    if len(references) != 1 or len(directions) != 1:
        raise ValueError(f'{row_id}: expected one reference and one relative direction')
    actions = canonical_etd_actions(row_id, source_action_specs_from_gold(gold))
    selected: list[dict[str, Any]] = []
    reference = copy.deepcopy(references[0])
    reference['structured_claim'] = {'claim_type': 'reference_frame', 'reference': reference['structured_claim']['reference'], 'target': reference['structured_claim']['target']}
    reference['text'] = f"Use the {reference['structured_claim']['reference']} view as the reference and trace to the {reference['structured_claim']['target']} position."
    reference['verifier'] = {'rule': 'reference_frame_equals', 'expected': {'reference': reference['structured_claim']['reference'], 'target': reference['structured_claim']['target']}}
    selected.append(reference)
    for sequence_index, action in enumerate(actions):
        if action['kind'] == 'forward':
            move, turn = ('forward', 'none')
        else:
            move = 'stationary'
            if action['direction'] == 'back':
                turn = 'around_left' if action.get('source_direction') == 'left' else 'around_right'
            else:
                turn = action['direction']
        selected.append({'claim_type': 'action', 'structured_claim': {'claim_type': 'action', 'sequence_index': sequence_index, 'move': move, 'turn': turn}, 'text': action_text(action), 'verifier': {'rule': 'camera_action_equals', 'expected': {'sequence_index': sequence_index, 'move': move, 'turn': turn}}})
    direction = copy.deepcopy(directions[0])
    direction['structured_claim'] = {'claim_type': 'relative_direction', 'direction': direction['structured_claim']['direction']}
    direction['text'] = f"So the target position is {direction_phrase(direction['structured_claim']['direction'])}."
    direction['verifier'] = {'rule': 'relative_direction_equals', 'expected': direction['structured_claim']['direction']}
    selected.append(direction)
    return reset_steps(selected)

def minimal_shape_steps(gold: dict[str, Any]) -> list[dict[str, Any]]:
    row_id = str(gold['provenance']['source_id'])
    selected: list[dict[str, Any]] = []
    action_index = 0
    for source in gold['reasoning_steps']:
        claim_type = source['claim_type']
        claim = source['structured_claim']
        if claim_type in {'route_action', 'orientation_action'}:
            role = claim.get('role', 'orientation_only')
            action_type = claim['action_type']
            selected.append({'claim_type': 'route_action', 'structured_claim': {'claim_type': 'route_action', 'phase': 'forward', 'sequence_index': action_index, 'action_type': action_type, 'role': role}, 'text': {'GO_STRAIGHT': 'Go straight.', 'TURN_LEFT': 'Turn left.', 'TURN_RIGHT': 'Turn right.', 'TURN_BACK': 'Turn back.'}[action_type], 'verifier': {'rule': 'pose_action_at_index_equals', 'expected': {'phase': 'forward', 'sequence_index': action_index, 'action_type': action_type, 'role': role}}})
            action_index += 1
        elif claim_type == 'stationarity_evidence':
            selected.append({'claim_type': 'stationarity_evidence', 'structured_claim': {'claim_type': 'stationarity_evidence', 'position_fixed': bool(claim['position_fixed'])}, 'text': 'My position stays fixed; only my viewing direction changes.', 'verifier': {'rule': 'maximum_displacement_within_in_place_threshold', 'expected': bool(claim['position_fixed'])}})
        elif claim_type == 'route_shape':
            selected.append({'claim_type': 'route_shape', 'structured_claim': {'claim_type': 'route_shape', 'shape': claim['shape']}, 'text': f"So the route shape is {claim['shape']}.", 'verifier': {'rule': 'route_shape_from_travel_geometry_equals', 'expected': claim['shape']}})
    if not selected or selected[-1]['claim_type'] != 'route_shape':
        raise ValueError(f'{row_id}: canonical route-shape trace lacks final shape')
    return reset_steps(selected)

def update_gold(source_gold: dict[str, Any], task: str) -> dict[str, Any]:
    gold = copy.deepcopy(source_gold)
    gold['schema_version'] = VERSION
    if task == 'egocentric_temporal_direction':
        steps = minimal_etd_steps(gold)
        definition = 'reference_plus_first_forward_plus_all_discrete_turns_plus_direction'
    elif task == 'route_shape':
        steps = minimal_shape_steps(gold)
        definition = 'complete_distance_free_topological_route_plus_shape'
    else:
        return gold
    gold['reasoning_steps'] = steps
    gold['reward_spec']['process']['required_claim_ids'] = [step['step_id'] for step in steps]
    gold['reward_spec']['process']['canonical_claim_definition'] = definition
    gold['provenance']['canonicalization'] = {'version': VERSION, 'source_grounding_preserved': True, 'continuous_geometry_removed_from_required_trace': True, 'human_override': gold['provenance']['source_id'] in HUMAN_OVERRIDES}
    return gold

def update_policy(source: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(source)
    task = str(row['task_type'])
    if task not in TARGET_TASKS:
        return row
    part = policy_text_part(row)
    if PROMPT_MARKER not in part['text']:
        raise ValueError(f"{row['id']}: cannot locate structured-trace instruction")
    question = part['text'].split(PROMPT_MARKER, 1)[0]
    part['text'] = question + '\n\n' + CANONICAL_INSTRUCTIONS[task]
    return row
