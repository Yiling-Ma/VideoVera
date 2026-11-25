"""Final data-construction component: formats/targets.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import copy
import math
import re
from pathlib import Path
from typing import Any
VERSION = 'video_rl_structured_gold_v4_constrained_decode_policy_prompt'
OPTION_RE = re.compile('^([A-Z])\\.\\s+(.*)$')
NUMBERED_RE = re.compile('^\\s*(\\d+)\\.\\s+(.*\\S)\\s*$')
FINAL_RE = re.compile('^Answer:\\s*([A-Z])\\s*$')
ACTION_TEXT = {'GO_STRAIGHT': 'Go straight.', 'TURN_LEFT': 'Turn left.', 'TURN_RIGHT': 'Turn right.', 'TURN_BACK': 'Turn back.'}
TRACE_REQUIREMENTS = {'relative_direction': 'Use separate steps for anchor_object {label}, facing_anchor {label}, target_object {label}, longitudinal_relation {relation}, lateral_relation {relation}, and relative_direction {direction}. Do not combine fields into one claim. longitudinal relation is front or back; lateral relation is left or right; direction is front-left, front-right, back-left, or back-right.', 'before_or_after_action': 'Allowed claims are event_observation {order_index, action} and temporal_relation {relation, action}. Use one event_observation per observed event, zero-based order_index, and relation before or after.', 'event_order': 'Use one separate step per observed event, in video order, with zero-based indices. The only allowed claim is ordered_event {order_index, action}. Never put arrays in order_index or action.', 'object_grounded_route_planning': 'Allowed claims are landmark {role, label}, initial_orientation {facing_label}, turn {turn_index, action_type}, travel_to_landmark {travel_index, to_label}, and turn_sequence {actions}. landmark role is start, waypoint, or destination. Indices are zero-based. action_type and actions must use uppercase fixed enums.', 'route_plan': 'Use one separate route_action per action with zero-based sequence_index. The only allowed claim is route_action {phase, sequence_index, action_type}. The literal value of phase is always "forward", including for TURN_LEFT, TURN_RIGHT, and TURN_BACK. action_type must be GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, or TURN_BACK. Do not add distance, turns, direction, shape, description, or any other field.', 'reverse_route_plan': 'Use one separate route_action per action with a zero-based index within its phase. Allowed claims are route_action {phase, sequence_index, action_type} and reverse_plan {actions}. phase is only forward or reverse, never turn/left/right. action_type and actions may only contain GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, or TURN_BACK. Do not add any other fields.', 'route_shape': 'Allowed claims are route_action {phase, sequence_index, action_type, role}, stationarity_evidence {position_fixed}, and route_shape {shape}. The literal phase is always "forward". route_action role must be exactly travel, path_bend, initial_orientation, or orientation_only. shape must be Manhattan, straight line, L-shape, U-shape, or in-place orientation. Do not use first/last as roles and do not add any other fields.', 'egocentric_temporal_direction': 'Each fact must be its own step. Allowed claims are reference_frame {reference, target}, action {sequence_index, move, turn}, and relative_direction {direction}. reference and target are start or end; move is forward or stationary; turn is none, left, right, around_left, or around_right. Use zero-based action indices and do not nest one claim inside another. direction must be one of front, front-right, right, back-right, back, back-left, left, front-left, or same_location.'}

def trace_instruction(task: str) -> str:
    return 'Return exactly one standard JSON object. The first character must be { and the last character must be }. Do not output Markdown, analysis, a numbered natural-language list, or any text before or after the JSON. The outer object must contain exactly schema_version, task_type, steps, and final_answer. schema_version must be "video_rl_trace_v1" and task_type must be "' + task + '". final_answer must be exactly {"option":"<chosen option letter>"}; replace the angle-bracket description with one letter that exists in the question options. step_id must start at 1 and be consecutive. Use only the fields and categorical values in the legal step shapes below. Never add another field. ' + TRACE_REQUIREMENTS[task]

def prompt_text(row: dict[str, Any]) -> str:
    texts = [part['text'] for message in row['messages'] if message['role'] == 'user' for part in message['content'] if part.get('type') == 'text']
    if len(texts) != 1:
        raise ValueError(f"{row['id']}: expected one user text part")
    return texts[0]

def parse_prompt(row: dict[str, Any]) -> tuple[str, dict[str, str]]:
    text = prompt_text(row)
    if 'Question:' not in text or '\n\nOptions:\n' not in text:
        raise ValueError(f"{row['id']}: malformed question/options prompt")
    question = text.split('Question:', 1)[1].split('\n\nOptions:\n', 1)[0].strip()
    option_text = text.split('\n\nOptions:\n', 1)[1]
    choices: dict[str, str] = {}
    for line in option_text.splitlines():
        match = OPTION_RE.match(line)
        if match:
            choices[match.group(1)] = match.group(2).strip()
        elif choices:
            break
    if list(choices) != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]) or len(choices) < 2:
        raise ValueError(f"{row['id']}: malformed options {choices}")
    return (question, choices)

def assistant_text(row: dict[str, Any]) -> str:
    texts = [part['text'] for message in row['messages'] if message['role'] == 'assistant' for part in message['content'] if part.get('type') == 'text']
    if len(texts) != 1:
        raise ValueError(f"{row['id']}: expected one assistant text part")
    return texts[0]

def assistant_steps(row: dict[str, Any]) -> list[str]:
    result = []
    final = None
    for line in assistant_text(row).splitlines():
        numbered = NUMBERED_RE.match(line)
        if numbered:
            if int(numbered.group(1)) != len(result) + 1:
                raise ValueError(f"{row['id']}: nonconsecutive reasoning numbering")
            result.append(numbered.group(2))
            continue
        match = FINAL_RE.match(line)
        if match:
            final = match.group(1)
        elif line.strip():
            raise ValueError(f"{row['id']}: unexpected assistant line {line!r}")
    if final != row['answer_key']:
        raise ValueError(f"{row['id']}: final line does not match answer_key")
    if not result:
        raise ValueError(f"{row['id']}: empty reasoning")
    return result

def segment(value: dict[str, Any]) -> dict[str, Any]:
    keys = ('segment_uid', 'video_id', 'label', 'narration', 'verb', 'noun', 'start_frame', 'stop_frame', 'start_sec', 'stop_sec')
    return {key: copy.deepcopy(value[key]) for key in keys if key in value}

def action(value: dict[str, Any]) -> dict[str, Any]:
    keys = ('action_type', 'role', 'action_index', 'heading_before_deg', 'heading_after_deg', 'heading_change_deg', 'video_frame_start', 'video_frame_end', 'video_start_sec', 'video_end_sec', 'distance_m', 'from_waypoint_index', 'to_waypoint_index', 'support_frames', 'retraces_frames', 'pose_verified')
    return {key: copy.deepcopy(value[key]) for key in keys if key in value}

def object_ref(value: dict[str, Any]) -> dict[str, Any]:
    keys = ('object_key', 'uid', 'label', 'centroid', 'axes_lengths_m')
    return {key: copy.deepcopy(value[key]) for key in keys if key in value}

def make_step(index: int, claim_type: str, text: str, structured_claim: dict[str, Any], rule: str, expected: Any) -> dict[str, Any]:
    return {'step_id': f'claim_{index:02d}', 'claim_type': claim_type, 'text': text, 'structured_claim': structured_claim, 'verifier': {'rule': rule, 'expected': expected}}

def attach_texts(row: dict[str, Any], specs: list[tuple[str, dict[str, Any], str, Any]]) -> list[dict[str, Any]]:
    texts = assistant_steps(row)
    if len(texts) != len(specs):
        raise ValueError(f"{row['id']}: {len(texts)} text steps != {len(specs)} structured claims")
    return [make_step(index, claim_type, text, structured, rule, expected) for index, (text, (claim_type, structured, rule, expected)) in enumerate(zip(texts, specs), 1)]

def reward_spec(task: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {'version': 'video_rl_reward_v1', 'format': {'rule': 'strict_json_and_task_schema', 'weight': 0.05, 'invalid_reward': 0.0}, 'outcome': {'rule': 'exact_current_option_key', 'weight': 0.45, 'invalid_or_missing_reward': 0.0}, 'process': {'rule': 'required_structured_claim_slots', 'weight': 0.5, 'required_claim_ids': [step['step_id'] for step in steps], 'missing_claim_reward': 0.0, 'free_text_similarity_used': False, 'llm_judge_used_as_authority': False, 'claim_weight_policy': 'video_rl_reward_v1_normalized_by_expected_weight'}, 'task': task}

def finalize_gold(row: dict[str, Any], source_path: Path, steps: list[dict[str, Any]], grounding: dict[str, Any], derived_answer: str) -> dict[str, Any]:
    question, choices = parse_prompt(row)
    if choices.get(row['answer_key']) != row['answer']:
        raise ValueError(f"{row['id']}: current answer key/value mismatch")
    if derived_answer != row['answer']:
        raise ValueError(f"{row['id']}: source-derived answer {derived_answer!r} != current {row['answer']!r}")
    return {'schema_version': VERSION, 'question': question, 'choices': choices, 'answer': {'key': row['answer_key'], 'value': row['answer']}, 'reasoning_steps': steps, 'grounding': grounding, 'reward_spec': reward_spec(str(row['task_type']), steps), 'provenance': {'source_id': row['id'], 'source_jsonl': str(source_path), 'authority': 'deterministic_geometry_pose_or_temporal_annotation'}}

def relative_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    g = raw['metadata']['grounding']
    observer, facing, target = (g[name] for name in ('observer_object', 'facing_object', 'target_object'))
    geometry = g['geometry']
    direction = geometry['quadrant']
    longitudinal, lateral = direction.split('-')
    specs = [('anchor_object', {'role': 'observer', **object_ref(observer)}, 'object_key_equals', observer['object_key']), ('facing_anchor', {'role': 'facing', **object_ref(facing)}, 'object_key_equals', facing['object_key']), ('target_object', {'role': 'target', **object_ref(target)}, 'object_key_equals', target['object_key']), ('reference_frame', {'observer_object_key': observer['object_key'], 'facing_object_key': facing['object_key'], 'forward_vector_xy': geometry['forward_vector'][:2]}, 'reference_frame_equals', {'observer': observer['object_key'], 'facing': facing['object_key']}), ('longitudinal_relation', {'relation': longitudinal, 'forward_component_m': geometry['forward_component']}, 'signed_forward_component', longitudinal), ('lateral_relation', {'relation': lateral, 'right_component_m': geometry['right_component']}, 'signed_right_component', lateral), ('relative_direction', {'direction': direction}, 'quadrant_equals', direction)]
    steps = attach_texts(row, specs)
    return finalize_gold(row, source, steps, {'coordinate_basis': g['note'], 'observer_object': object_ref(observer), 'facing_object': object_ref(facing), 'target_object': object_ref(target), 'geometry': copy.deepcopy(geometry), 'min_component_margin': g['min_component_margin'], 'target_extent_check': copy.deepcopy(g.get('target_extent_check'))}, direction)

def temporal_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    g = raw['grounding']
    task = row['task_type']
    specs: list[tuple[str, dict[str, Any], str, Any]] = []
    if task == 'before_or_after_action':
        anchor = segment(g['anchor_segment'])
        adjacent = segment(g['adjacent_segment'])
        query = g['query_direction']
        ordered = [adjacent, anchor] if query == 'before' else [anchor, adjacent]
        if not ordered[0]['start_sec'] < ordered[1]['start_sec']:
            raise ValueError(f"{row['id']}: temporal source order is not strictly increasing")
        for index, event in enumerate(ordered):
            specs.append(('event_observation', {'order_index': index, 'event': event}, 'segment_uid_equals', event['segment_uid']))
        specs.extend([('temporal_relation', {'anchor_segment_uid': anchor['segment_uid'], 'answer_segment_uid': adjacent['segment_uid'], 'relation': query}, 'immediate_temporal_neighbor_equals', {'relation': query, 'segment_uid': adjacent['segment_uid']}), ('answer', {'answer_key': row['answer_key'], 'action': adjacent['label']}, 'answer_key_and_value_equals', {'key': row['answer_key'], 'value': adjacent['label']})])
        derived = adjacent['label']
        grounding = {'time_basis': 'source_video_seconds', 'query_direction': query, 'direction': g['direction'], 'anchor_segment': anchor, 'adjacent_segment': adjacent}
    else:
        ordered = [segment(item) for item in g['ordered_segments']]
        if len(ordered) != 4 or any((ordered[index]['start_sec'] >= ordered[index + 1]['start_sec'] for index in range(len(ordered) - 1))):
            raise ValueError(f"{row['id']}: invalid four-event time order")
        for index, event in enumerate(ordered):
            specs.append(('ordered_event', {'order_index': index, 'event': event}, 'segment_uid_at_order_index_equals', {'order_index': index, 'segment_uid': event['segment_uid']}))
        labels = [event['label'] for event in ordered]
        derived = ', '.join(labels)
        specs.append(('answer', {'answer_key': row['answer_key'], 'ordered_actions': labels}, 'answer_key_and_order_equals', {'key': row['answer_key'], 'ordered_actions': labels}))
        grounding = {'time_basis': 'source_video_seconds', 'ordered_segments': ordered}
    return finalize_gold(row, source, attach_texts(row, specs), grounding, derived)

def object_route_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    metadata = raw['metadata']
    keys = metadata['route_object_keys']
    labels = metadata['route_labels']
    turns = metadata['turn_sequence']
    angles = metadata['signed_turn_angles_deg']
    if not (len(keys) == len(labels) == 3 and len(turns) == len(angles) == 2):
        raise ValueError(f"{row['id']}: expected three landmarks and two turns")
    normalized = [turn.upper().replace(' ', '_') for turn in turns]
    specs = [('landmark', {'role': 'start', 'object_key': keys[0], 'label': labels[0]}, 'object_key_equals', keys[0]), ('initial_orientation', {'facing_object_key': metadata['facing_object_key'], 'facing_label': metadata['facing_label']}, 'facing_object_key_equals', metadata['facing_object_key']), ('turn', {'turn_index': 0, 'action_type': normalized[0], 'signed_angle_deg': angles[0]}, 'turn_action_equals', normalized[0]), ('travel_to_landmark', {'from_object_key': keys[0], 'to_object_key': keys[1], 'to_label': labels[1]}, 'destination_object_key_equals', keys[1]), ('turn', {'turn_index': 1, 'action_type': normalized[1], 'signed_angle_deg': angles[1]}, 'turn_action_equals', normalized[1]), ('travel_to_landmark', {'from_object_key': keys[1], 'to_object_key': keys[2], 'to_label': labels[2]}, 'destination_object_key_equals', keys[2]), ('turn_sequence', {'actions': normalized, 'answer_key': row['answer_key']}, 'ordered_turns_equal', normalized)]
    derived = ', '.join(turns)
    return finalize_gold(row, source, attach_texts(row, specs), {'route_object_keys': copy.deepcopy(keys), 'route_labels': copy.deepcopy(labels), 'facing_object_key': metadata['facing_object_key'], 'facing_label': metadata['facing_label'], 'turn_sequence': normalized, 'signed_turn_angles_deg': copy.deepcopy(angles), 'ground_truth_source': metadata['gt_source']}, derived)

def pose_grounding(pose_route: dict[str, Any] | None) -> dict[str, Any]:
    if pose_route is None:
        return {'pose_route': None}
    return {'pose_route': {'version': pose_route['version'], 'pose_signature': pose_route['pose_signature'], 'video_sha256': pose_route['video_sha256'], 'policy': copy.deepcopy(pose_route['policy']), 'start_heading_deg': pose_route['start_heading_deg'], 'end_heading_deg': pose_route['end_heading_deg'], 'canonical_actions': [action(item) for item in pose_route['canonical_actions']], 'reverse_actions': [action(item) for item in pose_route['reverse_actions']], 'forward_actions': copy.deepcopy(pose_route['forward_actions']), 'reverse_action_sequence': copy.deepcopy(pose_route['reverse_action_sequence'])}}

def turn_answer(sequence: list[str]) -> str:
    return ' '.join((ACTION_TEXT[item] for item in sequence if item != 'GO_STRAIGHT'))

def action_spec(claim_type: str, phase: str, index: int, value: dict[str, Any]) -> tuple[str, dict[str, Any], str, Any]:
    normalized = action(value)
    return (claim_type, {'phase': phase, 'sequence_index': index, **normalized}, 'pose_action_at_index_equals', {'phase': phase, 'sequence_index': index, 'action_type': normalized['action_type']})

def route_plan_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    pose = raw['grounding']['pose_route']
    actions = pose['canonical_actions']
    sequence = pose['forward_actions']
    if [item['action_type'] for item in actions] != sequence:
        raise ValueError(f"{row['id']}: canonical/forward action mismatch")
    specs = [action_spec('route_action', 'forward', index, item) for index, item in enumerate(actions)]
    turns = [item for item in sequence if item != 'GO_STRAIGHT']
    specs.append(('answer', {'answer_key': row['answer_key'], 'missing_turns': turns}, 'answer_key_and_turns_equal', {'key': row['answer_key'], 'turns': turns}))
    return finalize_gold(row, source, attach_texts(row, specs), pose_grounding(pose), turn_answer(sequence))

def reverse_route_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    pose = raw['grounding']['pose_route']
    forward_actions = pose['canonical_actions']
    reverse_actions = pose['reverse_actions']
    forward_sequence = pose['forward_actions']
    reverse_sequence = pose['reverse_action_sequence']
    if [item['action_type'] for item in forward_actions] != forward_sequence:
        raise ValueError(f"{row['id']}: invalid forward sequence")
    if [item['action_type'] for item in reverse_actions] != reverse_sequence:
        raise ValueError(f"{row['id']}: invalid reverse sequence")
    specs = [action_spec('route_action', 'forward', index, item) for index, item in enumerate(forward_actions)]
    specs.append(('reverse_plan', {'actions': reverse_sequence}, 'reverse_sequence_equals', reverse_sequence))
    specs.extend((action_spec('route_action', 'reverse', index, item) for index, item in enumerate(reverse_actions)))
    specs.append(('answer', {'answer_key': row['answer_key'], 'actions': reverse_sequence}, 'answer_key_and_actions_equal', {'key': row['answer_key'], 'actions': reverse_sequence}))
    derived = ' '.join((ACTION_TEXT[item] for item in reverse_sequence))
    return finalize_gold(row, source, attach_texts(row, specs), pose_grounding(pose), derived)

def route_shape_from_pose(pose: dict[str, Any] | None) -> str:
    if pose is None:
        return 'in-place orientation'
    actions = pose['canonical_actions']
    if not any((item['action_type'] == 'GO_STRAIGHT' for item in actions)):
        return 'in-place orientation'
    bends = [item['action_type'] for item in actions if item.get('role') == 'path_bend']
    if not bends:
        return 'straight line'
    if len(bends) == 1:
        return 'L-shape'
    if len(bends) == 2 and bends[0] == bends[1]:
        return 'U-shape'
    return 'Manhattan'

def route_shape_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    pose = raw['grounding']['pose_route']
    derived = route_shape_from_pose(pose)
    specs: list[tuple[str, dict[str, Any], str, Any]] = []
    if pose is None:
        source_actions = [step for step in raw['gold_trace'] if step['claim_type'] == 'route_action']
        for index, item in enumerate(source_actions):
            claim = item['claim'].lower()
            if 'left' in claim:
                action_type = 'TURN_LEFT'
            elif 'right' in claim:
                action_type = 'TURN_RIGHT'
            elif 'back' in claim or 'around' in claim:
                action_type = 'TURN_BACK'
            else:
                raise ValueError(f"{row['id']}: cannot normalize in-place action {item['claim']!r}")
            specs.append(('orientation_action', {'sequence_index': index, 'action_type': action_type, 'role': 'orientation_only', 'evidence': copy.deepcopy(item.get('evidence', {}))}, 'orientation_action_equals', action_type))
        shape_evidence = next((step for step in raw['gold_trace'] if step['claim_type'] == 'route_shape'))
        specs.append(('stationarity_evidence', {'position_fixed': True, 'evidence': copy.deepcopy(shape_evidence.get('evidence', {}))}, 'maximum_displacement_within_in_place_threshold', True))
    else:
        sequence = pose['forward_actions']
        if [item['action_type'] for item in pose['canonical_actions']] != sequence:
            raise ValueError(f"{row['id']}: route-shape action mismatch")
        specs.extend((action_spec('route_action', 'forward', index, item) for index, item in enumerate(pose['canonical_actions'])))
    path_bends = [] if pose is None else [item['action_type'] for item in pose['canonical_actions'] if item.get('role') == 'path_bend']
    specs.append(('route_shape', {'shape': derived, 'path_bends': path_bends, 'initial_orientation_ignored': True}, 'route_shape_from_travel_geometry_equals', derived))
    grounding = pose_grounding(pose)
    if pose is None:
        grounding['in_place_evidence'] = copy.deepcopy([step.get('evidence', {}) for step in raw['gold_trace'] if step['claim_type'] != 'answer'])
    return finalize_gold(row, source, attach_texts(row, specs), grounding, derived)

def classify_direction(front: float, right: float, distance: float, same_threshold: float) -> str:
    if distance <= same_threshold:
        return 'same_location'
    angle = math.degrees(math.atan2(right, front))
    index = int(math.floor((angle + 22.5) / 45.0)) % 8
    return ('front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left')[index]

def etd_gold(row: dict[str, Any], raw: dict[str, Any], source: Path) -> dict[str, Any]:
    g = raw['grounding']
    derived = classify_direction(float(g['front_m']), float(g['right_m']), float(g['horizontal_distance_m']), float(g['policy']['same_location_m']))
    specs = []
    for raw_step in raw['gold_reasoning_steps']:
        claim = copy.deepcopy(raw_step['structured_claim'])
        claim_type = raw_step['claim_type']
        if claim_type == 'answer':
            claim['answer_key'] = row['answer_key']
            claim['direction'] = row['answer']
            expected: Any = {'key': row['answer_key'], 'direction': row['answer']}
            rule = 'answer_key_and_direction_equal'
        elif claim_type == 'action':
            expected = {'move': claim['move'], 'turn': claim['turn']}
            rule = 'camera_action_equals'
        elif claim_type == 'relative_direction':
            expected = claim['direction']
            rule = 'relative_direction_equals'
        elif claim_type == 'reference_frame':
            expected = {'reference': claim['reference'], 'target': claim['target']}
            rule = 'reference_frame_equals'
        else:
            raise ValueError(f"{row['id']}: unknown ETD claim {claim_type}")
        specs.append((claim_type, claim, rule, expected))
    grounding = copy.deepcopy(g)
    grounding['time_basis'] = 'clip_seconds'
    return finalize_gold(row, source, attach_texts(row, specs), grounding, derived)

def policy_row(row: dict[str, Any]) -> dict[str, Any]:
    user_messages = [copy.deepcopy(message) for message in row['messages'] if message['role'] == 'user']
    if len(user_messages) != 1:
        raise ValueError(f"{row['id']}: policy row needs exactly one user message")
    text_parts = [item for item in user_messages[0]['content'] if item.get('type') == 'text']
    if len(text_parts) != 1:
        raise ValueError(f"{row['id']}: policy row needs exactly one text part")
    text = text_parts[0]['text']
    for marker in ('\n\nThink step by step', '\n\nEnd with exactly one final line'):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    text_parts[0]['text'] = text + '\n\n' + trace_instruction(str(row['task_type']))
    return {'id': row['id'], 'task_type': row['task_type'], 'video_path': row['video_path'], 'messages': user_messages}

def hidden_target(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    return {'id': row['id'], 'task_type': row['task_type'], 'structured_gold': copy.deepcopy(gold)}

def no_policy_leak(rows: list[dict[str, Any]]) -> None:
    forbidden = {'answer', 'answer_key', 'assistant', 'structured_gold', 'grounding', 'verifier', 'gold_reasoning'}

    def visit(value: Any, row_id: str) -> None:
        if isinstance(value, dict):
            overlap = forbidden & set(value)
            if overlap:
                raise ValueError(f'{row_id}: policy leak keys {sorted(overlap)}')
            for child in value.values():
                visit(child, row_id)
        elif isinstance(value, list):
            for child in value:
                visit(child, row_id)
    for row in rows:
        if any((message.get('role') != 'user' for message in row['messages'])):
            raise ValueError(f"{row['id']}: non-user policy message")
        visit(row, str(row['id']))
