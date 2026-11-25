"""Strict task-discriminated JSON Schemas for video RL traces."""
from __future__ import annotations
import json
import copy
from typing import Any
from jsonschema import Draft7Validator
VERSION = 'video_rl_trace_v1'
ACTIONS = ['GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'TURN_BACK']
DIRECTIONS = ['front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left', 'same_location']
SHAPES = ['Manhattan', 'straight line', 'L-shape', 'U-shape', 'in-place orientation']

def claim_schema(claim_type: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {'type': 'object', 'additionalProperties': False, 'required': ['step_id', 'claim_type', 'claim'], 'properties': {'step_id': {'type': 'integer', 'minimum': 1}, 'claim_type': {'const': claim_type}, 'claim': {'type': 'object', 'additionalProperties': False, 'required': required, 'properties': properties}}}
STRING = {'type': 'string', 'minLength': 1, 'maxLength': 160}
INDEX = {'type': 'integer', 'minimum': 0, 'maximum': 32}
ACTION = {'enum': ACTIONS}
STEP_SCHEMAS: dict[str, list[dict[str, Any]]] = {'relative_direction': [claim_schema(name, {'label': STRING}, ['label']) for name in ('anchor_object', 'facing_anchor', 'target_object')] + [claim_schema('longitudinal_relation', {'relation': {'enum': ['front', 'back']}}, ['relation']), claim_schema('lateral_relation', {'relation': {'enum': ['left', 'right']}}, ['relation']), claim_schema('relative_direction', {'direction': {'enum': ['front-left', 'front-right', 'back-left', 'back-right']}}, ['direction'])], 'before_or_after_action': [claim_schema('event_observation', {'order_index': INDEX, 'action': STRING}, ['order_index', 'action']), claim_schema('temporal_relation', {'relation': {'enum': ['before', 'after']}, 'action': STRING}, ['relation', 'action'])], 'event_order': [claim_schema('ordered_event', {'order_index': INDEX, 'action': STRING}, ['order_index', 'action'])], 'object_grounded_route_planning': [claim_schema('landmark', {'role': {'enum': ['start', 'waypoint', 'destination']}, 'label': STRING}, ['role', 'label']), claim_schema('initial_orientation', {'facing_label': STRING}, ['facing_label']), claim_schema('turn', {'turn_index': INDEX, 'action_type': ACTION}, ['turn_index', 'action_type']), claim_schema('travel_to_landmark', {'travel_index': INDEX, 'to_label': STRING}, ['travel_index', 'to_label']), claim_schema('turn_sequence', {'actions': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': ACTION}}, ['actions'])], 'route_plan': [claim_schema('route_action', {'phase': {'const': 'forward'}, 'sequence_index': INDEX, 'action_type': ACTION}, ['phase', 'sequence_index', 'action_type'])], 'reverse_route_plan': [claim_schema('route_action', {'phase': {'enum': ['forward', 'reverse']}, 'sequence_index': INDEX, 'action_type': ACTION}, ['phase', 'sequence_index', 'action_type']), claim_schema('reverse_plan', {'actions': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': ACTION}}, ['actions'])], 'route_shape': [claim_schema('route_action', {'phase': {'const': 'forward'}, 'sequence_index': INDEX, 'action_type': ACTION, 'role': {'enum': ['travel', 'path_bend', 'initial_orientation', 'orientation_only']}}, ['phase', 'sequence_index', 'action_type', 'role']), claim_schema('stationarity_evidence', {'position_fixed': {'type': 'boolean'}}, ['position_fixed']), claim_schema('route_shape', {'shape': {'enum': SHAPES}}, ['shape'])], 'egocentric_temporal_direction': [claim_schema('reference_frame', {'reference': {'enum': ['start', 'end']}, 'target': {'enum': ['start', 'end']}}, ['reference', 'target']), claim_schema('action', {'sequence_index': INDEX, 'move': {'enum': ['forward', 'stationary']}, 'turn': {'enum': ['none', 'left', 'right', 'around_left', 'around_right']}}, ['sequence_index', 'move', 'turn']), claim_schema('relative_direction', {'direction': {'enum': DIRECTIONS}}, ['direction'])]}
STEP_MAX_ITEMS = {'before_or_after_action': 3, 'egocentric_temporal_direction': 9, 'event_order': 4, 'object_grounded_route_planning': 9, 'relative_direction': 6, 'reverse_route_plan': 12, 'route_plan': 9, 'route_shape': 7}

def canonical_prefix_step_schemas(task: str) -> list[dict[str, Any]] | None:
    """Return canonical slot schemas when every valid trace is a prefix.

    This keeps partial process traces legal (including an empty trace) while
    making duplicate, reordered, or surplus claims impossible at JSON-Schema
    decoding time for tasks with a deterministic slot layout.
    """
    by_type = {schema['properties']['claim_type']['const']: schema for schema in STEP_SCHEMAS[task]}

    def slot(claim_type: str, step_id: int, **constants: Any) -> dict[str, Any]:
        schema = copy.deepcopy(by_type[claim_type])
        schema['properties']['step_id'] = {'const': step_id}
        claim_properties = schema['properties']['claim']['properties']
        for key, value in constants.items():
            claim_properties[key] = {'const': value}
        return schema
    if task == 'relative_direction':
        return [slot('anchor_object', 1), slot('facing_anchor', 2), slot('target_object', 3), slot('longitudinal_relation', 4), slot('lateral_relation', 5), slot('relative_direction', 6)]
    if task == 'before_or_after_action':
        return [slot('event_observation', 1, order_index=0), slot('event_observation', 2, order_index=1), slot('temporal_relation', 3)]
    if task == 'event_order':
        return [slot('ordered_event', index + 1, order_index=index) for index in range(4)]
    if task == 'object_grounded_route_planning':
        return [slot('landmark', 1, role='start'), slot('initial_orientation', 2), slot('turn', 3, turn_index=0), slot('travel_to_landmark', 4, travel_index=0), slot('turn', 5, turn_index=1), slot('travel_to_landmark', 6, travel_index=1), slot('turn_sequence', 7)]
    if task == 'route_plan':
        return [slot('route_action', index + 1, phase='forward', sequence_index=index) for index in range(STEP_MAX_ITEMS[task])]
    return None

def schema_for(task: str) -> dict[str, Any]:
    if task not in STEP_SCHEMAS:
        raise KeyError(task)
    steps_schema: dict[str, Any] = {'type': 'array', 'minItems': 0, 'maxItems': STEP_MAX_ITEMS[task], 'items': {'oneOf': STEP_SCHEMAS[task]}}
    return {'$schema': 'http://json-schema.org/draft-07/schema#', 'title': f'{VERSION}:{task}', 'type': 'object', 'additionalProperties': False, 'required': ['schema_version', 'task_type', 'steps', 'final_answer'], 'properties': {'schema_version': {'const': VERSION}, 'task_type': {'const': task}, 'steps': steps_schema, 'final_answer': {'type': 'object', 'additionalProperties': False, 'required': ['option'], 'properties': {'option': {'type': 'string', 'pattern': '^[A-Z]$'}}}}}

def generation_schema_for(task: str) -> dict[str, Any]:
    """A decoder-side subset of the public schema that prevents surplus slots.

    Reward validation still uses the more general public schema so an isolated
    correct claim can earn its normalized partial credit. Constrained sampling
    may safely use a stricter subset because it does not reveal claim values.
    """
    return schema_for(task)
SCHEMAS = {task: schema_for(task) for task in STEP_SCHEMAS}
GENERATION_SCHEMAS = {task: generation_schema_for(task) for task in STEP_SCHEMAS}
VALIDATORS = {task: Draft7Validator(schema) for task, schema in SCHEMAS.items()}

def parse_strict_json(text: str) -> Any:

    def reject_constant(value: str) -> None:
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(text, parse_constant=reject_constant)

def validate_trace(value: Any, task: str, allowed_options: set[str] | None=None) -> list[str]:
    if task not in VALIDATORS:
        return [f'unknown_task:{task}']
    errors = [error.message for error in sorted(VALIDATORS[task].iter_errors(value), key=str)]
    if errors:
        return errors
    steps = value['steps']
    if [step['step_id'] for step in steps] != list(range(1, len(steps) + 1)):
        errors.append('step_id_not_consecutive')
    slots = []
    for step in steps:
        claim = step['claim']
        claim_type = step['claim_type']
        if 'sequence_index' in claim:
            slot = (claim_type, claim.get('phase'), claim['sequence_index'])
        elif 'order_index' in claim:
            slot = (claim_type, claim['order_index'])
        elif 'turn_index' in claim:
            slot = (claim_type, claim['turn_index'])
        elif 'travel_index' in claim:
            slot = (claim_type, claim['travel_index'])
        elif claim_type == 'landmark':
            slot = (claim_type, claim['role'])
        else:
            slot = (claim_type,)
        slots.append(slot)
    if len(slots) != len(set(slots)):
        errors.append('duplicate_claim_slot')
    if allowed_options is not None and value['final_answer']['option'] not in allowed_options:
        errors.append('final_option_not_in_prompt')
    return errors
