"""Final data-construction component: data/navigation/questions.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import hashlib
import json
import re
import random
import itertools
from data_generator.data.navigation.actions import UncertainRoute, extract_route
WORDS = {'GO_STRAIGHT': 'go straight', 'TURN_LEFT': 'turn left', 'TURN_RIGHT': 'turn right', 'TURN_BACK': 'turn back'}

def action_text(actions):
    return ' '.join((WORDS[a].capitalize() + '.' for a in actions))

def reverse_option_universe():
    return [action_text(['TURN_BACK', 'GO_STRAIGHT'] + list(itertools.chain.from_iterable(((turn, 'GO_STRAIGHT') for turn in turns)))) for count in (1, 2) for turns in itertools.product(('TURN_LEFT', 'TURN_RIGHT'), repeat=count)]

def choices_for(identity, actions, task_type=None):
    correct = action_text(actions)
    turns = [i for i, action in enumerate(actions) if action.startswith('TURN')]
    if not turns:
        raise UncertainRoute('insufficient_distinct_options')
    rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest(), 16))
    if task_type == 'reverse_route_plan' and correct in reverse_option_universe():
        candidates = [s for s in reverse_option_universe() if s != correct]
        options = [correct] + rng.sample(candidates, 3)
    elif len(turns) == 1:
        options = []
        for turn in ('TURN_LEFT', 'TURN_RIGHT', 'TURN_BACK'):
            candidate = list(actions)
            candidate[turns[0]] = turn
            options.append(action_text(candidate))
    else:
        rng.shuffle(turns)
        groups = [turns[::2], turns[1::2]]
        alternatives = {i: rng.choice([t for t in WORDS if t.startswith('TURN') and t != actions[i]]) for i in turns}
        options = []
        for mask in range(4):
            candidate = list(actions)
            for bit, group in enumerate(groups):
                if mask & 1 << bit:
                    for i in group:
                        candidate[i] = alternatives[i]
            options.append(action_text(candidate))
    distractors = [text for text in options if text != correct]
    rng.shuffle(distractors)
    option_count = len(options)
    index = int(hashlib.sha256(identity.encode()).hexdigest(), 16) % option_count
    distractors.insert(index, correct)
    labels = 'ABCD'[:option_count]
    choices = dict(zip(labels, distractors))
    return (choices, labels[index])

def choices_from_training_pool(identity, actions, frequencies):
    """Sample same-length negatives from observed training answer frequencies.

    The pool is training-only, never evaluation labels. Unlike single-edit
    negatives it is not centered on the gold; unlike uniform combinatorial
    negatives it does not flood the choices with rarely correct sequences.
    """
    if any((not a.startswith('TURN') for a in actions)):
        raise ValueError('Empirical route choices require turns-only answers')
    correct = action_text(actions)
    if frequencies.get(correct, 0) <= 0:
        raise ValueError('Gold absent from training answer pool')
    if len(actions) == 1:
        return choices_for(identity, actions, 'route_plan')
    candidates = sorted((text for text, count in frequencies.items() if count > 0 and text != correct))
    for text in candidates:
        turns = [part.strip() for part in text.split('.') if part.strip()]
        if len(turns) != len(actions) or any((turn not in ('Turn left', 'Turn right', 'Turn back') for turn in turns)):
            raise ValueError('Training answer pool mixes lengths or invalid turns')
    if len(candidates) < 3:
        raise UncertainRoute('insufficient_empirical_turn_sequences')
    rng = random.Random(identity + ':empirical-training-v1')
    negatives = []
    for _ in range(3):
        chosen = rng.choices(candidates, weights=[frequencies[s] for s in candidates], k=1)[0]
        negatives.append(chosen)
        candidates.remove(chosen)
    index = int(hashlib.sha256(identity.encode()).hexdigest(), 16) % 4
    negatives.insert(index, correct)
    return (dict(zip('ABCD', negatives)), 'ABCD'[index])

def step(claim, claim_type, expected, evidence):
    return dict(claim_type=claim_type, claim=claim, model_facing_step=claim, verifier_step=claim, evidence=evidence, verifier={'rule': 'measured_pose_route', 'expected': expected})

def landmark_labels(payload, route):
    metadata = payload.get('route_plan_landmarks') or {}
    if metadata.get('pose_signature') != route['pose_signature']:
        raise UncertainRoute('landmarks_need_pose_boundary_reanchoring')
    anchors = {a.get('waypoint_index'): a for a in metadata.get('anchors', [])}
    labels = []
    for boundary in route['boundaries']:
        anchor = anchors.get(boundary['waypoint_index'], {})
        if anchor.get('canonical_video_frame') != boundary['frame_index']:
            raise UncertainRoute('landmark_frame_mismatch')
        landmark = anchor.get('selected_landmark') or {}
        label = ' '.join(str(landmark.get('label') or '').split())
        if not landmark.get('usable_for_route_plan') or not label:
            raise UncertainRoute('missing_unique_visual_landmark')
        labels.append(label)
    cores = [re.split('\\s+(?:beside|next to|near|under|above|behind|with|by|on)\\s+', label.lower())[0] for label in labels]
    if len(set(cores)) != len(cores):
        raise UncertainRoute('repeated_landmark_identity')
    return labels

def build_record(payload, video_path, annotation_path, scene_id, video_id, task_type, record_id=None):
    route = extract_route(payload, video_path)
    identity = record_id or f'{video_id}_{task_type}_001'
    trace = []
    if task_type == 'reverse_route_plan':
        question = 'From my viewpoint at the end of the video, how can I go back to my starting point?'
        actions = route['reverse_action_sequence']
        for i, event in enumerate(route['canonical_actions']):
            prefix = 'From the starting viewpoint, first' if i == 0 else 'Then'
            trace.append(step(f"{prefix}, {WORDS[event['action_type']]}.", 'forward', event['action_type'], event))
        trace.append(step('From the final viewpoint, face along the last travelled segment toward the start, then retrace the route.', 'reverse', actions, {'end_heading_deg': route['end_heading_deg'], 'legs': route['legs']}))
        for i, event in enumerate(route['reverse_actions']):
            trace.append(step(f"{('First' if i == 0 else 'Then')}, {WORDS[event['action_type']]}.", 'reverse', event['action_type'], event))
    elif task_type == 'route_plan':
        labels = landmark_labels(payload, route)
        events = route['canonical_actions']
        events = [e for e in events if e['role'] != 'terminal_orientation']
        actions = [e['action_type'] for e in events if e['action_type'].startswith('TURN')]
        if not actions:
            raise UncertainRoute('route_plan_has_no_turn_blank')
        parts = []
        for i, event in enumerate(events, 1):
            if event['action_type'] == 'GO_STRAIGHT':
                text = f"go forward toward the {labels[event['to_waypoint_index']]}"
                parts.append(f'{i}. {text.capitalize()}.')
            else:
                text = WORDS[event['action_type']]
                parts.append(f'{i}. [?]')
            trace.append(step(f"{('First' if i == 1 else 'Then')}, {text}.", 'forward', event['action_type'], event))
        question = f"You are a robot beginning with the {labels[0]} in view. Navigate to the {labels[-1]} by following the route shown in the video. Fill in the missing turns in order: {' '.join(parts)}"
    else:
        raise ValueError(f'Unsupported pose route task: {task_type}')
    choices, key = choices_for(identity, actions, task_type)
    trace.append(step(f'Therefore, the correct answer is choice {key}: {choices[key]}', 'answer', key, {'actions': actions, 'choices': choices}))
    for i, item in enumerate(trace, 1):
        item['step_id'] = i
    return dict(id=identity, dataset_source='r2r', scene_id=scene_id, video_id=video_id, video_path=str(video_path), annotation_path=str(annotation_path), task_type=task_type, question=question, choices=choices, answer_key=key, answer=choices[key], gold_reasoning=[s['claim'] for s in trace], gold_trace=trace, verifier_reasoning=trace, gold_reasoning_original=[s['claim'] for s in trace], grounding={'pose_route': route, 'video_path': str(video_path), 'annotation_path': str(annotation_path)}, metadata={'choice_policy': 'shared_prefix_return_paths_v3' if task_type == 'reverse_route_plan' and choices[key] in reverse_option_universe() else 'symmetric_hamming_rectangle_v2', 'rebuild_status': 'rebuilt_pending_visual_review', 'human_review_required': True, 'evaluation_valid': False, 'pose_signature': route['pose_signature']}, quality_score=1.0, quality_notes=['Measured-pose checks passed; visual review pending.'], difficulty='medium', llm_rewrite={'status': 'skipped', 'reason': 'atomic pose-derived steps'})

def try_build_records(payload, video_path, annotation_path, scene_id, video_id):
    records, exclusions = ([], [])
    for task in ('route_plan', 'reverse_route_plan'):
        try:
            records.append(build_record(payload, video_path, annotation_path, scene_id, video_id, task))
        except (UncertainRoute, FileNotFoundError) as exc:
            exclusions.append(dict(video_id=video_id, task_type=task, reason=str(exc)))
    return (records, exclusions)
