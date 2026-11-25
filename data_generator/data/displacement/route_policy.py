"""Final data-construction component: data/displacement/route_policy.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""

def validate_turn_limits(min_turns, max_turns):
    if not 0 <= min_turns <= max_turns <= 3:
        raise ValueError('require_0_le_min_turns_le_max_turns_le_3')

def route_policy(steps, min_turns=1, max_turns=3):
    validate_turn_limits(min_turns, max_turns)
    if any((s['structured_claim'].get('claim_type') == 'action_sequence' for s in steps)):
        raise ValueError('one_action_per_step_required')
    actions = [s['structured_claim'] for s in steps if s['structured_claim'].get('claim_type') == 'action']
    turns = {'left', 'right', 'around_left', 'around_right'}
    if any((not (a.get('move') == 'forward' and a.get('turn') == 'none' or (a.get('move') == 'stationary' and a.get('turn') in turns)) for a in actions)):
        raise ValueError('require_separate_straight_and_turn_actions')
    if not any((a['move'] == 'forward' for a in actions)):
        raise ValueError('route_requires_translation')
    turn_count, previous = (0, None)
    for action in actions:
        direction = action['turn'].replace('around_', '') if action['turn'] != 'none' else None
        if direction and (previous is None or previous['direction'] != direction or abs(previous['end_sec'] - action['start_sec']) > 1e-06):
            turn_count += 1
        previous = {'direction': direction, 'end_sec': action['end_sec']} if direction else None
    if not min_turns <= turn_count <= max_turns:
        raise ValueError(f'turn_count_out_of_range:{turn_count}')
    return {'min_turns': min_turns, 'max_turns': max_turns, 'turn_count': turn_count, 'action_count': len(actions), 'one_action_per_step': True, 'turn_count_definition': 'contiguous same-direction stationary rotation; about-face counts once'}
