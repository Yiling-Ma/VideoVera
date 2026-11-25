import copy
import sys
from pathlib import Path

import pytest

from videovera.data.navigation.actions import extract_route
from tests.test_pose_actions import trajectory
from videovera.data.routes.generate import (
    SHAPES, REVERSE, crop_payload, options_for, shape_for, stationary_ranges,
    validate_stationary, replace_answer,
)


@pytest.mark.parametrize('actions,label', [
    ([('go', 3), ('turn', 90)], 'straight line'),
    ([('go', 3), ('turn', 90), ('go', 3)], 'L-shape'),
    ([('go', 3), ('turn', 90), ('go', 3), ('turn', 90), ('go', 3)], 'U-shape'),
    ([('go', 3), ('turn', 90), ('go', 3), ('turn', -90), ('go', 3)], 'manhattan-style piecewise linear'),
])
def test_shape_uses_path_bends_not_terminal_looks(actions, label):
    assert shape_for(extract_route(trajectory(actions))) == label


def test_two_same_sign_turns_do_not_always_mean_u():
    route = extract_route(trajectory([('go', 3), ('turn', 45), ('go', 3), ('turn', 45), ('go', 3)]))
    with pytest.raises(ValueError, match='ambiguous'):
        shape_for(route)


def test_stationary_clip_uses_actual_position_and_rotation():
    payload = trajectory([('go', 3), ('turn', 90), ('go', 3)])
    original = copy.deepcopy(payload)
    ranges = list(stationary_ranges(payload['camera_trajectory']['frames']))
    assert ranges
    child = crop_payload(payload, *ranges[0])
    assert validate_stationary(child) == pytest.approx(90)
    assert child['camera_trajectory']['frames'][0]['timestamp_sec'] == 0
    assert payload == original
    with pytest.raises(ValueError, match='not_in_place'):
        validate_stationary(payload)


def test_mixed_in_place_turns_are_not_described_as_one_action():
    with pytest.raises(ValueError, match='ambiguous_in_place'):
        validate_stationary(trajectory([('turn', 90), ('turn', -180)]))


def test_every_reverse_option_shares_noninformative_turn_back_prefix():
    for i, answer in enumerate(REVERSE):
        c = {'task': 'reverse_route_plan', 'id': str(i), 'answer': answer, 'letter': 'ABCD'[i%4]}
        choices = options_for(c, 123)
        assert len(set(choices.values())) == 4
        assert choices[c['letter']] == answer
        assert all(s.startswith('Turn back.') for s in choices.values())


def test_replace_answer_updates_every_model_facing_trace():
    row = {'gold_trace': [{'claim': 'Go straight.'}, {'claim': 'Answer is B.'}]}
    choices = {'A': 'straight line', 'B': 'L-shape'}
    replace_answer(row, choices, 'A')
    assert row['answer'] == 'straight line'
    assert row['gold_trace'][-1]['verifier']['expected'] == 'A'
    assert row['gold_reasoning'][-1].endswith('choice A: straight line')
    assert row['gold_reasoning_original'] == row['gold_reasoning']
