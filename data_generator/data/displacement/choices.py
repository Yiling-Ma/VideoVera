"""Final data-construction component: data/displacement/choices.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import json
from data_generator.data.displacement.geometry import DIRECTIONS

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def make_choices(answer, rng, answer_counts):
    key = min('ABCD', key=lambda k: (answer_counts[k], rng.random()))
    others = rng.sample([d for d in DIRECTIONS if d != answer], 3)
    choices = {}
    for k in 'ABCD':
        choices[k] = answer if k == key else others.pop()
    return (choices, key)
