"""Final data-construction component: formats/shape_labels.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import copy
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
VERSION = 'route_shape_five_class_v4'
LETTERS = tuple('ABCDE')
LABELS = ('Manhattan', 'straight line', 'L-shape', 'U-shape', 'in-place orientation')
LABEL_ALIASES = {'manhattan': 'Manhattan', 'manhattan-style piecewise linear': 'Manhattan', 'straight line': 'straight line', 'l-shape': 'L-shape', 'u-shape': 'U-shape', 'in-place orientation': 'in-place orientation'}

def stable_hash(*parts: object) -> str:
    payload = '\x1f'.join((str(part) for part in parts)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def normalized_label(value: str) -> str:
    label = LABEL_ALIASES.get(value.strip().lower())
    if label is None:
        raise ValueError(f'Unsupported Route Shape label: {value!r}')
    return label

def assigned_gold_letters(rows: list[dict], seed: int) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalized_label(row['answer'])].append(row)
    if set(groups) != set(LABELS):
        raise ValueError(f'Expected five labels, found {sorted(groups)}')
    assignments: dict[str, str] = {}
    for class_index, label in enumerate(LABELS):
        members = sorted(groups[label], key=lambda row: stable_hash(seed, label, row['id']))
        if len(members) % len(LETTERS):
            raise ValueError(f'{label} count {len(members)} cannot be balanced over five answer positions')
        for index, row in enumerate(members):
            assignments[row['id']] = LETTERS[(index + class_index) % len(LETTERS)]
    return assignments

def option_mapping(row_id: str, gold_label: str, gold_letter: str, seed: int) -> dict[str, str]:
    distractors = [label for label in LABELS if label != gold_label]
    distractors.sort(key=lambda label: stable_hash(seed, row_id, 'distractor', label))
    mapping: dict[str, str] = {}
    iterator = iter(distractors)
    for letter in LETTERS:
        mapping[letter] = gold_label if letter == gold_letter else next(iterator)
    return mapping

def prompt_text(options: dict[str, str]) -> str:
    rendered = '\n'.join((f'{letter}. {options[letter]}' for letter in LETTERS))
    return f"Question: What's the shape of my moving trajectory?\n\nOptions:\n{rendered}\n\nEnd with exactly one final line in this format: Answer: <option letter>"

def source_reasoning_steps(source: dict, label: str) -> list[str]:
    assistant = next((message for message in source['messages'] if message['role'] == 'assistant'))
    text = next((part['text'] for part in assistant['content'] if part.get('type') == 'text'))
    steps: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith('answer:'):
            continue
        if '. ' in line and line.split('. ', 1)[0].isdigit():
            line = line.split('. ', 1)[1].strip()
        lowered = line.lower()
        if 'correct answer' in lowered or lowered.startswith('therefore'):
            continue
        if lowered.startswith('the path ') or lowered.startswith('the movement follows') or lowered.startswith('two bends ') or lowered.startswith('two straight legs '):
            continue
        steps.append(line.rstrip('.'))
    conclusion = {'Manhattan': 'The movement contains multiple straight legs connected by more than one turn, forming a Manhattan path', 'straight line': 'Ignoring changes in viewing direction, the route shape is a straight line', 'L-shape': 'Two traveled straight legs meet at one turn, forming an L-shape', 'U-shape': 'The two turns bend in the same direction, forming a U-shape', 'in-place orientation': 'The viewing direction changes while the position stays fixed, so this is in-place orientation'}[label]
    if conclusion not in steps:
        steps.append(conclusion)
    if not 2 <= len(steps) <= 7:
        raise ValueError(f"Unexpected reasoning length {len(steps)} for {source['id']}")
    return [step + '.' for step in steps]

def assistant_text(reasoning: list[str], gold_letter: str) -> str:
    numbered = '\n'.join((f'{index}. {step}' for index, step in enumerate(reasoning, 1)))
    return f'{numbered}\nAnswer: {gold_letter}'

def rebuild_rows(source_rows: list[dict], seed: int, source_sha256: str) -> list[dict]:
    if len({row['id'] for row in source_rows}) != len(source_rows):
        raise ValueError('Duplicate source Route Shape IDs')
    assignments = assigned_gold_letters(source_rows, seed)
    rebuilt: list[dict] = []
    for source in source_rows:
        row = copy.deepcopy(source)
        label = normalized_label(row['answer'])
        gold_letter = assignments[row['id']]
        options = option_mapping(row['id'], label, gold_letter, seed)
        user = next((message for message in row['messages'] if message['role'] == 'user'))
        assistant = next((message for message in row['messages'] if message['role'] == 'assistant'))
        text_parts = [part for part in user['content'] if part.get('type') == 'text']
        assistant_parts = [part for part in assistant['content'] if part.get('type') == 'text']
        if len(text_parts) != 1 or len(assistant_parts) != 1:
            raise ValueError(f"Unexpected message structure for {row['id']}")
        text_parts[0]['text'] = prompt_text(options)
        reasoning = source_reasoning_steps(source, label)
        assistant_parts[0]['text'] = assistant_text(reasoning, gold_letter)
        row['answer'] = label
        row['answer_key'] = gold_letter
        row['question'] = "What's the shape of my moving trajectory?"
        row['choices'] = options
        row['gold_reasoning'] = reasoning
        row['sample_weight'] = 1.0
        metadata = row.setdefault('metadata', {})
        metadata['route_shape_redesign'] = {'version': VERSION, 'source_id': source['id'], 'source_sha256': source_sha256, 'seed': seed, 'ontology': list(LABELS), 'all_five_options_present': True, 'deterministic_option_permutation': True, 'assistant_target': 'concise_grounded_steps_then_answer', 'benchmark_qa_used': False}
        rebuilt.append(row)
    return rebuilt

def validate(rows: list[dict], expected_count: int=1000) -> dict:
    if len(rows) != expected_count:
        raise ValueError(f'Expected {expected_count} rows, found {len(rows)}')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate rebuilt IDs')
    if len({row['video_path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate rebuilt video paths')
    missing = [row['video_path'] for row in rows if not Path(row['video_path']).is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} videos; first: {missing[0]}')
    label_counts = Counter((row['answer'] for row in rows))
    position_counts = Counter((row['answer_key'] for row in rows))
    permutations = Counter((tuple((row['choices'][letter] for letter in LETTERS)) for row in rows))
    for row in rows:
        if set(row['choices']) != set(LETTERS) or set(row['choices'].values()) != set(LABELS):
            raise ValueError(f"Invalid five-choice ontology for {row['id']}")
        if row['choices'][row['answer_key']] != row['answer']:
            raise ValueError(f"Answer mismatch for {row['id']}")
        assistant = next((message for message in row['messages'] if message['role'] == 'assistant'))
        answer_text = assistant['content'][0]['text']
        if not answer_text.endswith(f"Answer: {row['answer_key']}"):
            raise ValueError(f"Non-canonical final answer for {row['id']}")
        if not 2 <= len(row['gold_reasoning']) <= 7:
            raise ValueError(f"Invalid reasoning length for {row['id']}")
        if 'correct answer' in answer_text.lower() or 'choice ' in answer_text.lower():
            raise ValueError(f"Old option reference remains in reasoning for {row['id']}")
    if len(set(position_counts.values())) != 1:
        raise ValueError(f'Answer positions are not balanced: {position_counts}')
    return {'records': len(rows), 'label_counts': dict(label_counts), 'answer_position_counts': dict(position_counts), 'unique_option_permutations': len(permutations), 'unique_videos': len({row['video_path'] for row in rows}), 'missing_videos': len(missing)}
