"""Final data-construction component: formats/options.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
OPTION_RE = re.compile('^([A-Z])\\.\\s+(.*)$')
CHOICE_REF_RE = re.compile('\\bchoice\\s+([A-Z])\\b', re.IGNORECASE)
FINAL_ANSWER_RE = re.compile('(?m)^Answer:\\s*([A-Z])\\s*$')
ANSWER_REFERENCE_RE = re.compile('(?P<prefix>(?i:\\b(?:choice|select option|(?:the )?(?:correct )?answer is|(?:the )?correct option is)\\s+(?:choice\\s+|option\\s+)?)|(?m:^Answer:\\s*))(?P<letter>[A-Z])\\b')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--max-video-frames', type=int, default=24)
    parser.add_argument('--route-plan-loss-weight', type=float, default=0.6)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def stable_hash(seed: int, *parts: object) -> str:
    text = ':'.join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def option_block(row: dict[str, Any]) -> tuple[list[str], list[str], int, int]:
    text = row['messages'][0]['content'][-1]['text']
    lines = text.splitlines()
    try:
        options_header = lines.index('Options:')
    except ValueError as exc:
        raise ValueError(f"{row.get('id')}: missing Options header") from exc
    start = options_header + 1
    labels: list[str] = []
    values: list[str] = []
    end = start
    while end < len(lines):
        match = OPTION_RE.match(lines[end])
        if not match:
            break
        labels.append(match.group(1))
        values.append(match.group(2))
        end += 1
    if len(labels) < 2 or labels != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(labels)]):
        raise ValueError(f"{row.get('id')}: invalid option labels {labels}")
    return (labels, values, start, end)

def replace_answer_references(text: str, mapping: dict[str, str]) -> str:
    return ANSWER_REFERENCE_RE.sub(lambda match: match['prefix'] + mapping.get(match['letter'], match['letter']), text)

def shuffle_group(rows: list[dict[str, Any]], *, labels: list[str], seed: int) -> None:
    ordered = sorted(rows, key=lambda row: stable_hash(seed, row['task_type'], row['id']))
    for index, row in enumerate(ordered):
        current_labels, values, start, end = option_block(row)
        if current_labels != labels:
            raise ValueError(f"{row.get('id')}: inconsistent option labels")
        old_answer = str(row['answer_key'])
        target_answer = labels[index % len(labels)]
        distractors = [label for label in labels if label != old_answer]
        rng = random.Random(int(stable_hash(seed, 'distractors', row['id']), 16))
        rng.shuffle(distractors)
        remaining_positions = [label for label in labels if label != target_answer]
        old_to_new = {old_answer: target_answer}
        old_to_new.update(zip(distractors, remaining_positions))
        if all((old_to_new[label] == label for label in labels)) and len(distractors) >= 2:
            left, right = distractors[:2]
            old_to_new[left], old_to_new[right] = (old_to_new[right], old_to_new[left])
        value_by_old = dict(zip(labels, values))
        old_by_new = {new: old for old, new in old_to_new.items()}
        user_text = row['messages'][0]['content'][-1]['text']
        lines = user_text.splitlines()
        lines[start:end] = [f'{label}. {value_by_old[old_by_new[label]]}' for label in labels]
        row['messages'][0]['content'][-1]['text'] = '\n'.join(lines)
        assistant = row['messages'][1]['content'][0]['text']
        row['messages'][1]['content'][0]['text'] = replace_answer_references(assistant, old_to_new)
        row['answer_key'] = target_answer
        row.setdefault('metadata', {})['v3_choice_shuffle'] = {'seed': seed, 'old_to_new': old_to_new, 'fixed': True}

def prepare_rows(rows: list[dict[str, Any]], *, seed: int, max_video_frames: int, route_plan_loss_weight: float) -> dict[str, Any]:
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels, _, _, _ = option_block(row)
        groups[str(row['task_type']), tuple(labels)].append(row)
    for (_, labels), group in sorted(groups.items()):
        shuffle_group(group, labels=list(labels), seed=seed)
    for row in rows:
        for content in row['messages'][0]['content']:
            if content.get('type') == 'video':
                content['max_frames'] = max_video_frames
                content.pop('nframes', None)
        row['sample_weight'] = route_plan_loss_weight if row['task_type'] == 'route_plan' else 1.0
        row.setdefault('metadata', {})['v3_training'] = {'max_video_frames': max_video_frames, 'uniform_sampling': True, 'sample_weight': row['sample_weight']}
    answer_positions = {f"{task}|{''.join(labels)}": dict(sorted(Counter((row['answer_key'] for row in group)).items())) for (task, labels), group in sorted(groups.items())}
    return {'records': len(rows), 'unique_ids': len({row['id'] for row in rows}), 'task_counts': dict(sorted(Counter((row['task_type'] for row in rows)).items())), 'answer_positions_by_task_and_option_set': answer_positions, 'seed': seed, 'max_video_frames': max_video_frames, 'route_plan_loss_weight': route_plan_loss_weight, 'missing_videos': sum((not Path(row['video_path']).is_file() for row in rows))}

def main() -> None:
    args = parse_args()
    if args.output.exists() and (not args.overwrite):
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    summary = prepare_rows(rows, seed=args.seed, max_video_frames=args.max_video_frames, route_plan_loss_weight=args.route_plan_loss_weight)
    if summary['unique_ids'] != summary['records']:
        raise ValueError('Duplicate IDs in V3 training input')
    if summary['missing_videos']:
        raise FileNotFoundError(f"Missing {summary['missing_videos']} video files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
