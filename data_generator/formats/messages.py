"""Final data-construction component: formats/messages.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
TASK_INSTRUCTIONS = {'route_plan': 'Infer the ordered navigation actions from the video.', 'route_shape': 'Infer the overall shape of the observed route.', 'reverse_route_plan': 'Recover the forward route, then infer the reverse route back to the start.', 'relative_direction': "Infer the target object's relative direction from the specified standing pose.", 'event_order': 'Infer the chronological order of the actions in the clip.', 'before_or_after_action': 'Infer the action immediately before or after the queried action.'}
TURN_RE = re.compile('\\bturn\\s+(?:left|right|back)\\b', re.IGNORECASE)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--extra-input', type=Path, action='append', default=[], help='Additional QA JSONL files to convert into the same output.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, default=None)
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--max-pixels', type=int, default=360 * 420)
    parser.add_argument('--allow-unreviewed', action='store_true', help='Explicitly export rows marked human_review_required.')
    parser.add_argument('--review-override-note', default='', help='Audit note recorded when --allow-unreviewed is used.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    inputs = [args.input, *args.extra_input]
    rows = [row for path in inputs for row in read_jsonl(path)]
    duplicate_ids = len(rows) - len({str(row.get('id')) for row in rows})
    if duplicate_ids:
        raise ValueError(f'Refusing to export {duplicate_ids} duplicate IDs')
    converted = [convert_row(row, fps=args.fps, max_pixels=args.max_pixels, allow_unreviewed=args.allow_unreviewed, review_override_note=args.review_override_note) for row in rows]
    write_jsonl(args.output, converted, overwrite=args.overwrite)
    summary = {'inputs': [str(path) for path in inputs], 'output': str(args.output), 'records': len(converted), 'unique_ids': len({row['id'] for row in converted}), 'unique_videos': len({row['video_path'] for row in converted}), 'task_counts': dict(sorted(Counter((row['task_type'] for row in converted)).items())), 'dataset_source_counts': dict(sorted(Counter((row['dataset_source'] for row in converted)).items())), 'missing_video_files': sum((not Path(row['video_path']).is_file() for row in converted)), 'allow_unreviewed': args.allow_unreviewed, 'review_override_note': args.review_override_note}
    summary_out = args.summary_out or args.output.with_suffix('.summary.json')
    Path(summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

def convert_row(row: dict[str, Any], *, fps: float, max_pixels: int, allow_unreviewed: bool=False, review_override_note: str='') -> dict[str, Any]:
    metadata = row.get('metadata') or {}
    blocked = str(metadata.get('rebuild_status', '')).startswith('blocked_')
    unreviewed = bool(metadata.get('human_review_required'))
    if blocked or (unreviewed and (not allow_unreviewed)):
        raise ValueError(f"Unreviewed rebuilt reference cannot be exported for SFT: {row.get('id')}")
    video_path = str(row.get('video_path') or '')
    if not video_path:
        raise ValueError(f"Missing video_path for {row.get('id')}")
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    choices = row.get('choices') or {}
    if not choices:
        raise ValueError(f"Missing choices for {row.get('id')}")
    if str(row.get('task_type') or '') == 'route_plan':
        blank_count = str(row.get('question') or '').count('[?]')
        choice_turn_counts = {str(key): len(TURN_RE.findall(str(value))) for key, value in choices.items()}
        if not blank_count or any((count != blank_count for count in choice_turn_counts.values())):
            raise ValueError(f"Route-plan choice lengths do not match {blank_count} blanks for {row.get('id')}: {choice_turn_counts}")
    answer_key = str(row.get('answer_key') or '').strip()
    answer = str(row.get('answer') or choices.get(answer_key) or '').strip()
    if not answer_key or not answer:
        raise ValueError(f"Missing answer for {row.get('id')}")
    task_type = str(row.get('task_type') or '')
    prompt = build_user_prompt(row, choices)
    target = build_assistant_target(row, answer_key, answer)
    return {'id': row.get('id'), 'dataset_source': row.get('dataset_source'), 'task_type': task_type, 'video_path': video_path, 'answer_key': answer_key, 'answer': answer, 'messages': [{'role': 'user', 'content': [{'type': 'video', 'video': video_path, 'fps': fps, 'max_pixels': max_pixels}, {'type': 'text', 'text': prompt}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': target}]}], 'metadata': {'source_id': row.get('id'), 'source_video_id': row.get('video_id'), 'source_scene_id': row.get('scene_id'), 'source_dataset': row.get('dataset_source'), 'source_evaluation_valid': metadata.get('evaluation_valid'), 'source_human_review_required': metadata.get('human_review_required'), 'source_rebuild_status': metadata.get('rebuild_status'), 'review_override': bool(allow_unreviewed and unreviewed), 'review_override_note': review_override_note if allow_unreviewed and unreviewed else ''}}

def build_user_prompt(row: dict[str, Any], choices: dict[str, Any]) -> str:
    task_type = str(row.get('task_type') or '')
    lines = [TASK_INSTRUCTIONS.get(task_type, 'Answer the video question.'), '', f"Question: {row.get('question')}", '', 'Options:']
    for key in sorted(choices):
        lines.append(f'{key}. {choices[key]}')
    lines.extend(['', 'Think step by step using only visible evidence from the video.', 'End with exactly one final line in this format: Answer: <option letter>'])
    return '\n'.join(lines)

def build_assistant_target(row: dict[str, Any], answer_key: str, answer: str) -> str:
    steps = [str(step).strip() for step in row.get('gold_reasoning') or [] if str(step).strip()]
    if not steps:
        steps = [f'The correct option is {answer_key}: {answer}.']
    body = '\n'.join((f'{idx}. {step}' for idx, step in enumerate(steps, start=1)))
    return f'{body}\nAnswer: {answer_key}'

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    main()
