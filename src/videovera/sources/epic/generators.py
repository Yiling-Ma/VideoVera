from __future__ import annotations
import hashlib
import random
import re
from pathlib import Path
from typing import Any
from .annotations import ActionSegment
from .media import extract_clip_range, extract_support_frames
from .qa_schema import make_record, make_step
DEFAULT_MAX_OVERLAP_RATIO = 0.55

def empty_reject_stats() -> dict[str, int]:
    return {'generated': 0, 'accepted': 0, 'rejected_duplicate_choices': 0, 'rejected_non_adjacent_pairs': 0, 'rejected_temporal_overlap': 0, 'rejected_insufficient_distractors': 0, 'rejected_validation': 0, 'rejected_other': 0}

def generate_records(*, segments: list[ActionSegment], video_path: Path, annotations_path: Path, clips_dir: Path, frames_dir: Path, frames_per_segment: int, pad_before: float, pad_after: float, clip_duration_limit: float | None, max_records: int | None=None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    single_cache: dict[str, tuple[str, list[str]]] = {}
    multi_cache: dict[str, tuple[str, list[str]]] = {}
    counters = {'event_order': 0, 'before_or_after_action': 0, 'constrained_action_count': 0, 'appearance_order': 0}
    for idx in range(len(segments)):
        if len(records) >= (max_records or 10 ** 9):
            break
        if idx + 3 < len(segments):
            window4 = segments[idx:idx + 4]
            if is_local_window(window4, 18.0) and distinct_labels(window4) and (not has_ambiguous_overlap(window4)):
                counters['event_order'] += 1
                row = make_event_order_record(window=window4, video_path=video_path, annotations_path=annotations_path, clips_dir=clips_dir, frames_dir=frames_dir, frames_per_segment=frames_per_segment, pad_before=pad_before, pad_after=pad_after, clip_duration_limit=clip_duration_limit, multi_cache=multi_cache, sample_idx=counters['event_order'])
                if row is not None:
                    records.append(row)
        if len(records) >= (max_records or 10 ** 9):
            break
        next_record = maybe_make_conditional_adjacent_record(idx=idx, direction='next', segments=segments, video_path=video_path, annotations_path=annotations_path, clips_dir=clips_dir, frames_dir=frames_dir, frames_per_segment=frames_per_segment, pad_before=pad_before, pad_after=pad_after, clip_duration_limit=clip_duration_limit, single_cache=single_cache, sample_idx=counters['before_or_after_action'] + 1)
        if next_record is not None:
            counters['before_or_after_action'] += 1
            records.append(next_record)
        if len(records) >= (max_records or 10 ** 9):
            break
        prev_record = maybe_make_conditional_adjacent_record(idx=idx, direction='previous', segments=segments, video_path=video_path, annotations_path=annotations_path, clips_dir=clips_dir, frames_dir=frames_dir, frames_per_segment=frames_per_segment, pad_before=pad_before, pad_after=pad_after, clip_duration_limit=clip_duration_limit, single_cache=single_cache, sample_idx=counters['before_or_after_action'] + 1)
        if prev_record is not None:
            counters['before_or_after_action'] += 1
            records.append(prev_record)
        if len(records) >= (max_records or 10 ** 9):
            break
        count_record = maybe_make_constrained_count_record(idx=idx, segments=segments, video_path=video_path, annotations_path=annotations_path, clips_dir=clips_dir, frames_dir=frames_dir, frames_per_segment=frames_per_segment, pad_before=pad_before, pad_after=pad_after, clip_duration_limit=clip_duration_limit, multi_cache=multi_cache, sample_idx=counters['constrained_action_count'] + 1)
        if count_record is not None:
            counters['constrained_action_count'] += 1
            records.append(count_record)
        if len(records) >= (max_records or 10 ** 9):
            break
        appearance_record = maybe_make_appearance_order_record(idx=idx, segments=segments, video_path=video_path, annotations_path=annotations_path, clips_dir=clips_dir, frames_dir=frames_dir, frames_per_segment=frames_per_segment, pad_before=pad_before, pad_after=pad_after, clip_duration_limit=clip_duration_limit, multi_cache=multi_cache, sample_idx=counters['appearance_order'] + 1)
        if appearance_record is not None:
            counters['appearance_order'] += 1
            records.append(appearance_record)
    return dedupe_records(records)[:max_records]

def is_local_window(window: list[ActionSegment], max_span_sec: float) -> bool:
    return bool(window) and window[-1].stop_sec - window[0].start_sec <= max_span_sec

def distinct_labels(window: list[ActionSegment]) -> bool:
    return len({canonicalize_action(seg.verb, seg.noun) for seg in window}) == len(window)

def sort_segments_chronologically(segments: list[ActionSegment]) -> list[ActionSegment]:
    return sorted(segments, key=lambda seg: (seg.start_frame, seg.start_sec, seg.uid))

def overlap_duration_sec(a: ActionSegment, b: ActionSegment) -> float:
    return max(0.0, min(a.stop_sec, b.stop_sec) - max(a.start_sec, b.start_sec))

def pair_overlap_too_large(a: ActionSegment, b: ActionSegment, max_overlap_ratio: float=DEFAULT_MAX_OVERLAP_RATIO) -> bool:
    overlap = overlap_duration_sec(a, b)
    if overlap <= 0:
        return False
    shorter = min(max(a.duration_sec, 1e-06), max(b.duration_sec, 1e-06))
    return overlap / shorter > max_overlap_ratio

def has_ambiguous_overlap(window: list[ActionSegment], max_overlap_ratio: float=DEFAULT_MAX_OVERLAP_RATIO) -> bool:
    ordered = sort_segments_chronologically(window)
    for i in range(len(ordered) - 1):
        if ordered[i].start_sec >= ordered[i + 1].start_sec:
            return True
        if pair_overlap_too_large(ordered[i], ordered[i + 1], max_overlap_ratio):
            return True
    return False

def normalize_token(text: str) -> str:
    text = (text or '').strip().lower()
    text = text.replace('-', ' ').replace('_', ' ')
    text = re.sub('\\s+', ' ', text)
    return text

def normalize_noun_display(noun: str) -> str:
    """Convert ontology nouns like food:remaining -> remaining food for display only."""
    noun = (noun or '').strip()
    if not noun:
        return ''
    if ':' in noun:
        parts = [normalize_token(p) for p in noun.split(':') if p.strip()]
        if len(parts) >= 2:
            return ' '.join([parts[-1]] + parts[:-1]).strip()
        return ' '.join(parts)
    return normalize_token(noun)

def canonicalize_action(verb: str, noun: str) -> tuple[str, str]:
    return (normalize_token(verb), normalize_noun_display(noun))

def canonicalize_choice_text(text: str) -> str:
    """Canonical key for MCQ uniqueness (handles sequences and single actions)."""
    text = normalize_token(text)
    text = text.replace(' the ', ' ')
    text = re.sub('\\b(a|an)\\b', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    text = text.replace(' -> ', ' ').replace('→', ' ')
    text = re.sub('\\s*,\\s*', ' ', text)
    return text

def _has_leading_determiner(text: str) -> bool:
    lowered = normalize_token(text)
    return lowered.startswith(('the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ', 'my ', 'your ', 'his ', 'her ', 'their ', 'its ', 'some ', 'another ', 'one ', 'two ', 'three '))

def _ensure_definite_np(text: str) -> str:
    phrase = re.sub('\\s+', ' ', (text or '').strip())
    if not phrase or _has_leading_determiner(phrase):
        return phrase
    return f'the {phrase}'

def _repair_narration_display(narration: str, *, verb_n: str, noun_disp: str) -> str:
    cleaned = re.sub('\\s+', ' ', narration.replace('-', ' ')).strip()
    if not cleaned:
        return cleaned
    lower = cleaned.lower()
    if lower.startswith('pick up '):
        rest = cleaned[len('pick up '):].strip()
        if ' from ' in rest:
            head, tail = rest.split(' from ', 1)
            return f'pick up {_ensure_definite_np(head)} from {_ensure_definite_np(tail)}'
        return f'pick up {_ensure_definite_np(rest)}'
    if lower.startswith('put on '):
        rest = cleaned[len('put on '):].strip()
        return f'put on {_ensure_definite_np(rest)}'
    if lower.startswith('put down '):
        rest = cleaned[len('put down '):].strip()
        return f'put down {_ensure_definite_np(rest)}'
    if lower.startswith('turn on '):
        rest = cleaned[len('turn on '):].strip()
        return f'turn on {_ensure_definite_np(rest)}'
    if lower.startswith('turn off '):
        rest = cleaned[len('turn off '):].strip()
        return f'turn off {_ensure_definite_np(rest)}'
    if lower.startswith('open '):
        rest = cleaned[len('open '):].strip()
        return f'open {_ensure_definite_np(rest)}'
    if lower.startswith('close '):
        rest = cleaned[len('close '):].strip()
        return f'close {_ensure_definite_np(rest)}'
    if lower.startswith('throw ') and ' for recycling' in lower:
        match = re.match('throw\\s+(.+?)\\s+for recycling$', cleaned, flags=re.IGNORECASE)
        if match:
            obj = _ensure_definite_np(match.group(1))
            return f'throw {obj} away for recycling'
    if lower.startswith('throw ') and ' into ' in lower:
        head, tail = re.split('\\s+into\\s+', cleaned, maxsplit=1, flags=re.IGNORECASE)
        obj = _ensure_definite_np(head[len('throw '):].strip())
        return f'throw {obj} into {_ensure_definite_np(tail)}'
    if lower.startswith('throw ') and ' in ' in lower:
        match = re.match('throw\\s+(.+?)\\s+in\\s+(.+)$', cleaned, flags=re.IGNORECASE)
        if match:
            obj = _ensure_definite_np(match.group(1))
            dest = _ensure_definite_np(match.group(2))
            return f'throw {obj} into {dest}'
    if lower.startswith('put ') and ' into ' in lower:
        match = re.match('put\\s+(.+?)\\s+into\\s+(.+)$', cleaned, flags=re.IGNORECASE)
        if match:
            obj = _ensure_definite_np(match.group(1))
            dest = _ensure_definite_np(match.group(2))
            return f'put {obj} into {dest}'
    if lower.startswith('put ') and ' inside ' in lower:
        match = re.match('put\\s+(.+?)\\s+inside\\s+(.+)$', cleaned, flags=re.IGNORECASE)
        if match:
            obj = _ensure_definite_np(match.group(1))
            dest = _ensure_definite_np(match.group(2))
            return f'put {obj} inside {dest}'
    if lower.startswith('put ') and ' in ' in lower:
        match = re.match('put\\s+(.+?)\\s+in\\s+(.+)$', cleaned, flags=re.IGNORECASE)
        if match:
            obj = _ensure_definite_np(match.group(1))
            dest = _ensure_definite_np(match.group(2))
            return f'put {obj} in {dest}'
    if noun_disp and noun_disp.split()[-1] in lower:
        return cleaned
    return cleaned

def _verb_gerund(verb_n: str, fallback_word: str) -> str:
    gerund_map = {'pick up': 'picking up', 'put down': 'putting down', 'put in': 'putting', 'put into': 'putting', 'put on': 'putting on', 'turn on': 'turning on', 'turn off': 'turning off', 'throw in': 'throwing', 'throw for': 'throwing', 'take out': 'taking out', 'take': 'taking', 'wash': 'washing', 'open': 'opening', 'close': 'closing', 'pour': 'pouring', 'cut': 'cutting', 'mix': 'mixing', 'stir': 'stirring', 'wipe': 'wiping', 'move': 'moving', 'insert': 'inserting', 'remove': 'removing', 'squeeze': 'squeezing', 'press': 'pressing', 'hold': 'holding', 'adjust': 'adjusting'}
    if verb_n in gerund_map:
        return gerund_map[verb_n]
    first = fallback_word.lower()
    if first == 'put':
        return 'putting'
    if first.endswith('e') and first not in {'be'}:
        return f'{first[:-1]}ing'
    return f'{first}ing'

def _verb_present(verb_n: str, fallback_word: str) -> str:
    present_map = {'pick up': 'picks up', 'put down': 'puts down', 'put in': 'puts', 'put into': 'puts', 'put on': 'puts on', 'turn on': 'turns on', 'turn off': 'turns off', 'throw in': 'throws', 'throw for': 'throws', 'take out': 'takes out', 'take': 'takes', 'open': 'opens', 'close': 'closes', 'wash': 'washes', 'pour': 'pours', 'cut': 'cuts', 'mix': 'mixes', 'stir': 'stirs', 'wipe': 'wipes', 'move': 'moves', 'hold': 'holds', 'press': 'presses', 'insert': 'inserts', 'remove': 'removes', 'squeeze': 'squeezes', 'adjust': 'adjusts'}
    if verb_n in present_map:
        return present_map[verb_n]
    first = fallback_word.lower()
    if first == 'do':
        return 'does'
    if first.endswith(('sh', 'ch', 'x', 'z', 's')):
        return f'{first}es'
    return f'{first}s'

def _replace_leading_verb_phrase(display: str, verb_n: str, replacement: str) -> str:
    parts = display.split()
    if not parts:
        return display
    verb_parts = verb_n.split()
    if len(verb_parts) >= 2 and len(parts) >= len(verb_parts):
        if [p.lower() for p in parts[:len(verb_parts)]] == verb_parts:
            return f"{replacement} {' '.join(parts[len(verb_parts):])}".strip()
    return f"{replacement} {' '.join(parts[1:])}".strip()

def display_action(seg: ActionSegment) -> str:
    return action_to_display_text(seg.verb, seg.noun, seg.narration)

def action_to_display_text(verb: str, noun: str, narration: str | None=None) -> str:
    """Model-facing action phrase. Does not mutate grounding verb/noun fields."""
    verb_n = normalize_token(verb)
    noun_disp = normalize_noun_display(noun)
    narr = (narration or '').strip()
    if narr and ':' not in narr and (narr.lower() not in {'', 'none'}):
        repaired = _repair_narration_display(narr, verb_n=verb_n, noun_disp=noun_disp)
        if repaired:
            return repaired
    article_noun = noun_disp
    if article_noun and (not article_noun.startswith(('the ', 'a ', 'an '))):
        article_noun = f'the {article_noun}'
    if verb_n == 'throw in':
        if narr and 'bin' in narr.lower():
            return f'throw {article_noun} into the bin'.replace('the the ', 'the ')
        return f'throw in {article_noun}'.replace('the the ', 'the ')
    if verb_n == 'put inside':
        return f'put {article_noun} inside'.replace('the the ', 'the ').strip()
    if verb_n in {'put in', 'put into'}:
        return f'put {article_noun} in'.replace('the the ', 'the ').strip()
    if verb_n == 'put on':
        return f'put on {article_noun}'.replace('the the ', 'the ')
    if verb_n in {'pick up', 'put down', 'turn on', 'turn off', 'take', 'open', 'close'}:
        if verb_n == 'take':
            verb_n = 'pick up'
        return f'{verb_n} {article_noun}'.replace('the the ', 'the ').strip()
    return f'{verb_n} {article_noun}'.replace('the the ', 'the ').strip()

def action_to_gerund(verb: str, noun: str, narration: str | None=None) -> str:
    """Gerund phrase for natural before/after questions, e.g. 'putting down the plate'."""
    verb_n = normalize_token(verb)
    display = action_to_display_text(verb, noun, narration)
    parts = display.split()
    if not parts:
        return display
    return _replace_leading_verb_phrase(display, verb_n, _verb_gerund(verb_n, parts[0]))

def _event_label(seg: ActionSegment) -> str:
    return display_action(seg)

def humanize_action(verb: str, noun: str) -> str:
    return action_to_display_text(verb, noun)

def observed_phrase(seg: ActionSegment) -> str:
    """Present-tense observation used in gold reasoning (chronological evidence)."""
    return display_action(seg)

def _stable_rng(seed_text: str) -> random.Random:
    digest = hashlib.md5(seed_text.encode('utf-8')).hexdigest()
    return random.Random(int(digest[:16], 16))

def shuffle_choices_and_update_key(gold: str, distractors: list[str], seed_text: str) -> tuple[dict[str, str], str]:
    """Shuffle gold+distractors into A-D and recompute answer_key."""
    options = [gold] + list(distractors)
    if len(options) != 4:
        raise ValueError(f'Expected 4 options before shuffle, got {len(options)}')
    rng = _stable_rng(seed_text)
    rng.shuffle(options)
    mapping = {letter: text for letter, text in zip(['A', 'B', 'C', 'D'], options)}
    answer_key = next((k for k, v in mapping.items() if v == gold))
    assert mapping[answer_key] == gold, 'answer_key does not match gold text after shuffle'
    return (mapping, answer_key)

def choices_are_unique(choice_texts: list[str]) -> bool:
    canon = [canonicalize_choice_text(t) for t in choice_texts]
    return len(canon) == 4 and len(set(canon)) == 4

def generate_unique_order_distractors(labels: list[str]) -> list[str] | None:
    """Use a four-corner permutation orbit, not a uniquely central gold option."""
    if len(labels) != 4:
        return None
    gold = tuple(labels)
    seed = hashlib.sha256('\\0'.join(sorted(labels)).encode()).hexdigest()
    indices = list(range(4))
    random.Random(seed).shuffle(indices)
    pairs = (indices[:2], indices[2:])
    candidates = []
    for mask in (1, 2, 3):
        order = list(labels)
        for bit, (a, b) in enumerate(pairs):
            if mask & 1 << bit:
                order[a], order[b] = (order[b], order[a])
        candidates.append(tuple(order))
    distractors: list[str] = []
    seen = {canonicalize_choice_text(', '.join(gold))}
    for order in candidates:
        text = ', '.join(order)
        key = canonicalize_choice_text(text)
        if key in seen:
            continue
        seen.add(key)
        distractors.append(text)
    if len(distractors) < 3:
        return None
    return distractors[:3]

def generate_unique_action_distractors(*, gold_seg: ActionSegment, exclude_uids: set[str], local_segments: list[ActionSegment], all_segments: list[ActionSegment]) -> list[ActionSegment] | None:
    """Prefer local/nearby/same-verb-or-noun confounders; require 3 unique actions."""
    gold_key = canonicalize_action(gold_seg.verb, gold_seg.noun)
    used = {gold_key}
    selected: list[ActionSegment] = []

    def try_add(seg: ActionSegment) -> None:
        if seg.uid in exclude_uids:
            return
        key = canonicalize_action(seg.verb, seg.noun)
        if key in used:
            return
        used.add(key)
        selected.append(seg)
    for seg in local_segments:
        try_add(seg)
        if len(selected) >= 3:
            return selected[:3]
    nearby = sorted([seg for seg in all_segments if seg.uid not in exclude_uids], key=lambda seg: abs(seg.start_sec - gold_seg.start_sec))
    same_verb_or_noun = [seg for seg in nearby if seg.verb == gold_seg.verb or seg.noun == gold_seg.noun]
    for pool in (same_verb_or_noun, nearby):
        for seg in pool:
            try_add(seg)
            if len(selected) >= 3:
                return selected[:3]
    return None

def _model_facing_texts(instance: dict[str, Any]) -> list[str]:
    texts = [str(instance.get('question') or ''), str(instance.get('answer') or ''), *[str(v) for v in (instance.get('choices') or {}).values()], *[str(s) for s in instance.get('gold_reasoning') or []]]
    for step in instance.get('gold_trace') or []:
        texts.append(str(step.get('claim') or ''))
        texts.append(str(step.get('model_facing_step') or ''))
    return texts

def validate_temporal_instance(instance: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    choices = instance.get('choices') or {}
    answer_key = instance.get('answer_key')
    answer = instance.get('answer')
    task_type = instance.get('task_type')
    grounding = instance.get('grounding') or {}
    if set(choices.keys()) != {'A', 'B', 'C', 'D'}:
        reasons.append('choices_not_abcd')
    choice_vals = [choices.get(k, '') for k in ['A', 'B', 'C', 'D']]
    if len(choice_vals) != 4:
        reasons.append('not_four_choices')
    if not choices_are_unique(choice_vals):
        reasons.append('duplicate_or_alias_choices')
    if answer_key not in {'A', 'B', 'C', 'D'}:
        reasons.append('invalid_answer_key')
    if answer_key in choices and choices[answer_key] != answer:
        reasons.append('answer_key_mismatch')
    if answer_key in choices:
        gold_hits = sum((1 for v in choice_vals if canonicalize_choice_text(v) == canonicalize_choice_text(str(answer))))
        if gold_hits != 1:
            reasons.append('gold_not_unique_in_choices')
    frame_paths: list[str] = []
    if grounding.get('shared_support_frames'):
        frame_paths.extend(grounding['shared_support_frames'])
    for key in ('ordered_segments',):
        for seg in grounding.get(key) or []:
            frame_paths.extend(seg.get('support_frames') or [])
    for key in ('anchor_segment', 'adjacent_segment'):
        seg = grounding.get(key) or {}
        frame_paths.extend(seg.get('support_frames') or [])
    for path in frame_paths:
        if path and (not Path(path).exists()):
            reasons.append(f'missing_support_frame:{path}')
            break
    for text in _model_facing_texts(instance):
        if ':' in text and re.search('\\w:\\w', text):
            reasons.append('ontology_marker_in_model_text')
            break
    for step in instance.get('gold_trace') or []:
        verifier = step.get('verifier') or {}
        rule = verifier.get('rule')
        expected = verifier.get('expected')
        if rule in {'choice_sequence_equals', 'choice_action_equals'} and expected != answer_key:
            reasons.append('verifier_answer_key_desync')
            break
    if task_type == 'event_order':
        q = str(instance.get('question') or '')
        gold_seq = str(answer or '')
        if gold_seq and gold_seq in q:
            reasons.append('event_order_question_leaks_gold')
        ordered = grounding.get('ordered_segments') or []
        starts = [float(seg.get('start_sec', 0)) for seg in ordered]
        if len(starts) >= 2 and any((starts[i] >= starts[i + 1] for i in range(len(starts) - 1))):
            reasons.append('event_order_onset_not_increasing')
        last_claim = ''
        if instance.get('gold_reasoning'):
            last_claim = str(instance['gold_reasoning'][-1])
        if 'wrong options only swap' in last_claim.lower():
            reasons.append('event_order_meta_reasoning')
    if task_type == 'before_or_after_action':
        direction = grounding.get('direction')
        anchor = grounding.get('anchor_segment') or {}
        adjacent = grounding.get('adjacent_segment') or {}
        if direction not in {'next', 'previous', 'before', 'after'}:
            reasons.append('missing_direction')
        a_start = float(anchor.get('start_sec', -1))
        b_start = float(adjacent.get('start_sec', -1))
        if direction == 'next' and (not a_start < b_start):
            reasons.append('after_relation_inconsistent')
        if direction == 'previous' and (not b_start < a_start):
            reasons.append('before_relation_inconsistent')
        claims = instance.get('gold_reasoning') or []
        if len(claims) >= 2:
            earlier = anchor if a_start <= b_start else adjacent
            later = adjacent if a_start <= b_start else anchor
            c0 = str(claims[0]).lower()
            c1 = str(claims[1]).lower()
            earlier_label = action_to_display_text(str(earlier.get('verb') or ''), str(earlier.get('noun') or ''), str(earlier.get('narration') or '')).lower()
            later_label = action_to_display_text(str(later.get('verb') or ''), str(later.get('noun') or ''), str(later.get('narration') or '')).lower()
            if earlier_label and later_label:
                earlier_hit_first = any((tok in c0 for tok in earlier_label.split()[-2:]))
                later_hit_first = any((tok in c0 for tok in later_label.split()[-2:]))
                later_hit_second = any((tok in c1 for tok in later_label.split()[-2:]))
                if later_hit_first and (not earlier_hit_first) and (later_hit_second is False):
                    reasons.append('reasoning_frame_order_reversed')
    return (len(reasons) == 0, reasons)

def _segment_payload(seg: ActionSegment, support_frames: list[str]) -> dict[str, Any]:
    return {'segment_uid': seg.uid, 'video_id': seg.video_id, 'narration': seg.narration, 'verb': seg.verb, 'noun': seg.noun, 'label': display_action(seg), 'start_frame': seg.start_frame, 'stop_frame': seg.stop_frame, 'start_sec': round(seg.start_sec, 3), 'stop_sec': round(seg.stop_sec, 3), 'support_frames': support_frames}

def frame_span(start_idx: int, end_idx: int) -> str:
    if start_idx == end_idx:
        return f'Frame {start_idx}'
    return f'Frames {start_idx}-{end_idx}'

def with_frame_prefix(text: str, start_idx: int, end_idx: int) -> str:
    return f'{frame_span(start_idx, end_idx)}: {text}'

def ordered_observation(prefix: str, text: str) -> str:
    return f'{prefix}, {text}.'

def _sample_evenly_from_clip(*, video_path: Path, start_sec: float, end_sec: float, out_dir: Path, num_frames: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [out_dir / f'frame_{idx + 1:02d}.jpg' for idx in range(num_frames)]
    if all((p.exists() and p.stat().st_size > 0 for p in existing)):
        return [str(p) for p in existing]
    span = max(0.05, end_sec - start_sec)
    sampled: list[str] = []
    for idx in range(num_frames):
        t = start_sec + span * ((idx + 0.5) / num_frames)
        frame_paths = extract_support_frames(video_path=video_path, segment=ActionSegment(uid='sample', video_id='sample', participant_id='sample', narration='sample', verb='sample', verb_class=-1, noun='sample', noun_class=-1, start_frame=0, stop_frame=0, start_timestamp='00:00:00.00', stop_timestamp='00:00:00.00', start_sec=t, stop_sec=t + 0.001), out_dir=out_dir / f'tmp_{idx + 1:02d}', num_frames=1)
        src = Path(frame_paths[0])
        dst = out_dir / f'frame_{idx + 1:02d}.jpg'
        if not src.exists() or src.stat().st_size <= 0:
            raise FileNotFoundError(f'Failed to extract support frame {idx + 1}/{num_frames} at t={t:.3f}s from {video_path}; expected {src}')
        if src != dst:
            dst.write_bytes(src.read_bytes())
        sampled.append(str(dst))
    return sampled

def _materialize_multi(*, label: str, window: list[ActionSegment], video_path: Path, clips_dir: Path, frames_dir: Path, per_segment_frames: int, pad_before: float, pad_after: float, clip_duration_limit: float | None, cache: dict[str, tuple[str, list[str]]], existing_clip_path: Path | str | None=None, clip_start_sec: float | None=None, clip_end_sec: float | None=None) -> tuple[str, list[str]]:
    if label in cache:
        return cache[label]
    ordered_window = sort_segments_chronologically(list(window))
    if not ordered_window:
        raise ValueError('Cannot materialize clip for empty window')
    first_seg = ordered_window[0]
    last_seg = ordered_window[-1]
    if first_seg.start_frame > last_seg.stop_frame:
        raise AssertionError(f'clip frame order invalid for {label}: {first_seg.uid} start_frame={first_seg.start_frame} > {last_seg.uid} stop_frame={last_seg.stop_frame}')
    if existing_clip_path is not None:
        clip_path = Path(existing_clip_path)
        if not clip_path.exists() or clip_path.stat().st_size <= 0:
            raise FileNotFoundError(f'Pre-sliced clip not found: {clip_path}')
        start_sec = float(clip_start_sec if clip_start_sec is not None else max(0.0, first_seg.start_sec - pad_before))
        end_sec = float(clip_end_sec if clip_end_sec is not None else last_seg.stop_sec + pad_after)
        frame_source = video_path
        frame_start = start_sec
        frame_end = end_sec
    else:
        start_sec = max(0.0, first_seg.start_sec - pad_before)
        end_sec = last_seg.stop_sec + pad_after
        if clip_duration_limit is not None and end_sec - start_sec > clip_duration_limit:
            end_sec = start_sec + clip_duration_limit
        clip_path = clips_dir / first_seg.video_id / f'{label}.mp4'
        extract_clip_range(video_path=video_path, start_sec=start_sec, end_sec=end_sec, out_path=clip_path)
        frame_source = video_path
        frame_start = start_sec
        frame_end = end_sec
    if start_sec > end_sec:
        raise AssertionError(f'clip time order invalid for {label}: start_sec={start_sec} > end_sec={end_sec}')
    all_frames = _sample_evenly_from_clip(video_path=frame_source, start_sec=frame_start, end_sec=frame_end, out_dir=frames_dir / first_seg.video_id / label, num_frames=max(len(window) * per_segment_frames, per_segment_frames))
    value = (str(clip_path), all_frames)
    cache[label] = value
    return value

def make_event_order_record(**kwargs) -> dict[str, Any] | None:
    """Build an event_order MCQ. Returns None if distractors/validation fail."""
    reject_stats: dict[str, int] | None = kwargs.get('reject_stats')
    window: list[ActionSegment] = sort_segments_chronologically(list(kwargs['window']))
    if has_ambiguous_overlap(window):
        if reject_stats is not None:
            reject_stats['rejected_temporal_overlap'] += 1
        return None
    if not distinct_labels(window) or len(window) != 4:
        if reject_stats is not None:
            reject_stats['rejected_other'] += 1
        return None
    clip_label = kwargs.get('clip_id') or f'{window[0].video_id}_event_order_{window[0].uid}_{window[-1].uid}'
    clip_path, support_frames = _materialize_multi(label=clip_label, window=window, video_path=kwargs['video_path'], clips_dir=kwargs['clips_dir'], frames_dir=kwargs['frames_dir'], per_segment_frames=2, pad_before=kwargs['pad_before'], pad_after=kwargs['pad_after'], clip_duration_limit=max(kwargs['clip_duration_limit'] or 0, 16.0) if kwargs['clip_duration_limit'] else 16.0, cache=kwargs['multi_cache'], existing_clip_path=kwargs.get('existing_clip_path'), clip_start_sec=kwargs.get('clip_start_sec'), clip_end_sec=kwargs.get('clip_end_sec'))
    labels = [display_action(seg) for seg in window]
    gold_text = ', '.join(labels)
    distractors = generate_unique_order_distractors(labels)
    if distractors is None:
        if reject_stats is not None:
            reject_stats['rejected_insufficient_distractors'] += 1
        return None
    if not choices_are_unique([gold_text] + distractors):
        if reject_stats is not None:
            reject_stats['rejected_duplicate_choices'] += 1
        return None
    seed = str(kwargs.get('record_id') or kwargs.get('clip_id') or clip_label)
    mapping, answer_key = shuffle_choices_and_update_key(gold_text, distractors, seed)
    assert mapping[answer_key] == gold_text
    frame_groups = [support_frames[i * 2:(i + 1) * 2] for i in range(4)]
    order_arrow = ' → '.join(labels)
    gold_trace = [make_step(1, 'first_event', ordered_observation('First', labels[0]), _segment_payload(window[0], frame_groups[0]), 'segment_uid_equals', window[0].uid, verifier_step=with_frame_prefix(labels[0], 1, 2)), make_step(2, 'middle_event', ordered_observation('Next', labels[1]), _segment_payload(window[1], frame_groups[1]), 'segment_uid_equals', window[1].uid, verifier_step=with_frame_prefix(labels[1], 3, 4)), make_step(3, 'middle_event', ordered_observation('Then', labels[2]), _segment_payload(window[2], frame_groups[2]), 'segment_uid_equals', window[2].uid, verifier_step=with_frame_prefix(labels[2], 5, 6)), make_step(4, 'last_event', ordered_observation('Finally', labels[3]), _segment_payload(window[3], frame_groups[3]), 'segment_uid_equals', window[3].uid, verifier_step=with_frame_prefix(labels[3], 7, 8)), make_step(5, 'answer_grounding', f'Therefore, the chronological order is {order_arrow}. The correct answer is {answer_key}.', {'candidate_orders': mapping}, 'choice_sequence_equals', answer_key)]
    question = 'Which option correctly gives the chronological order of the four actions shown in the clip?'
    record = make_record(record_id=kwargs.get('record_id') or f"{window[0].video_id}_event_order_{kwargs['sample_idx']:03d}", video_id=window[0].video_id, annotation_path=kwargs['annotations_path'], task_type='event_order', question=question, choices=mapping, answer_key=answer_key, answer=gold_text, gold_trace=gold_trace, grounding={'ordered_segments': [_segment_payload(seg, frames) for seg, frames in zip(window, frame_groups)], 'clip_path': clip_path, 'clip_id': kwargs.get('clip_id'), 'shared_support_frames': support_frames, 'candidate_orders': mapping}, quality_score=4.0 + (window[-1].stop_sec - window[0].start_sec), quality_notes=['four-action order with hard distractors built from a local EPIC sequence'])
    if kwargs.get('source_video_path'):
        record['source_video_path'] = str(kwargs['source_video_path'])
    ok, reasons = validate_temporal_instance(record)
    if not ok:
        if reject_stats is not None:
            reject_stats['rejected_validation'] += 1
            reject_stats.setdefault('validation_failures', [])
            if isinstance(reject_stats.get('validation_failures'), list):
                reject_stats['validation_failures'].append({'id': record['id'], 'reasons': reasons})
        print(f"[skip] event_order validation failed for {record['id']}: {reasons}")
        return None
    return record

def maybe_make_conditional_adjacent_record(**kwargs) -> dict[str, Any] | None:
    idx: int = kwargs['idx']
    segments: list[ActionSegment] = kwargs['segments']
    direction: str = kwargs['direction']
    reject_stats: dict[str, int] | None = kwargs.get('reject_stats')
    local_segments: list[ActionSegment] = list(kwargs.get('local_segments') or [])
    if direction == 'next':
        if idx + 1 >= len(segments):
            return None
        current, neighbor = (segments[idx], segments[idx + 1])
        expected_gap = 1
    else:
        if idx - 1 < 0:
            return None
        current, neighbor = (segments[idx], segments[idx - 1])
        expected_gap = -1
    current_pos = next((i for i, seg in enumerate(segments) if seg.uid == current.uid), None)
    neighbor_pos = next((i for i, seg in enumerate(segments) if seg.uid == neighbor.uid), None)
    if current_pos is None or neighbor_pos is None or neighbor_pos - current_pos != expected_gap:
        if reject_stats is not None:
            reject_stats['rejected_non_adjacent_pairs'] = reject_stats.get('rejected_non_adjacent_pairs', 0) + 1
        return None
    chrono_pair = sort_segments_chronologically([current, neighbor])
    if not is_local_window(chrono_pair, 8.0):
        return None
    if pair_overlap_too_large(chrono_pair[0], chrono_pair[1]):
        if reject_stats is not None:
            reject_stats['rejected_temporal_overlap'] += 1
        return None
    if canonicalize_action(current.verb, current.noun) == canonicalize_action(neighbor.verb, neighbor.noun):
        if reject_stats is not None:
            reject_stats['rejected_duplicate_choices'] += 1
        return None
    clip_label = kwargs.get('clip_id') or f'{current.video_id}_before_or_after_action_{chrono_pair[0].uid}_{chrono_pair[-1].uid}'
    clip_path, shared_frames = _materialize_multi(label=clip_label, window=chrono_pair, video_path=kwargs['video_path'], clips_dir=kwargs['clips_dir'], frames_dir=kwargs['frames_dir'], per_segment_frames=2, pad_before=kwargs['pad_before'], pad_after=kwargs['pad_after'], clip_duration_limit=kwargs['clip_duration_limit'], cache=kwargs['multi_cache'] if 'multi_cache' in kwargs else {}, existing_clip_path=kwargs.get('existing_clip_path'), clip_start_sec=kwargs.get('clip_start_sec'), clip_end_sec=kwargs.get('clip_end_sec'))
    earlier, later = (chrono_pair[0], chrono_pair[1])
    earlier_frames = shared_frames[0:2]
    later_frames = shared_frames[2:4]
    current_frames = earlier_frames if current.uid == earlier.uid else later_frames
    neighbor_frames = earlier_frames if neighbor.uid == earlier.uid else later_frames
    distractor_segs = generate_unique_action_distractors(gold_seg=neighbor, exclude_uids={current.uid, neighbor.uid}, local_segments=local_segments or segments, all_segments=segments)
    if distractor_segs is None:
        if reject_stats is not None:
            reject_stats['rejected_insufficient_distractors'] += 1
        return None
    current_label = display_action(current)
    neighbor_label = display_action(neighbor)
    earlier_label = display_action(earlier)
    later_label = display_action(later)
    gold_text = neighbor_label
    distractor_texts = [display_action(seg) for seg in distractor_segs]
    if not choices_are_unique([gold_text] + distractor_texts):
        if reject_stats is not None:
            reject_stats['rejected_duplicate_choices'] += 1
        return None
    seed = str(kwargs.get('record_id') or kwargs.get('clip_id') or f'{clip_label}_{direction}')
    mapping, answer_key = shuffle_choices_and_update_key(gold_text, distractor_texts, seed)
    assert mapping[answer_key] == gold_text
    if direction == 'next':
        question = f'What happens immediately after "{current_label}"?'
        relation_step = f'Therefore, "{neighbor_label}" occurs immediately after "{current_label}".'
        rule = 'next_segment_uid_equals'
        direction_label = 'after'
    else:
        question = f'What happens immediately before "{current_label}"?'
        relation_step = f'Therefore, "{neighbor_label}" occurs immediately before "{current_label}".'
        rule = 'previous_segment_uid_equals'
        direction_label = 'before'
    gold_trace = [make_step(1, 'visual_observation', ordered_observation('First', earlier_label), _segment_payload(earlier, earlier_frames), rule if earlier.uid == neighbor.uid else 'segment_uid_equals', neighbor.uid if earlier.uid == neighbor.uid else earlier.uid, verifier_step=with_frame_prefix(earlier_label, 1, 2)), make_step(2, 'visual_observation', ordered_observation('Next', later_label), _segment_payload(later, later_frames), rule if later.uid == neighbor.uid else 'segment_uid_equals', neighbor.uid if later.uid == neighbor.uid else later.uid, verifier_step=with_frame_prefix(later_label, 3, 4)), make_step(3, 'temporal_relation', relation_step, {'candidate_actions': mapping, 'direction': direction, 'query_direction': direction_label, 'anchor_segment_uid': current.uid, 'adjacent_segment_uid': neighbor.uid, 'relation_rule': rule}, 'choice_action_equals', answer_key), make_step(4, 'answer_grounding', f'Therefore, the correct answer is {answer_key}.', {'candidate_actions': mapping}, 'choice_action_equals', answer_key)]
    record = make_record(record_id=kwargs.get('record_id') or f"{current.video_id}_before_or_after_action_{kwargs['sample_idx']:03d}", video_id=current.video_id, annotation_path=kwargs['annotations_path'], task_type='before_or_after_action', question=question, choices=mapping, answer_key=answer_key, answer=gold_text, gold_trace=gold_trace, grounding={'anchor_segment': _segment_payload(current, current_frames), 'adjacent_segment': _segment_payload(neighbor, neighbor_frames), 'clip_path': clip_path, 'clip_id': kwargs.get('clip_id'), 'direction': direction, 'query_direction': direction_label, 'shared_support_frames': shared_frames, 'candidate_actions': mapping}, quality_score=max(1.5, abs(neighbor.start_sec - current.stop_sec) + 1.0), quality_notes=['adjacent action with hard distractors sampled from real but non-adjacent events'])
    if kwargs.get('source_video_path'):
        record['source_video_path'] = str(kwargs['source_video_path'])
    ok, reasons = validate_temporal_instance(record)
    if not ok:
        if reject_stats is not None:
            reject_stats['rejected_validation'] += 1
            if isinstance(reject_stats.get('validation_failures'), list):
                reject_stats['validation_failures'].append({'id': record['id'], 'reasons': reasons})
            else:
                reject_stats['validation_failures'] = [{'id': record['id'], 'reasons': reasons}]
        print(f"[skip] before_or_after_action validation failed for {record['id']}: {reasons}")
        return None
    return record

def segment_from_dict(row: dict[str, Any]) -> ActionSegment:
    return ActionSegment(uid=str(row['uid']), video_id=str(row['video_id']), participant_id=str(row.get('participant_id') or str(row['video_id']).split('_')[0]), narration=str(row.get('narration') or ''), verb=str(row.get('verb') or ''), verb_class=int(row.get('verb_class') or -1), noun=str(row.get('noun') or ''), noun_class=int(row.get('noun_class') or -1), start_frame=int(row.get('start_frame') or 0), stop_frame=int(row.get('stop_frame') or 0), start_timestamp=str(row.get('start_timestamp') or '00:00:00.00'), stop_timestamp=str(row.get('stop_timestamp') or '00:00:00.00'), start_sec=float(row['start_sec']), stop_sec=float(row['stop_sec']))

def generate_records_from_clips(*, clip_rows: list[dict[str, Any]], annotations_path: Path, all_segments_by_video: dict[str, list[ActionSegment]], frames_dir: Path, frames_per_segment: int, pad_before: float, pad_after: float, clip_duration_limit: float | None, max_records: int | None=None, reject_stats: dict[str, int] | None=None) -> list[dict[str, Any]]:
    """From each local clip, try both event_order and before_or_after when eligible."""
    records: list[dict[str, Any]] = []
    multi_cache: dict[str, tuple[str, list[str]]] = {}
    counters = {'event_order': 0, 'before_or_after_action': 0}
    stats = reject_stats if reject_stats is not None else empty_reject_stats()
    if 'validation_failures' not in stats:
        stats['validation_failures'] = []
    for clip in clip_rows:
        if max_records is not None and len(records) >= max_records:
            break
        window = [segment_from_dict(row) for row in clip.get('segments') or [] if isinstance(row, dict)]
        if not window:
            continue
        window = sort_segments_chronologically(window)
        video_id = str(clip.get('video_id') or window[0].video_id)
        source_video = Path(str(clip.get('source_video_path') or ''))
        clip_path = Path(str(clip.get('clip_path') or ''))
        clips_dir = clip_path.parent
        common = {'video_path': source_video if source_video.exists() else clip_path, 'annotations_path': annotations_path, 'clips_dir': clips_dir, 'frames_dir': frames_dir, 'pad_before': pad_before, 'pad_after': pad_after, 'clip_duration_limit': clip_duration_limit, 'multi_cache': multi_cache, 'existing_clip_path': clip_path, 'clip_start_sec': clip.get('start_sec'), 'clip_end_sec': clip.get('end_sec'), 'clip_id': clip.get('clip_id'), 'source_video_path': source_video, 'frames_per_segment': frames_per_segment, 'reject_stats': stats, 'local_segments': window}
        order_window = _pick_event_order_window(window)
        if order_window is None and len(window) >= 4:
            for start in range(0, len(window) - 3):
                candidate = window[start:start + 4]
                if distinct_labels(candidate) and is_local_window(candidate, 18.0) and has_ambiguous_overlap(candidate):
                    stats['rejected_temporal_overlap'] += 1
                    break
        if order_window is not None:
            if max_records is not None and len(records) >= max_records:
                break
            stats['generated'] += 1
            counters['event_order'] += 1
            row = make_event_order_record(window=order_window, sample_idx=counters['event_order'], record_id=f"{video_id}_event_order_{counters['event_order']:03d}", **common)
            if row is not None:
                records.append(row)
                stats['accepted'] += 1
        if len(window) >= 2:
            segments = all_segments_by_video.get(video_id) or window
            pair = window[:2]
            for direction, anchor_uid in (('next', pair[0].uid), ('previous', pair[1].uid)):
                if max_records is not None and len(records) >= max_records:
                    break
                current_idx = next((i for i, seg in enumerate(segments) if seg.uid == anchor_uid), None)
                if current_idx is None:
                    continue
                stats['generated'] += 1
                counters['before_or_after_action'] += 1
                row = maybe_make_conditional_adjacent_record(idx=current_idx, direction=direction, segments=segments, sample_idx=counters['before_or_after_action'], record_id=f"{video_id}_before_or_after_action_{counters['before_or_after_action']:03d}", **common)
                if row is not None:
                    records.append(row)
                    stats['accepted'] += 1
    out = dedupe_records(records)
    if max_records is not None:
        out = out[:max_records]
    if reject_stats is not None:
        reject_stats.update(stats)
    return out

def _pick_event_order_window(window: list[ActionSegment]) -> list[ActionSegment] | None:
    if len(window) < 4:
        return None
    for start in range(0, len(window) - 3):
        candidate = sort_segments_chronologically(window[start:start + 4])
        if distinct_labels(candidate) and is_local_window(candidate, 18.0) and (not has_ambiguous_overlap(candidate)):
            return candidate
    return None

def maybe_make_constrained_count_record(**kwargs) -> dict[str, Any] | None:
    idx: int = kwargs['idx']
    segments: list[ActionSegment] = kwargs['segments']
    window = segments[idx:min(len(segments), idx + 6)]
    if len(window) < 4 or not is_local_window(window, 14.0):
        return None
    best_verb = None
    best_matches: list[ActionSegment] = []
    for verb in {seg.verb for seg in window}:
        matches = [seg for seg in window if seg.verb == verb]
        if 2 <= len(matches) <= 4 and len(matches) > len(best_matches):
            best_verb = verb
            best_matches = matches
    if best_verb is None:
        return None
    clip_label = f'{window[0].video_id}_constrained_count_{window[0].uid}_{window[-1].uid}'
    clip_path, support_frames = _materialize_multi(label=clip_label, window=window, video_path=kwargs['video_path'], clips_dir=kwargs['clips_dir'], frames_dir=kwargs['frames_dir'], per_segment_frames=2, pad_before=kwargs['pad_before'], pad_after=kwargs['pad_after'], clip_duration_limit=max(kwargs['clip_duration_limit'] or 0, 14.0) if kwargs['clip_duration_limit'] else 14.0, cache=kwargs['multi_cache'])
    choices = {'A': '1 time', 'B': '2 times', 'C': '3 times', 'D': '4 or more times'}
    count = len(best_matches)
    answer_key = 'A' if count == 1 else 'B' if count == 2 else 'C' if count == 3 else 'D'
    frame_groups = {}
    for local_idx, seg in enumerate(window):
        frame_groups[seg.uid] = support_frames[local_idx * 2:(local_idx + 1) * 2]
    evidence_rows = [_segment_payload(seg, frame_groups[seg.uid]) for seg in best_matches]
    gold_trace = [make_step(1, 'count_grounding', f"In this short clip, the person performs '{best_verb}' on {count} separate occasions before the interaction sequence moves on.", {'matching_segments': evidence_rows, 'clip_path': clip_path}, 'matching_verb_count', count), make_step(2, 'instance_alignment', 'The support frames are grouped by occurrence: each pair of consecutive frames marks one separate instance, so occurrences must be counted rather than merged.', {'support_frame_groups': {seg.uid: frame_groups[seg.uid] for seg in best_matches}}, 'frame_group_count', count), make_step(3, 'answer_grounding', f"Using the grouped support frames, count the separate '{best_verb}' events within this bounded clip and choose the matching total.", {'candidate_counts': choices, 'count': count}, 'choice_count_equals', answer_key)]
    return make_record(record_id=f"{window[0].video_id}_constrained_action_count_{kwargs['sample_idx']:03d}", video_id=window[0].video_id, annotation_path=kwargs['annotations_path'], task_type='constrained_action_count', question=f'In this short clip, before the activity shifts away, how many separate times does the person {best_verb} something?', choices=choices, answer_key=answer_key, answer=choices[answer_key], gold_trace=gold_trace, grounding={'window_segments': [_segment_payload(seg, frame_groups[seg.uid]) for seg in window], 'matching_segments': evidence_rows, 'clip_path': clip_path, 'shared_support_frames': support_frames}, quality_score=count + 1.5, quality_notes=['count computed from a tightly bounded local clip with per-instance support frames'])

def maybe_make_appearance_order_record(**kwargs) -> dict[str, Any] | None:
    idx: int = kwargs['idx']
    segments: list[ActionSegment] = kwargs['segments']
    window = segments[idx:min(len(segments), idx + 6)]
    if len(window) < 4 or not is_local_window(window, 16.0):
        return None
    noun_segments: list[ActionSegment] = []
    seen = set()
    for seg in window:
        if seg.noun in seen:
            continue
        seen.add(seg.noun)
        noun_segments.append(seg)
        if len(noun_segments) == 4:
            break
    if len(noun_segments) < 4:
        return None
    clip_label = f'{window[0].video_id}_appearance_order_{window[0].uid}_{noun_segments[-1].uid}'
    clip_path, support_frames = _materialize_multi(label=clip_label, window=noun_segments, video_path=kwargs['video_path'], clips_dir=kwargs['clips_dir'], frames_dir=kwargs['frames_dir'], per_segment_frames=2, pad_before=kwargs['pad_before'], pad_after=kwargs['pad_after'], clip_duration_limit=max(kwargs['clip_duration_limit'] or 0, 16.0) if kwargs['clip_duration_limit'] else 16.0, cache=kwargs['multi_cache'])
    nouns = [seg.noun for seg in noun_segments]
    correct = tuple(nouns)
    distractors = [(nouns[0], nouns[2], nouns[1], nouns[3]), (nouns[1], nouns[0], nouns[2], nouns[3]), (nouns[0], nouns[1], nouns[3], nouns[2])]
    mapping = {letter: ', '.join(order) for letter, order in zip(['A', 'B', 'C', 'D'], [correct] + distractors)}
    frame_groups = [support_frames[i * 2:(i + 1) * 2] for i in range(4)]
    gold_trace = [make_step(1, 'appearance_grounding', f'First, the first new task-relevant object to appear is {nouns[0]}.', _segment_payload(noun_segments[0], frame_groups[0]), 'segment_noun_equals', nouns[0], verifier_step=with_frame_prefix(f'the first new task-relevant object to appear is {nouns[0]}.', 1, 2)), make_step(2, 'appearance_grounding', f'Next, {nouns[1]} appears before {nouns[2]}.', {'second_segment': _segment_payload(noun_segments[1], frame_groups[1]), 'third_segment': _segment_payload(noun_segments[2], frame_groups[2])}, 'ordered_noun_pair_equals', [nouns[1], nouns[2]], verifier_step=with_frame_prefix(f'{nouns[1]} appears before {nouns[2]}.', 3, 6)), make_step(3, 'appearance_grounding', f'Finally, the last of the listed objects to appear is {nouns[3]}.', _segment_payload(noun_segments[3], frame_groups[3]), 'segment_noun_equals', nouns[3], verifier_step=with_frame_prefix(f'the last of the listed objects to appear is {nouns[3]}.', 7, 8)), make_step(4, 'answer_grounding', 'Therefore, choose the option whose full object appearance order matches the clip.', {'candidate_orders': mapping}, 'choice_sequence_equals', 'A')]
    return make_record(record_id=f"{window[0].video_id}_event_order_appearance_{kwargs['sample_idx']:03d}", video_id=window[0].video_id, annotation_path=kwargs['annotations_path'], task_type='event_order', question=f"What is the first-time appearance order of the following objects in this short clip: {', '.join(nouns)}?", choices=mapping, answer_key='A', answer=mapping['A'], gold_trace=gold_trace, grounding={'ordered_noun_segments': [_segment_payload(seg, frames) for seg, frames in zip(noun_segments, frame_groups)], 'clip_path': clip_path, 'shared_support_frames': support_frames, 'note': 'appearance order is approximated by first local action-linked visibility of each noun'}, quality_score=4.5, quality_notes=['appearance order approximated from first local action-linked visibility of each noun'])

def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in records:
        key = (row['task_type'], row['question'], tuple(sorted((row.get('choices') or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def build_qc_summary(records: list[dict[str, Any]], reject_stats: dict[str, Any]) -> dict[str, Any]:
    """Dataset-level QC summary for Temporal Event Reasoning outputs."""
    import statistics
    task_counts: dict[str, int] = {}
    key_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    before_after = {'before': 0, 'after': 0, 'previous': 0, 'next': 0}
    quality_by_task: dict[str, list[float]] = {}
    videos: set[str] = set()
    clips: set[str] = set()
    unique_choice_ok = 0
    for row in records:
        tt = row.get('task_type') or 'unknown'
        task_counts[tt] = task_counts.get(tt, 0) + 1
        ak = row.get('answer_key')
        if ak in key_counts:
            key_counts[ak] += 1
        qscore = float(row.get('quality_score') or 0.0)
        quality_by_task.setdefault(tt, []).append(qscore)
        videos.add(str(row.get('video_id') or ''))
        g = row.get('grounding') or {}
        if g.get('clip_id'):
            clips.add(str(g['clip_id']))
        choices = list((row.get('choices') or {}).values())
        if len(choices) == 4 and choices_are_unique([str(c) for c in choices]):
            unique_choice_ok += 1
        if tt == 'before_or_after_action':
            direction = g.get('direction')
            qdir = g.get('query_direction')
            if direction in before_after:
                before_after[direction] += 1
            if qdir in before_after:
                before_after[qdir] += 1
    n = max(len(records), 1)
    key_pct = {k: round(100.0 * v / n, 2) for k, v in key_counts.items()}
    warnings: list[str] = []
    for k, pct in key_pct.items():
        if pct > 35.0 and len(records) >= 8:
            warnings.append(f'answer_key_{k}_imbalanced:{pct}%')
    quality_summary = {}
    for tt, vals in quality_by_task.items():
        quality_summary[tt] = {'avg': round(statistics.mean(vals), 4) if vals else None, 'median': round(statistics.median(vals), 4) if vals else None, 'n': len(vals)}
    return {'num_records': len(records), 'task_counts': task_counts, 'answer_key_counts': key_counts, 'answer_key_percentages': key_pct, 'unique_choice_valid_pct': round(100.0 * unique_choice_ok / n, 2), 'reject_stats': {'generated': reject_stats.get('generated', 0), 'accepted': reject_stats.get('accepted', 0), 'rejected_duplicate_choices': reject_stats.get('rejected_duplicate_choices', 0), 'rejected_temporal_overlap': reject_stats.get('rejected_temporal_overlap', 0), 'rejected_insufficient_distractors': reject_stats.get('rejected_insufficient_distractors', 0), 'rejected_validation': reject_stats.get('rejected_validation', 0), 'rejected_other': reject_stats.get('rejected_other', 0)}, 'quality_score_by_task': quality_summary, 'before_after_counts': before_after, 'num_distinct_videos': len({v for v in videos if v}), 'num_distinct_clips': len({c for c in clips if c}), 'warnings': warnings}
