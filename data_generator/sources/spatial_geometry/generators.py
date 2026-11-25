from __future__ import annotations
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
from .geometry import Object3D, category_counts, centroid_distance, classify_ego_quadrant, round_float, rounded_list
from .qa_schema import make_aligned_step, make_record
CHOICE_LETTERS = ['A', 'B', 'C', 'D']
DEFAULT_TASK_QUOTAS = {'object_count': 3, 'relative_direction': 9}

def _choice_map(options: Sequence[str]) -> dict[str, str]:
    return {letter: value for letter, value in zip(CHOICE_LETTERS, options)}

def _answer_key(choices: dict[str, str], answer: str) -> str:
    for k, v in choices.items():
        if v == answer:
            return k
    raise ValueError(f'Answer not found in choices: {answer}')

def _object_grounding(obj: Object3D) -> dict:
    return {'object_key': obj.object_key, 'label': obj.label, 'uid': obj.uid, 'centroid': rounded_list(obj.centroid), 'axes_lengths_m': rounded_list(obj.axes_lengths)}

def _display_name(obj: Object3D, label_counts: dict[str, int]) -> str:
    if label_counts.get(obj.label, 0) > 1:
        return f'{obj.label} ({obj.object_key})'
    return obj.label

def _label_text(label: str) -> str:
    return label.replace('_', ' ')

def _display_names(objects: Sequence[Object3D]) -> dict[str, str]:
    label_counts = {label: count for label, count in category_counts(objects).items()}
    unique_anchors = [obj for obj in objects if label_counts.get(obj.label, 0) == 1]
    names: dict[str, str] = {}
    for obj in objects:
        if label_counts.get(obj.label, 0) == 1:
            names[obj.object_key] = _label_text(obj.label)
            continue
        if unique_anchors:
            anchor = min(unique_anchors, key=lambda other: centroid_distance(obj, other))
            names[obj.object_key] = f'{_label_text(obj.label)} near the {_label_text(anchor.label)}'
        else:
            names[obj.object_key] = _label_text(obj.label)
    seen: dict[str, int] = {}
    for obj in objects:
        name = names[obj.object_key]
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            names[obj.object_key] = _insert_instance_suffix(name, seen[name])
    return names

def _name(obj: Object3D, display_names: dict[str, str]) -> str:
    return display_names.get(obj.object_key, _label_text(obj.label))

def _insert_instance_suffix(name: str, instance_index: int) -> str:
    suffix = f' #{instance_index}'
    if ' near the ' in name:
        head, tail = name.split(' near the ', 1)
        return f'{head}{suffix} near the {tail}'
    return f'{name}{suffix}'

def _with_quality(record: dict, score: float, notes: list[str]) -> dict:
    record['quality_score'] = round_float(score)
    record['quality_notes'] = notes
    return record

def _aligned_trace(model_steps: list[str], verifier_specs: list[dict]) -> list[dict]:
    if len(model_steps) != len(verifier_specs):
        raise ValueError('model_steps and verifier_specs must have the same length.')
    out = []
    for idx, (model_step, spec) in enumerate(zip(model_steps, verifier_specs), start=1):
        out.append(make_aligned_step(idx, spec['claim_type'], model_step, spec['verifier_step'], spec.get('evidence', {}), spec['rule'], spec['expected']))
    return out

def generate_object_count(scene_id: str, annotation_path: Path, objects: Sequence[Object3D], max_per_scene: int, rng: random.Random, frame_evidence: dict[str, dict] | None=None) -> list[dict]:
    counts = category_counts(objects)
    labels = [label for label, n in counts.items() if n > 0]
    rng.shuffle(labels)
    records = []
    for label in labels[:max_per_scene]:
        count = counts[label]
        options = sorted(set([max(0, count - 1), count, count + 1, count + 2]))
        while len(options) < 4:
            options.append(max(options) + 1)
        options = [str(x) for x in options[:4]]
        rng.shuffle(options)
        choices = _choice_map(options)
        answer = str(count)
        rec_id = f'{scene_id}_object_count_{len(records):04d}_{label}'
        matching = [obj for obj in objects if obj.label == label]
        frame_evidence = frame_evidence or {}
        matching_frame_evidence = [frame_evidence[obj.object_key] for obj in matching if obj.object_key in frame_evidence]
        if frame_evidence and len(matching_frame_evidence) < len(matching):
            continue
        if matching_frame_evidence:
            frame_steps, frame_specs = _frame_steps_and_specs_for_count(label, matching, matching_frame_evidence)
            gold_reasoning = [f'Scan the sampled video frames for {_label_text(label)} objects.', *frame_steps, _repeated_view_count_step(label, count), f'No additional distinct {_label_text(label)} object is observed beyond these instances.', f'Counting the distinct {_label_text(label)} objects gives {count}.']
            verifier_specs = [{'claim_type': 'video_scan', 'verifier_step': f"Use frame evidence to scan for visible '{label}' instances.", 'evidence': {'category': label, 'num_frame_evidence_objects': len(matching_frame_evidence)}, 'rule': 'frame_evidence_available_for_category', 'expected': True}, *frame_specs, {'claim_type': 'object_identity', 'verifier_step': 'Group multiple visible frames with the same object key as repeated views of one physical object.', 'evidence': {'object_keys': [obj.object_key for obj in matching], 'frame_evidence': matching_frame_evidence}, 'rule': 'same_object_key_repeated_views_not_new_instances', 'expected': True}, {'claim_type': 'object_exclusion', 'verifier_step': f"Verify that the unique annotated '{label}' object ids are exactly the counted instances.", 'evidence': {'object_keys': [obj.object_key for obj in matching], 'count': count}, 'rule': 'no_extra_unique_instance_ids(category)', 'expected': [obj.object_key for obj in matching]}]
            verifier_specs.extend([{'claim_type': 'count', 'verifier_step': f'Count the filtered unique instance ids; len={count}.', 'evidence': {'category': label, 'count': count}, 'rule': 'len(unique_instance_ids(category))', 'expected': count}])
        else:
            gold_reasoning = [f'Identify the objects in the room that belong to the {label} category.', 'Treat repeated views of the same physical object as one object.', f'Counting the unique {_label_text(label)} objects gives {count}.']
            verifier_specs = [{'claim_type': 'object_identification', 'verifier_step': f"Filter all annotated 3D objects whose label equals '{label}', and attach frame evidence when available.", 'evidence': {'objects': [_object_grounding(obj) for obj in matching], 'frame_evidence': matching_frame_evidence}, 'rule': 'filter_objects_by_category', 'expected': label}, {'claim_type': 'object_identity', 'verifier_step': 'Use unique ARKitScenes object keys so duplicate views are not double-counted.', 'evidence': {'object_keys': [obj.object_key for obj in matching]}, 'rule': 'unique_instance_ids(category)', 'expected': [obj.object_key for obj in matching]}, {'claim_type': 'count', 'verifier_step': f'Count the filtered unique instance ids; len={count}.', 'evidence': {'category': label, 'count': count}, 'rule': 'len(unique_instance_ids(category))', 'expected': count}]
        gold_trace = _aligned_trace(gold_reasoning, verifier_specs)
        score = 0.7 + min(count, 4) * 0.05
        records.append(_with_quality(make_record(record_id=rec_id, scene_id=scene_id, annotation_path=annotation_path, task_type='object_count', question=f'How many {_label_text(label)} objects are in this room?', choices=choices, answer_key=_answer_key(choices, answer), answer=answer, gold_trace=gold_trace, gold_reasoning=gold_reasoning, grounding={'category_counts': counts}, difficulty='easy'), score, [f'count={count}', 'Higher score if the category has multiple instances.']))
    return records

def generate_object_relative_direction(scene_id: str, annotation_path: Path, objects: Sequence[Object3D], max_per_scene: int, rng: random.Random, min_component_margin: float, display_names: dict[str, str] | None=None, frame_evidence: dict[str, dict] | None=None, require_role_frame_evidence: bool=True, min_role_bbox_area_ratio: float=0.003, min_role_visible_frames: int=2) -> list[dict]:
    records = []
    triples = list(itertools.permutations(objects, 3))
    rng.shuffle(triples)
    labels_by_category = defaultdict(int)
    for obj in objects:
        labels_by_category[obj.label] += 1
    display_names = display_names or _display_names(objects)
    frame_evidence = frame_evidence or {}
    for observer, facing, target in triples:
        if observer.label == facing.label == target.label:
            continue
        try:
            geom = classify_ego_quadrant(observer, facing, target, min_component_margin=min_component_margin)
        except ValueError:
            continue
        if geom['quadrant'] == 'ambiguous':
            continue
        observer_name = _name(observer, display_names)
        facing_name = _name(facing, display_names)
        target_name = _name(target, display_names)
        choices = _choice_map(['front-left', 'front-right', 'back-left', 'back-right'])
        answer = geom['quadrant']
        rec_id = f'{scene_id}_relative_direction_{len(records):04d}_{observer.object_key}_{facing.object_key}_{target.object_key}'
        component_margin = min(abs(geom['forward_component']), abs(geom['right_component']))
        observer_frame = frame_evidence.get(observer.object_key)
        facing_frame = frame_evidence.get(facing.object_key)
        target_frame = frame_evidence.get(target.object_key)
        role_frames = [observer_frame, facing_frame, target_frame]
        if require_role_frame_evidence and (not all((isinstance(row, dict) for row in role_frames))):
            continue
        if require_role_frame_evidence and (not all((_frame_evidence_meets_threshold(row, min_bbox_area_ratio=min_role_bbox_area_ratio, min_visible_frames=min_role_visible_frames) for row in role_frames))):
            continue
        front_back = 'front' if geom['forward_component'] >= 0 else 'back'
        left_right = 'right' if geom['right_component'] >= 0 else 'left'
        gold_reasoning = [_canonical_pose_reasoning_step(observer_name, observer_frame, 'standing point'), _canonical_pose_reasoning_step(facing_name, facing_frame, 'facing landmark'), _canonical_pose_reasoning_step(target_name, target_frame, 'target object'), f'Use the {observer_name} as the standing point and face directly toward the center of the {facing_name}.', f'The {target_name} lies on the {front_back} side of your facing direction.', f'It also lies to your {left_right} side.', f'Combining these two cues, the {target_name} is {answer}.']
        gold_trace = _aligned_trace(gold_reasoning, [{'claim_type': 'object_identification', 'verifier_step': f'Set observer origin to object id {observer.object_key}.', 'evidence': {'observer_object': _object_grounding(observer), 'frame_evidence': observer_frame}, 'rule': 'observer_origin', 'expected': observer.object_key}, {'claim_type': 'spatial_frame', 'verifier_step': 'Compute forward vector from observer centroid to facing-object centroid.', 'evidence': {'observer_object': observer.object_key, 'facing_object': facing.object_key, 'observer_frame_evidence': observer_frame, 'facing_frame_evidence': facing_frame, 'forward_vector': [round_float(x) for x in geom['forward_vector']]}, 'rule': 'forward_vector = facing.centroid - observer.centroid', 'expected': [round_float(x) for x in geom['forward_vector']]}, {'claim_type': 'object_identification', 'verifier_step': 'Compute target vector from observer centroid to target centroid.', 'evidence': {'target_object': _object_grounding(target), 'target_frame_evidence': target_frame, 'target_vector': [round_float(x) for x in geom['target_vector']]}, 'rule': 'target_vector = target.centroid - observer.centroid', 'expected': [round_float(x) for x in geom['target_vector']]}, {'claim_type': 'spatial_frame', 'verifier_step': f'Anchor the egocentric pose at {observer.object_key} and orient it toward the centroid of {facing.object_key}.', 'evidence': {'observer_object': observer.object_key, 'facing_object': facing.object_key, 'forward_vector': [round_float(x) for x in geom['forward_vector']]}, 'rule': 'ego_pose(origin=observer.centroid, forward=facing.centroid-observer.centroid)', 'expected': {'observer_object': observer.object_key, 'facing_object': facing.object_key}}, {'claim_type': 'spatial_relation', 'verifier_step': f"Project target vector onto forward direction; component={geom['forward_component']:.4f}, side={front_back}.", 'evidence': {'forward_component': round_float(geom['forward_component'])}, 'rule': 'front if dot(target_vector, forward_unit) >= 0 else back', 'expected': front_back}, {'claim_type': 'spatial_relation', 'verifier_step': f"Project target vector onto right direction; component={geom['right_component']:.4f}, side={left_right}.", 'evidence': {'right_component': round_float(geom['right_component'])}, 'rule': 'right if dot(target_vector, right_unit) >= 0 else left', 'expected': left_right}, {'claim_type': 'answer_grounding', 'verifier_step': f'Combine front/back={front_back} and left/right={left_right}; quadrant={answer}.', 'evidence': {'front_back': front_back, 'left_right': left_right, 'quadrant': answer, 'min_abs_component': round_float(component_margin)}, 'rule': 'quadrant(sign(forward_component), sign(right_component))', 'expected': answer}])
        score = min(1.0, 0.6 + component_margin / 2.0)
        records.append(_with_quality(make_record(record_id=rec_id, scene_id=scene_id, annotation_path=annotation_path, task_type='relative_direction', question=_canonical_pose_question(observer_name, facing_name, target_name), choices=choices, answer_key=_answer_key(choices, answer), answer=answer, gold_trace=gold_trace, gold_reasoning=gold_reasoning, grounding={'observer_object': _object_grounding(observer), 'facing_object': _object_grounding(facing), 'target_object': _object_grounding(target), 'geometry': {k: round_float(v) if isinstance(v, float) else rounded_list(v) if isinstance(v, list) else v for k, v in geom.items()}, 'min_component_margin': min_component_margin, 'note': 'Direction is computed in the horizontal x-y plane from ARKitScenes 3D box centroids.'}, difficulty='hard'), score, [f'min_abs_component={component_margin:.4f}', 'Higher score when target is far from front/back and left/right decision boundaries.']))
        if len(records) >= max_per_scene:
            break
    return records

def select_top_k_records(records: Sequence[dict], *, top_k: int | None, task_quotas: dict[str, int] | None=None) -> list[dict]:
    if top_k is None or top_k <= 0:
        return list(records)
    quotas = task_quotas or DEFAULT_TASK_QUOTAS
    quotas = _scale_quotas(quotas, top_k)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_task[rec['task_type']].append(rec)
    selected: list[dict] = []
    selected_ids = set()
    for task_type, quota in quotas.items():
        ranked = sorted(by_task.get(task_type, []), key=lambda r: (r.get('quality_score', 0.0), r['id']), reverse=True)
        for rec in ranked[:quota]:
            selected.append(rec)
            selected_ids.add(rec['id'])
    if len(selected) < top_k:
        remaining = [rec for rec in records if rec['id'] not in selected_ids]
        remaining = sorted(remaining, key=lambda r: (r.get('quality_score', 0.0), r['id']), reverse=True)
        selected.extend(remaining[:top_k - len(selected)])
    if len(selected) > top_k:
        selected = sorted(selected, key=lambda r: (r.get('quality_score', 0.0), r['id']), reverse=True)[:top_k]
    selected = sorted(selected, key=lambda r: (r['scene_id'], r['task_type'], -r.get('quality_score', 0.0), r['id']))
    return selected

def _scale_quotas(quotas: dict[str, int], top_k: int) -> dict[str, int]:
    quota_sum = sum(quotas.values())
    if top_k >= quota_sum:
        return dict(quotas)
    scaled: dict[str, int] = {}
    remainders = []
    for task_type, quota in quotas.items():
        raw = quota / quota_sum * top_k
        base = int(raw)
        if quota > 0 and base == 0:
            base = 1
        scaled[task_type] = base
        remainders.append((raw - int(raw), task_type))
    while sum(scaled.values()) > top_k:
        candidates = sorted([(scaled[t], quotas[t], t) for t in scaled if scaled[t] > 1], reverse=True)
        if not candidates:
            break
        _, _, task_type = candidates[0]
        scaled[task_type] -= 1
    while sum(scaled.values()) < top_k:
        for _, task_type in sorted(remainders, reverse=True):
            if sum(scaled.values()) >= top_k:
                break
            if scaled[task_type] < quotas[task_type]:
                scaled[task_type] += 1
    return scaled

def generate_scene_records(scene_id: str, annotation_path: Path, objects: Sequence[Object3D], *, seed: int, max_object_count: int, max_relative_direction: int, min_direction_component_margin: float, unique_label_objects_only: bool=True, frame_evidence: dict[str, dict] | None=None, require_role_frame_evidence: bool=True, min_role_bbox_area_ratio: float=0.003, min_role_visible_frames: int=2) -> list[dict]:
    rng = random.Random(f'{seed}:{scene_id}')
    records: list[dict] = []
    records.extend(generate_object_count(scene_id, annotation_path, objects, max_object_count, rng, frame_evidence=frame_evidence))
    object_source = objects
    if unique_label_objects_only and frame_evidence:
        object_source = [obj for obj in objects if obj.object_key in frame_evidence]
    object_level_pool = _one_object_per_label(object_source, frame_evidence) if unique_label_objects_only else list(objects)
    representative_names = _representative_display_names(objects, object_level_pool, frame_evidence) if unique_label_objects_only else None
    records.extend(generate_object_relative_direction(scene_id, annotation_path, object_level_pool, max_relative_direction, rng, min_component_margin=min_direction_component_margin, display_names=representative_names, frame_evidence=frame_evidence, require_role_frame_evidence=require_role_frame_evidence, min_role_bbox_area_ratio=min_role_bbox_area_ratio, min_role_visible_frames=min_role_visible_frames))
    return records

def _one_object_per_label(objects: Sequence[Object3D], frame_evidence: dict[str, dict] | None=None) -> list[Object3D]:
    frame_evidence = frame_evidence or {}
    by_label: dict[str, list[Object3D]] = defaultdict(list)
    for obj in objects:
        by_label[obj.label].append(obj)
    selected = []
    for label in sorted(by_label):
        candidates = sorted(by_label[label], key=lambda obj: (1 if obj.object_key in frame_evidence else 0, (frame_evidence.get(obj.object_key) or {}).get('bbox_area_ratio', 0.0), obj.longest_dimension_m), reverse=True)
        selected.append(candidates[0])
    return selected

def _representative_display_names(all_objects: Sequence[Object3D], selected_objects: Sequence[Object3D], frame_evidence: dict[str, dict] | None=None) -> dict[str, str]:
    frame_evidence = frame_evidence or {}
    counts = category_counts(all_objects)
    names = {}
    for obj in selected_objects:
        label = _label_text(obj.label)
        if counts.get(obj.label, 0) > 1:
            if obj.object_key in frame_evidence:
                names[obj.object_key] = _ordinal_object_name(obj, label, all_objects, frame_evidence)
            else:
                names[obj.object_key] = f'representative {label}'
        else:
            names[obj.object_key] = label
    return names

def _frame_steps_and_specs_for_count(label: str, matching_objects: Sequence[Object3D], frame_rows: list[dict]) -> tuple[list[str], list[dict]]:
    steps = []
    specs = []
    sorted_rows = _sorted_frame_rows(frame_rows)
    for idx, row in enumerate(sorted_rows, start=1):
        frame_id = row.get('frame_id')
        caption = row.get('semantic_caption')
        ordinal = _ordinal_word(idx) if len(sorted_rows) > 1 else 'visible'
        if frame_id is None:
            if len(sorted_rows) > 1:
                prefix = f'The {ordinal} {_label_text(label)} is visible in a sampled frame'
            else:
                prefix = f'A {_label_text(label)} is visible in a sampled frame'
        elif len(sorted_rows) > 1:
            prefix = f'In Frame {frame_id}, the {ordinal} {_label_text(label)} is visible'
        else:
            prefix = f'In Frame {frame_id}, a {_label_text(label)} is visible'
        steps.append(_caption_backed_step(prefix, caption))
        specs.append({'claim_type': 'object_identification', 'verifier_step': f"Use frame evidence for the {ordinal} annotated '{label}' object.", 'evidence': {'object': _object_grounding(_object_by_key(matching_objects, row.get('object_key'))), 'frame_evidence': row, 'supporting_visible_frames': row.get('visible_frame_samples', [])}, 'rule': 'frame_evidence_object_label', 'expected': label})
    return (steps, specs)

def _repeated_view_count_step(label: str, count: int) -> str:
    label_text = _label_text(label)
    if count == 1:
        return f'Later views of the same {label_text} are treated as repeated views, not new objects.'
    return f'Later views of the same {label_text} objects are grouped with those instances rather than counted as new objects.'

def _sorted_frame_rows(frame_rows: list[dict]) -> list[dict]:
    return sorted(frame_rows, key=lambda r: (r.get('frame_id') or 10 ** 9, r.get('object_key', '')))

def _object_by_key(objects: Sequence[Object3D], object_key: str | None) -> Object3D:
    for obj in objects:
        if obj.object_key == object_key:
            return obj
    raise ValueError(f'Object not found for frame evidence key: {object_key}')

def _caption_backed_step(prefix: str, caption: str | None) -> str:
    if not caption:
        return prefix + '.'
    cleaned = caption.strip().rstrip('.')
    return f'{prefix}: {cleaned}.'

def _ordinal_object_name(obj: Object3D, label_text: str, all_objects: Sequence[Object3D], frame_evidence: dict[str, dict]) -> str:
    same_label = [other for other in all_objects if other.label == obj.label and other.object_key in frame_evidence]
    same_label.sort(key=lambda other: (frame_evidence.get(other.object_key, {}).get('frame_id') or 10 ** 9, other.object_key))
    if obj in same_label:
        return f'{_ordinal_word(same_label.index(obj) + 1)} {label_text} to appear'
    return f'representative {label_text}'

def _ordinal_word(index: int) -> str:
    words = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
    return words.get(index, f'{index}th')

def _frame_reference_sentence(object_name: str, frame_row: dict | None, role: str) -> str:
    if frame_row and frame_row.get('frame_id') is not None:
        return f"Frame {frame_row['frame_id']} shows the {object_name}, which is the {role}."
    return f'Identify the {object_name} as the {role} in the video.'

def _canonical_pose_question(observer_name: str, facing_name: str, target_name: str) -> str:
    return f'If I stand at the position of the {observer_name} and face directly toward the center of the {facing_name}, is the {target_name} in the front-left, front-right, back-left, or back-right quadrant?'

def _canonical_pose_reasoning_step(object_name: str, frame_row: dict | None, role: str) -> str:
    if role == 'standing point':
        return f'Use the {object_name} as the standing point.'
    if role == 'facing landmark':
        return f'Use the {object_name} as the facing landmark and face directly toward its center.'
    return f'Use the {object_name} as the target object.'

def _frame_evidence_meets_threshold(frame_row: dict | None, *, min_bbox_area_ratio: float, min_visible_frames: int) -> bool:
    if not isinstance(frame_row, dict):
        return False
    bbox_area_ratio = frame_row.get('bbox_area_ratio')
    num_visible_frames = frame_row.get('num_visible_frames')
    try:
        bbox_value = float(bbox_area_ratio)
    except (TypeError, ValueError):
        return False
    try:
        visible_count = int(num_visible_frames)
    except (TypeError, ValueError):
        return False
    return bbox_value >= min_bbox_area_ratio and visible_count >= min_visible_frames
