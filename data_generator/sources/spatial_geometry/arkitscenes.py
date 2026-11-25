from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List
from .geometry import Object3D

def find_annotation_files(root: Path) -> List[Path]:
    if root.is_file():
        if root.name.endswith('_3dod_annotation.json'):
            return [root]
        raise ValueError(f'Expected an ARKitScenes *_3dod_annotation.json file, got: {root}')
    for depth in range(1, 5):
        pattern = '/'.join(['*'] * (depth - 1) + ['*_3dod_annotation.json'])
        found = sorted(root.glob(pattern))
        if found:
            return found
    return []

def load_arkitscenes_objects(annotation_path: Path) -> List[Object3D]:
    data = json.loads(annotation_path.read_text())
    objects: list[Object3D] = []
    for idx, raw in enumerate(data.get('data', [])):
        label = str(raw.get('label', 'unknown')).strip()
        uid = str(raw.get('uid', ''))
        seg = raw.get('segments', {})
        obb = seg.get('obbAligned') or seg.get('obb')
        if not obb:
            continue
        centroid = tuple((float(x) for x in obb.get('centroid', [])))
        axes_lengths = tuple((float(x) for x in obb.get('axesLengths', [])))
        normalized_axes = tuple((float(x) for x in obb.get('normalizedAxes', [])))
        if len(centroid) != 3 or len(axes_lengths) != 3 or len(normalized_axes) != 9:
            continue
        object_key = f'obj_{idx:03d}_{label}'
        objects.append(Object3D(object_key=object_key, uid=uid, label=label, centroid=centroid, axes_lengths=axes_lengths, normalized_axes=normalized_axes))
    return objects

def scene_id_from_annotation(annotation_path: Path) -> str:
    name = annotation_path.name
    if name.endswith('_3dod_annotation.json'):
        return name.replace('_3dod_annotation.json', '')
    return annotation_path.parent.name

def load_scenes(data_root: Path) -> Iterable[tuple[str, Path, list[Object3D]]]:
    for ann_path in find_annotation_files(data_root):
        scene_id = scene_id_from_annotation(ann_path)
        objects = load_arkitscenes_objects(ann_path)
        if objects:
            yield (scene_id, ann_path, objects)
