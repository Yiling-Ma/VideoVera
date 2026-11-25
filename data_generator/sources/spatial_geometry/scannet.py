from __future__ import annotations
import json
import struct
from pathlib import Path
from typing import Iterable, List
from .geometry import Object3D
_IDENTITY_AXES = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_STRUCTURAL_LABELS = {'wall', 'floor', 'ceiling', 'doorframe', 'bath walls', 'closet ceiling', 'closet floor', 'shower walls', 'shower floor', 'windowsill', 'object'}
_PLY_TYPE_STRUCT = {'char': ('b', 1), 'int8': ('b', 1), 'uchar': ('B', 1), 'uint8': ('B', 1), 'short': ('h', 2), 'int16': ('h', 2), 'ushort': ('H', 2), 'uint16': ('H', 2), 'int': ('i', 4), 'int32': ('i', 4), 'uint': ('I', 4), 'uint32': ('I', 4), 'float': ('f', 4), 'float32': ('f', 4), 'double': ('d', 8), 'float64': ('d', 8)}

def find_scenes(root: Path) -> List[Path]:
    if root.is_file():
        if root.name.endswith('.aggregation.json'):
            return [root]
        raise ValueError(f'Expected a ScanNet *.aggregation.json file, got: {root}')
    return sorted(root.rglob('*.aggregation.json'))

def scene_id_from_aggregation(agg_path: Path) -> str:
    name = agg_path.name
    if name.endswith('.aggregation.json'):
        return name[:-len('.aggregation.json')]
    return agg_path.parent.name

def _mesh_paths_for_scene(agg_path: Path, scene_id: str) -> tuple[Path, Path]:
    scene_dir = agg_path.parent
    ply_path = scene_dir / f'{scene_id}_vh_clean_2.ply'
    segs_path = scene_dir / f'{scene_id}_vh_clean_2.0.010000.segs.json'
    return (ply_path, segs_path)

def load_ply_vertex_positions(ply_path: Path) -> list[tuple[float, float, float]]:
    """Read x/y/z vertex positions from a binary_little_endian ScanNet PLY mesh.

    Only the vertex element is decoded (faces and any trailing elements are ignored),
    since object bounding boxes only need vertex positions.
    """
    with ply_path.open('rb') as f:
        raw = f.read()
    header_end = raw.find(b'end_header\n')
    if header_end == -1:
        raise ValueError(f'Malformed PLY (no end_header): {ply_path}')
    header_text = raw[:header_end].decode('ascii', errors='replace')
    body = raw[header_end + len(b'end_header\n'):]
    lines = header_text.splitlines()
    if not lines or lines[0].strip() != 'ply':
        raise ValueError(f'Not a PLY file: {ply_path}')
    fmt_line = next((l for l in lines if l.startswith('format')), '')
    if 'binary_little_endian' not in fmt_line:
        raise ValueError(f'Unsupported PLY format (expected binary_little_endian): {fmt_line!r} in {ply_path}')
    vertex_count = 0
    vertex_props: list[tuple[str, str]] = []
    in_vertex_element = False
    for line in lines[1:]:
        line = line.strip()
        if line.startswith('comment') or not line:
            continue
        if line.startswith('element'):
            _, elem_name, elem_count = line.split()
            in_vertex_element = elem_name == 'vertex'
            if in_vertex_element:
                vertex_count = int(elem_count)
            continue
        if line.startswith('property') and in_vertex_element:
            parts = line.split()
            if parts[1] == 'list':
                raise ValueError(f'Unexpected list property on vertex element in {ply_path}')
            prop_type, prop_name = (parts[1], parts[2])
            vertex_props.append((prop_name, prop_type))
    fmt_chars = []
    prop_names = []
    for prop_name, prop_type in vertex_props:
        if prop_type not in _PLY_TYPE_STRUCT:
            raise ValueError(f'Unsupported PLY property type {prop_type!r} in {ply_path}')
        code, _size = _PLY_TYPE_STRUCT[prop_type]
        fmt_chars.append(code)
        prop_names.append(prop_name)
    struct_fmt = '<' + ''.join(fmt_chars)
    stride = struct.calcsize(struct_fmt)
    x_idx, y_idx, z_idx = (prop_names.index('x'), prop_names.index('y'), prop_names.index('z'))
    positions: list[tuple[float, float, float]] = []
    offset = 0
    for _ in range(vertex_count):
        values = struct.unpack_from(struct_fmt, body, offset)
        positions.append((float(values[x_idx]), float(values[y_idx]), float(values[z_idx])))
        offset += stride
    return positions

def load_scannet_objects(agg_path: Path) -> List[Object3D]:
    scene_id = scene_id_from_aggregation(agg_path)
    ply_path, segs_path = _mesh_paths_for_scene(agg_path, scene_id)
    if not ply_path.is_file() or not segs_path.is_file():
        return []
    aggregation = json.loads(agg_path.read_text())
    seg_groups = aggregation.get('segGroups', [])
    if not seg_groups:
        return []
    segs_data = json.loads(segs_path.read_text())
    seg_indices: list[int] = segs_data.get('segIndices', [])
    if not seg_indices:
        return []
    positions = load_ply_vertex_positions(ply_path)
    if len(positions) != len(seg_indices):
        return []
    segment_to_vertices: dict[int, list[int]] = {}
    for vertex_idx, seg_id in enumerate(seg_indices):
        segment_to_vertices.setdefault(seg_id, []).append(vertex_idx)
    objects: list[Object3D] = []
    for idx, group in enumerate(seg_groups):
        label = str(group.get('label', 'unknown')).strip()
        if label.lower() in _STRUCTURAL_LABELS:
            continue
        object_id = group.get('objectId', group.get('id', idx))
        segment_ids = group.get('segments', [])
        vertex_indices: list[int] = []
        for seg_id in segment_ids:
            vertex_indices.extend(segment_to_vertices.get(seg_id, []))
        if len(vertex_indices) < 4:
            continue
        xs = [positions[v][0] for v in vertex_indices]
        ys = [positions[v][1] for v in vertex_indices]
        zs = [positions[v][2] for v in vertex_indices]
        min_x, max_x = (min(xs), max(xs))
        min_y, max_y = (min(ys), max(ys))
        min_z, max_z = (min(zs), max(zs))
        centroid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0)
        axes_lengths = (max_x - min_x, max_y - min_y, max_z - min_z)
        if min(axes_lengths) < 0:
            continue
        object_key = f'obj_{idx:03d}_{label}'
        objects.append(Object3D(object_key=object_key, uid=str(object_id), label=label, centroid=centroid, axes_lengths=axes_lengths, normalized_axes=_IDENTITY_AXES))
    return objects

def load_scenes(data_root: Path) -> Iterable[tuple[str, Path, list[Object3D]]]:
    for agg_path in find_scenes(data_root):
        scene_id = scene_id_from_aggregation(agg_path)
        objects = load_scannet_objects(agg_path)
        if objects:
            yield (scene_id, agg_path, objects)
