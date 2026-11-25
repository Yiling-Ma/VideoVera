from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
Vec3 = Tuple[float, float, float]

@dataclass(frozen=True)
class Object3D:
    object_key: str
    uid: str
    label: str
    centroid: Vec3
    axes_lengths: Vec3
    normalized_axes: Tuple[float, ...]

    @property
    def longest_dimension_m(self) -> float:
        return max(self.axes_lengths)

    @property
    def longest_dimension_cm(self) -> float:
        return self.longest_dimension_m * 100.0

def vec_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2]))

def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(vec_dot(a, a))

def vec_unit(a: Sequence[float]) -> Vec3:
    n = vec_norm(a)
    if n < 1e-12:
        raise ValueError('Cannot normalize near-zero vector.')
    return (float(a[0] / n), float(a[1] / n), float(a[2] / n))

def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return vec_norm(vec_sub(a, b))

def horizontal_vec(a: Sequence[float]) -> Vec3:
    """Use the x-y plane for ARKitScenes object-relative direction.

    ARKitScenes OBB coordinates use z as vertical for aligned boxes in the local
    sample. Direction questions should ignore vertical height differences.
    """
    return (float(a[0]), float(a[1]), 0.0)

def right_vector_from_forward_xy(forward: Sequence[float]) -> Vec3:
    """Return a right vector in the horizontal x-y plane.

    For forward=(0, 1, 0), right=(1, 0, 0).
    """
    fx, fy, _ = horizontal_vec(forward)
    return vec_unit((fy, -fx, 0.0))

def classify_ego_quadrant(observer: Object3D, facing: Object3D, target: Object3D, min_component_margin: float=0.05) -> dict:
    forward_raw = horizontal_vec(vec_sub(facing.centroid, observer.centroid))
    target_raw = horizontal_vec(vec_sub(target.centroid, observer.centroid))
    forward_unit = vec_unit(forward_raw)
    right_unit = right_vector_from_forward_xy(forward_unit)
    forward_component = vec_dot(target_raw, forward_unit)
    right_component = vec_dot(target_raw, right_unit)
    if abs(forward_component) < min_component_margin:
        front_back = 'boundary'
    elif forward_component >= 0:
        front_back = 'front'
    else:
        front_back = 'back'
    if abs(right_component) < min_component_margin:
        left_right = 'boundary'
    elif right_component >= 0:
        left_right = 'right'
    else:
        left_right = 'left'
    if 'boundary' in (front_back, left_right):
        quadrant = 'ambiguous'
    else:
        quadrant = f'{front_back}-{left_right}'
    bearing_rad = math.atan2(right_component, forward_component)
    bearing_deg = math.degrees(bearing_rad)
    return {'observer_position': list(observer.centroid), 'facing_position': list(facing.centroid), 'target_position': list(target.centroid), 'forward_vector': list(forward_raw), 'target_vector': list(target_raw), 'forward_component': forward_component, 'right_component': right_component, 'bearing_deg': bearing_deg, 'quadrant': quadrant}

def centroid_distance(a: Object3D, b: Object3D) -> float:
    return euclidean_distance(a.centroid, b.centroid)

def category_counts(objects: Iterable[Object3D]) -> dict:
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.label] = counts.get(obj.label, 0) + 1
    return counts

def round_float(value: float, ndigits: int=4) -> float:
    return round(float(value), ndigits)

def rounded_list(values: Sequence[float], ndigits: int=4) -> list:
    return [round_float(v, ndigits) for v in values]
