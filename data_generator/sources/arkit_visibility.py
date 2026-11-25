from __future__ import annotations
import bisect
import json
import math
import struct
import zlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image
from data_generator.sources.spatial_geometry.arkitscenes import load_arkitscenes_objects
from data_generator.sources.spatial_geometry.geometry import Object3D
METHOD = 'arkitscenes_obb_projection_with_metric_depth_support_v2'
OUTPUT_SUFFIX = '_w2c_row.json'
BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))

class SceneDataError(RuntimeError):
    pass

@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    world_to_camera: np.ndarray

@dataclass(frozen=True)
class CameraIntrinsics:
    timestamp: float
    path: Path
    width: int
    height: int
    matrix: np.ndarray

@dataclass(frozen=True)
class TimestampedPath:
    timestamp: float
    path: Path

@dataclass(frozen=True)
class VisibilityConfig:
    pose_tolerance_sec: float = 0.035
    depth_tolerance_m: float = 0.1
    min_projected_area_ratio: float = 0.001
    min_depth_support_ratio: float = 0.05
    interval_gap_sec: float = 0.35
    coverage_threshold: float = 0.9
    max_frames: int | None = None
    sample_frames: int | None = None

    def validate(self) -> None:
        if self.pose_tolerance_sec < 0:
            raise ValueError('pose_tolerance_sec must be non-negative')
        if self.depth_tolerance_m < 0:
            raise ValueError('depth_tolerance_m must be non-negative')
        for name, value in (('min_projected_area_ratio', self.min_projected_area_ratio), ('min_depth_support_ratio', self.min_depth_support_ratio), ('coverage_threshold', self.coverage_threshold)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.interval_gap_sec < 0:
            raise ValueError('interval_gap_sec must be non-negative')
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError('max_frames must be positive')
        if self.sample_frames is not None and self.sample_frames <= 1:
            raise ValueError('sample_frames must be greater than one')
        if self.max_frames is not None and self.sample_frames is not None:
            raise ValueError('max_frames and sample_frames are mutually exclusive')

def axis_angle_to_matrix(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = value / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)

def parse_trajectory(path: Path) -> list[PoseSample]:
    samples: list[PoseSample] = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) != 7 or not all((math.isfinite(value) for value in values)):
            continue
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = axis_angle_to_matrix(values[1:4])
        matrix[:3, 3] = values[4:7]
        samples.append(PoseSample(values[0], matrix))
    if not samples:
        raise SceneDataError(f'No valid poses in {path}')
    return sorted(samples, key=lambda sample: sample.timestamp)

def parse_timestamp(path: Path) -> float:
    try:
        return float(path.stem.rsplit('_', 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f'Filename does not end in a numeric timestamp: {path.name}') from exc

def list_timestamped_files(directory: Path, suffix: str) -> list[TimestampedPath]:
    rows: list[TimestampedPath] = []
    for path in directory.glob(f'*{suffix}'):
        try:
            rows.append(TimestampedPath(parse_timestamp(path), path))
        except ValueError:
            continue
    return sorted(rows, key=lambda row: (row.timestamp, row.path.name))

def parse_pincam(path: Path, timestamp: float | None=None) -> CameraIntrinsics:
    values = [float(value) for value in path.read_text(encoding='utf-8').split()]
    if len(values) != 6:
        raise ValueError(f'Expected six values in {path}')
    width, height = (int(values[0]), int(values[1]))
    if width <= 0 or height <= 0 or values[2] <= 0 or (values[3] <= 0):
        raise ValueError(f'Invalid camera intrinsics in {path}')
    matrix = np.array([[values[2], 0.0, values[4]], [0.0, values[3], values[5]], [0.0, 0.0, 1.0]], dtype=np.float64)
    return CameraIntrinsics(timestamp=parse_timestamp(path) if timestamp is None else timestamp, path=path, width=width, height=height, matrix=matrix)

def nearest_timestamped(timestamp: float, rows: Sequence[TimestampedPath], tolerance_sec: float) -> TimestampedPath | None:
    if not rows:
        return None
    timestamps = [row.timestamp for row in rows]
    index = bisect.bisect_left(timestamps, timestamp)
    candidates = rows[max(0, index - 1):min(len(rows), index + 1)]
    best = min(candidates, key=lambda row: (abs(row.timestamp - timestamp), row.timestamp, row.path.name))
    return best if abs(best.timestamp - timestamp) <= tolerance_sec else None

def obb_corners(obj: Object3D) -> np.ndarray:
    axes = np.asarray(obj.normalized_axes, dtype=np.float64).reshape(3, 3)
    half_lengths = np.asarray(obj.axes_lengths, dtype=np.float64) / 2.0
    local = np.asarray(list(product((-1.0, 1.0), repeat=3)), dtype=np.float64) * half_lengths
    return local @ axes + np.asarray(obj.centroid, dtype=np.float64)

def transform_world_to_camera(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]

def _near_clipped_points(camera_corners: np.ndarray, near: float=1e-06) -> np.ndarray:
    points = [point for point in camera_corners if point[2] >= near]
    for start_index, end_index in BOX_EDGES:
        start = camera_corners[start_index]
        end = camera_corners[end_index]
        if (start[2] >= near) == (end[2] >= near):
            continue
        alpha = (near - start[2]) / (end[2] - start[2])
        points.append(start + alpha * (end - start))
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)

def project_obb(world_corners: np.ndarray, world_to_camera: np.ndarray, intrinsic: np.ndarray, image_width: int, image_height: int) -> dict[str, Any] | None:
    camera_corners = transform_world_to_camera(world_corners, world_to_camera)
    clipped_points = _near_clipped_points(camera_corners)
    if clipped_points.size == 0:
        return None
    pixels_h = clipped_points @ intrinsic.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    raw_min = pixels.min(axis=0)
    raw_max = pixels.max(axis=0)
    x0 = float(np.clip(raw_min[0], 0.0, float(image_width)))
    y0 = float(np.clip(raw_min[1], 0.0, float(image_height)))
    x1 = float(np.clip(raw_max[0], 0.0, float(image_width)))
    y1 = float(np.clip(raw_max[1], 0.0, float(image_height)))
    if x1 <= x0 or y1 <= y0:
        return None
    positive_depths = clipped_points[:, 2]
    area_ratio = (x1 - x0) * (y1 - y0) / float(image_width * image_height)
    raw_area = max(0.0, float(raw_max[0] - raw_min[0])) * max(0.0, float(raw_max[1] - raw_min[1]))
    clipped_area = (x1 - x0) * (y1 - y0)
    return {'bbox_2d': [x0, y0, x1, y1], 'raw_bbox_2d': [float(raw_min[0]), float(raw_min[1]), float(raw_max[0]), float(raw_max[1])], 'center_2d': [(x0 + x1) / 2.0, (y0 + y1) / 2.0], 'bbox_area_ratio': float(area_ratio), 'in_frame_fraction': float(clipped_area / raw_area) if raw_area > 0 else 0.0, 'min_corner_depth_m': float(camera_corners[:, 2].min()), 'all_corners_in_front': bool(np.all(camera_corners[:, 2] > 0.0)), 'expected_depth_range_m': [float(positive_depths.min()), float(positive_depths.max())]}

def evaluate_obb_depth_support(depth_m: np.ndarray, obj: Object3D, world_to_camera: np.ndarray, intrinsic: np.ndarray, projection: dict[str, Any], depth_tolerance_m: float) -> dict[str, Any]:
    """Measure metric-depth support using per-pixel ray/OBB intersections.

    The older visibility test compared every depth pixel in the projected 2D
    rectangle with one global min/max depth range.  For close or partly
    off-screen objects, unrelated walls and furniture inside that rectangle
    could therefore be counted as support.  Here each pixel ray is intersected
    with the oriented 3D box and is supported only when the measured depth lies
    on that ray's own box interval.
    """
    height, width = depth_m.shape
    x0, y0, x1, y1 = projection['bbox_2d']
    ix0 = max(0, min(width, int(math.floor(x0))))
    iy0 = max(0, min(height, int(math.floor(y0))))
    ix1 = max(0, min(width, int(math.ceil(x1))))
    iy1 = max(0, min(height, int(math.ceil(y1))))
    if ix1 <= ix0 or iy1 <= iy0:
        return {'visible': False, 'ray_box_pixels': 0, 'valid_depth_pixels': 0, 'support_depth_pixels': 0, 'valid_depth_coverage': 0.0, 'depth_support_ratio': 0.0}
    ys, xs = np.mgrid[iy0:iy1, ix0:ix1]
    fx, fy = (float(intrinsic[0, 0]), float(intrinsic[1, 1]))
    cx, cy = (float(intrinsic[0, 2]), float(intrinsic[1, 2]))
    camera_dirs = np.stack(((xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, np.ones_like(xs)), axis=-1).reshape(-1, 3)
    rotation = np.asarray(world_to_camera[:3, :3], dtype=np.float64)
    translation = np.asarray(world_to_camera[:3, 3], dtype=np.float64)
    camera_center_world = -(translation @ rotation)
    axes = np.asarray(obj.normalized_axes, dtype=np.float64).reshape(3, 3)
    centroid = np.asarray(obj.centroid, dtype=np.float64)
    half_lengths = np.asarray(obj.axes_lengths, dtype=np.float64) / 2.0
    world_dirs = camera_dirs @ rotation
    local_dirs = world_dirs @ axes.T
    local_origin = (camera_center_world - centroid) @ axes.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (-half_lengths - local_origin) / local_dirs
        t1 = (half_lengths - local_origin) / local_dirs
    parallel = np.abs(local_dirs) < 1e-12
    outside_parallel = parallel & ((local_origin < -half_lengths) | (local_origin > half_lengths))
    t_low = np.where(parallel, -np.inf, np.minimum(t0, t1))
    t_high = np.where(parallel, np.inf, np.maximum(t0, t1))
    near = np.max(t_low, axis=1)
    far = np.min(t_high, axis=1)
    intersects = ~np.any(outside_parallel, axis=1) & np.isfinite(near) & np.isfinite(far) & (far >= np.maximum(near, 1e-06))
    crop_depth = depth_m[iy0:iy1, ix0:ix1].reshape(-1)
    valid = intersects & np.isfinite(crop_depth) & (crop_depth > 0.0)
    supported = valid & (np.abs(crop_depth - np.maximum(near, 0.0)) <= depth_tolerance_m)
    ray_count = int(np.count_nonzero(intersects))
    valid_count = int(np.count_nonzero(valid))
    support_count = int(np.count_nonzero(supported))
    return {'visible': bool(ray_count > 0 and support_count > 0), 'ray_box_pixels': ray_count, 'valid_depth_pixels': valid_count, 'support_depth_pixels': support_count, 'valid_depth_coverage': valid_count / ray_count if ray_count else 0.0, 'depth_support_ratio': support_count / ray_count if ray_count else 0.0}

def evaluate_depth_support(depth_m: np.ndarray, projection: dict[str, Any], depth_tolerance_m: float, min_depth_support_ratio: float) -> dict[str, Any]:
    height, width = depth_m.shape
    x0, y0, x1, y1 = projection['bbox_2d']
    ix0 = max(0, min(width, int(math.floor(x0))))
    iy0 = max(0, min(height, int(math.floor(y0))))
    ix1 = max(0, min(width, int(math.ceil(x1))))
    iy1 = max(0, min(height, int(math.ceil(y1))))
    crop = depth_m[iy0:iy1, ix0:ix1]
    valid = np.isfinite(crop) & (crop > 0.0)
    valid_count = int(np.count_nonzero(valid))
    near, far = projection['expected_depth_range_m']
    consistent = valid & (crop >= near - depth_tolerance_m) & (crop <= far + depth_tolerance_m)
    support_count = int(np.count_nonzero(consistent))
    support_ratio = support_count / valid_count if valid_count else 0.0
    return {'visible': bool(valid_count > 0 and support_ratio >= min_depth_support_ratio), 'valid_depth_pixels': valid_count, 'support_depth_pixels': support_count, 'depth_support_ratio': float(support_ratio)}

def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
    return (left, above, upper_left)[distances.index(min(distances))]

def load_depth_png_meters(path: Path) -> np.ndarray:
    """Read an ARKitScenes 16-bit grayscale depth PNG in meters."""
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise ValueError(f'Expected a 16-bit grayscale depth PNG: {path}')
    return depth.astype(np.float32) / 1000.0

def load_depth_png_meters_reference(path: Path) -> np.ndarray:
    """Pure-Python reference decoder retained for format validation tests."""
    data = path.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Not a PNG file: {path}')
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b'IHDR':
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack('>IIBBBBB', payload)
        elif chunk_type == b'IDAT':
            compressed.extend(payload)
        elif chunk_type == b'IEND':
            break
    if (bit_depth, color_type, interlace) != (16, 0, 0) or width is None or height is None:
        raise ValueError(f'Expected non-interlaced 16-bit grayscale depth PNG: {path}')
    raw = zlib.decompress(bytes(compressed))
    stride = width * 2
    expected_size = height * (stride + 1)
    if len(raw) != expected_size:
        raise ValueError(f'Unexpected decompressed PNG size in {path}')
    rows = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row_index in range(height):
        filter_type = raw[cursor]
        source = raw[cursor + 1:cursor + 1 + stride]
        cursor += stride + 1
        decoded = np.empty(stride, dtype=np.uint8)
        for index, value in enumerate(source):
            left = int(decoded[index - 2]) if index >= 2 else 0
            above = int(previous[index])
            upper_left = int(previous[index - 2]) if index >= 2 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise ValueError(f'Unsupported PNG filter {filter_type} in {path}')
            decoded[index] = int(value) + predictor & 255
        rows[row_index] = decoded
        previous = decoded
    millimeters = np.frombuffer(rows.tobytes(), dtype='>u2').reshape(height, width)
    return millimeters.astype(np.float32) / 1000.0

def scale_intrinsics(intrinsics: CameraIntrinsics, width: int, height: int) -> np.ndarray:
    matrix = intrinsics.matrix.copy()
    matrix[0, :] *= width / intrinsics.width
    matrix[1, :] *= height / intrinsics.height
    return matrix

def map_arkit_timestamp(timestamp: float, first_timestamp: float) -> float:
    return max(0.0, float(timestamp - first_timestamp))

def merge_visible_timestamps(timestamps: Sequence[float], max_gap_sec: float) -> list[list[float]]:
    ordered = sorted(set((float(value) for value in timestamps)))
    if not ordered:
        return []
    intervals: list[list[float]] = [[ordered[0], ordered[0]]]
    for timestamp in ordered[1:]:
        if timestamp - intervals[-1][1] <= max_gap_sec + 1e-12:
            intervals[-1][1] = timestamp
        else:
            intervals.append([timestamp, timestamp])
    return intervals

def assess_coverage(total_frames: int, processed_frames: int, first_timestamp: float, last_timestamp: float, processed_timestamps: Sequence[float], threshold: float, truncated: bool=False, object_errors: int=0) -> dict[str, Any]:
    frame_ratio = processed_frames / total_frames if total_frames else 0.0
    duration = last_timestamp - first_timestamp
    if duration <= 1e-12:
        span_ratio = 1.0 if processed_timestamps else 0.0
    elif processed_timestamps:
        span = max(processed_timestamps) - min(processed_timestamps)
        span_ratio = max(0.0, min(1.0, span / duration))
    else:
        span_ratio = 0.0
    complete = bool(not truncated and object_errors == 0 and (total_frames > 0) and (frame_ratio >= threshold) and (span_ratio >= threshold))
    return {'frame_coverage_ratio': float(frame_ratio), 'timeline_span_coverage_ratio': float(span_ratio), 'coverage_complete': complete, 'visibility_reliable': complete}

def _frame_record(pose: PoseSample, video_timestamp: float, frame_row: TimestampedPath | None, depth_row: TimestampedPath, intrinsics: CameraIntrinsics, projection: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    return {'timestamp': video_timestamp, 'video_timestamp_sec': video_timestamp, 'arkit_timestamp': pose.timestamp, 'frame_name': frame_row.path.name if frame_row else depth_row.path.name, 'frame_arkit_timestamp': frame_row.timestamp if frame_row else depth_row.timestamp, 'depth_frame_name': depth_row.path.name, 'depth_arkit_timestamp': depth_row.timestamp, 'intrinsics_frame_name': intrinsics.path.name, 'intrinsics_arkit_timestamp': intrinsics.timestamp, **projection, **{key: value for key, value in support.items() if key != 'visible'}}

def build_scene_visibility(scene_dir: Path, config: VisibilityConfig) -> dict[str, Any]:
    config.validate()
    scene_id = scene_dir.name
    annotation_path = scene_dir / f'{scene_id}_3dod_annotation.json'
    trajectory_path = scene_dir / 'lowres_wide.traj'
    if not annotation_path.is_file():
        raise SceneDataError(f'Missing annotation: {annotation_path}')
    if not trajectory_path.is_file():
        raise SceneDataError(f'Missing trajectory: {trajectory_path}')
    objects = sorted(load_arkitscenes_objects(annotation_path), key=lambda obj: obj.object_key)
    poses = parse_trajectory(trajectory_path)
    depth_rows = list_timestamped_files(scene_dir / 'lowres_depth', '.png')
    intrinsic_rows = list_timestamped_files(scene_dir / 'lowres_wide_intrinsics', '.pincam')
    rgb_timeline_rows = list_timestamped_files(scene_dir / 'lowres_wide', '.png')
    timeline_rows = rgb_timeline_rows or list(depth_rows)
    if not depth_rows:
        raise SceneDataError(f"No depth PNG files in {scene_dir / 'lowres_depth'}")
    if not intrinsic_rows:
        raise SceneDataError(f"No .pincam files in {scene_dir / 'lowres_wide_intrinsics'}")
    if not timeline_rows:
        raise SceneDataError(f'No frame timeline metadata for scene {scene_id}')
    timeline_first = timeline_rows[0].timestamp
    timeline_last = timeline_rows[-1].timestamp
    eligible_poses = [pose for pose in poses if timeline_first - config.pose_tolerance_sec <= pose.timestamp <= timeline_last + config.pose_tolerance_sec]
    truncated = config.max_frames is not None and len(eligible_poses) > config.max_frames
    if config.max_frames is not None:
        selected_poses = eligible_poses[:config.max_frames]
        sampling_strategy = 'prefix_smoke_limit'
    elif config.sample_frames is not None and len(eligible_poses) > config.sample_frames:
        selected_poses = _uniform_sample(eligible_poses, config.sample_frames)
        sampling_strategy = 'uniform_full_timeline'
    else:
        selected_poses = eligible_poses
        sampling_strategy = 'all_eligible_poses'
    visible_by_object: dict[str, list[dict[str, Any]]] = {obj.object_key: [] for obj in objects}
    corners_by_object = {obj.object_key: obb_corners(obj) for obj in objects}
    stats = {'trajectory_samples': len(poses), 'timeline_eligible_samples': len(eligible_poses), 'selected_samples': len(selected_poses), 'processed_frames': 0, 'missing_depth_match': 0, 'missing_intrinsics_match': 0, 'missing_frame_timeline_match': 0, 'depth_read_errors': 0, 'intrinsics_read_errors': 0, 'object_projection_errors': 0, 'smoke_limited': truncated, 'sampling_strategy': sampling_strategy, 'full_timeline_sampling_ratio': len(selected_poses) / len(eligible_poses) if eligible_poses else 0.0}
    processed_timestamps: list[float] = []
    for pose in selected_poses:
        depth_row = nearest_timestamped(pose.timestamp, depth_rows, config.pose_tolerance_sec)
        if depth_row is None:
            stats['missing_depth_match'] += 1
            continue
        intrinsic_row = nearest_timestamped(pose.timestamp, intrinsic_rows, config.pose_tolerance_sec)
        if intrinsic_row is None:
            stats['missing_intrinsics_match'] += 1
            continue
        try:
            depth_m = load_depth_png_meters(depth_row.path)
        except Exception:
            stats['depth_read_errors'] += 1
            continue
        try:
            intrinsics = parse_pincam(intrinsic_row.path, intrinsic_row.timestamp)
            matrix = scale_intrinsics(intrinsics, depth_m.shape[1], depth_m.shape[0])
        except Exception:
            stats['intrinsics_read_errors'] += 1
            continue
        frame_row = nearest_timestamped(pose.timestamp, timeline_rows, config.pose_tolerance_sec)
        if frame_row is None:
            stats['missing_frame_timeline_match'] += 1
            continue
        video_timestamp = map_arkit_timestamp(pose.timestamp, timeline_first)
        for obj in objects:
            try:
                projection = project_obb(corners_by_object[obj.object_key], pose.world_to_camera, matrix, depth_m.shape[1], depth_m.shape[0])
                if projection is None or projection['bbox_area_ratio'] < config.min_projected_area_ratio:
                    continue
                support = evaluate_depth_support(depth_m, projection, config.depth_tolerance_m, config.min_depth_support_ratio)
                if support['visible']:
                    visible_by_object[obj.object_key].append(_frame_record(pose, video_timestamp, frame_row, depth_row, intrinsics, projection, support))
            except Exception:
                stats['object_projection_errors'] += 1
        stats['processed_frames'] += 1
        processed_timestamps.append(pose.timestamp)
    coverage = assess_coverage(len(selected_poses) if sampling_strategy == 'uniform_full_timeline' else len(eligible_poses), stats['processed_frames'], timeline_first, timeline_last, processed_timestamps, config.coverage_threshold, truncated=truncated, object_errors=stats['object_projection_errors'])
    object_rows = []
    for obj in objects:
        frames = sorted(visible_by_object[obj.object_key], key=lambda row: (row['timestamp'], row['depth_frame_name']))
        timestamps = sorted(set((float(frame['timestamp']) for frame in frames)))
        object_rows.append({'object_key': obj.object_key, 'uid': obj.uid, 'label': obj.label, 'num_visible_frames': len(frames), 'visible_timestamps': timestamps, 'visible_intervals': merge_visible_timestamps(timestamps, config.interval_gap_sec), 'visible_frames': frames, 'first_visible_frame': frames[0] if frames else None, 'last_visible_frame': frames[-1] if frames else None})
    thresholds = {'pose_tolerance_sec': config.pose_tolerance_sec, 'depth_tolerance_m': config.depth_tolerance_m, 'min_projected_area_ratio': config.min_projected_area_ratio, 'min_depth_support_ratio': config.min_depth_support_ratio, 'interval_gap_sec': config.interval_gap_sec, 'coverage_threshold': config.coverage_threshold, 'sample_frames': config.sample_frames}
    return {'scene_id': scene_id, 'visibility_reliable': coverage['visibility_reliable'], 'coverage_complete': coverage['coverage_complete'], 'video_duration_sec': map_arkit_timestamp(timeline_last, timeline_first), 'uses_rgb': False, 'uses_llm': False, 'method': METHOD, 'thresholds': thresholds, 'timestamp_mapping': {'method': 'subtract_first_frame_arkit_timestamp', 'source': 'lowres_wide_filenames' if rgb_timeline_rows else 'depth_filenames', 'first_frame_arkit_timestamp': timeline_first, 'last_frame_arkit_timestamp': timeline_last, 'video_start_sec': 0.0, 'video_end_sec': map_arkit_timestamp(timeline_last, timeline_first), 'formula': 'video_timestamp_sec = arkit_timestamp - first_frame_arkit_timestamp'}, 'frame_processing_stats': {**stats, **coverage}, 'objects': object_rows}

def _uniform_sample(rows: Sequence[Any], sample_count: int) -> list[Any]:
    if len(rows) <= sample_count:
        return list(rows)
    indices = [round(index * (len(rows) - 1) / (sample_count - 1)) for index in range(sample_count)]
    return [rows[index] for index in indices]

def output_path(output_dir: Path, scene_id: str) -> Path:
    return output_dir / f'visibility_{scene_id}{OUTPUT_SUFFIX}'

def write_scene_visibility(scene_dir: Path, output_dir: Path, config: VisibilityConfig, overwrite: bool=False) -> Path:
    destination = output_path(output_dir, scene_dir.name)
    if destination.exists() and (not overwrite):
        raise FileExistsError(f'Output already exists: {destination}')
    result = build_scene_visibility(scene_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(destination)
    return destination
