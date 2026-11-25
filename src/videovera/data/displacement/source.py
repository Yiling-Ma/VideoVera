"""Final data-construction component: data/displacement/source.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import sys
from pathlib import Path
import numpy as np
from videovera.data.navigation.actions import validate_trace
WORLD_HABITAT_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CAMERA_OPENCV_TO_HABITAT = np.diag([1.0, -1.0, -1.0])

def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError('invalid_habitat_quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def normalized_frames(payload, video_path=None):
    source, fps = validate_trace(payload, video_path)
    frames = []
    for frame in source:
        pose = np.eye(4)
        pose[:3, :3] = WORLD_HABITAT_TO_Z_UP @ quaternion_matrix(frame['rotation_xyzw']) @ CAMERA_OPENCV_TO_HABITAT
        pose[:3, 3] = WORLD_HABITAT_TO_Z_UP @ np.asarray(frame['position_xyz_m'], dtype=float)
        frames.append({'frame_id': frame['frame_index'], 'timestamp': frame['timestamp_sec'], 'camera_to_world': pose.ravel().tolist()})
    return (frames, fps)

def direct_endpoint_components(payload, reference):
    """Independent check in Habitat's native Y-up / -Z-forward coordinates."""
    frames = payload['camera_trajectory']['frames']
    ref, target = (frames[0], frames[-1]) if reference == 'start' else (frames[-1], frames[0])
    forward = quaternion_matrix(ref['rotation_xyzw']) @ np.array([0.0, 0.0, -1.0])
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    delta = np.asarray(target['position_xyz_m']) - np.asarray(ref['position_xyz_m'])
    return (float(delta @ forward), float(delta @ right))
