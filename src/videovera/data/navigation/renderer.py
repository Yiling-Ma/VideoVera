"""Final data-construction component: data/navigation/renderer.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
from videovera.io import preserve
import argparse
import gzip
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
FPS = 8
WIDTH = 640
HEIGHT = 480
HFOV_DEG = 90.0
SENSOR_HEIGHT_M = 1.5
FORWARD_STEP_M = 0.25
TURN_DEG = 15.0
WAYPOINT_GOAL_RADIUS_M = 0.35
INITIAL_HOLD_FRAMES = 4
FINAL_HOLD_EXTRA_FRAMES = 8

@dataclass(frozen=True)
class RenderConfig:
    episodes_path: Path
    scenes_dir: Path
    out_root: Path
    max_episodes: int
    output_index_offset: int
    max_per_scene: int
    skip_episodes: int
    max_failures: int
    max_near_black_ratio: float
    min_distance_m: float
    max_distance_m: float
    gpu_device_id: int
    overwrite: bool
    resume: bool
    selection_manifest: Path | None

def parse_args() -> RenderConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True, help='R2R-VLNCE split JSON or JSON.GZ, e.g. train.json.gz.')
    parser.add_argument('--scenes-dir', type=Path, required=True, help='Directory containing mp3d/<scan>/<scan>.glb.')
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path, help='Render the exact scene/episode/trajectory entries in an existing R2R manifest, preserving manifest order and output numbering.')
    parser.add_argument('--max-episodes', type=int, default=300)
    parser.add_argument('--output-index-offset', type=int, default=0, help='Add this offset to output file numbering for collision-free parallel shards.')
    parser.add_argument('--max-per-scene', type=int, default=1, help='Maximum eligible episodes selected from one Matterport scene.')
    parser.add_argument('--skip-episodes', type=int, default=0, help='Skip this many eligible per-scene episodes before rendering.')
    parser.add_argument('--max-failures', type=int, default=30, help='Maximum failed controller candidates to inspect while filling the requested output count.')
    parser.add_argument('--max-near-black-ratio', type=float, default=0.1, help='Reject trajectories when any rendered frame exceeds this near-black pixel ratio.')
    parser.add_argument('--min-distance-m', type=float, default=5.0)
    parser.add_argument('--max-distance-m', type=float, default=12.0)
    parser.add_argument('--gpu-device-id', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Keep complete existing video/sidecar pairs and render only the remaining episodes.')
    args = parser.parse_args()
    if args.max_episodes <= 0:
        raise ValueError('--max-episodes must be positive.')
    if args.output_index_offset < 0:
        raise ValueError('--output-index-offset must be non-negative.')
    if args.max_per_scene <= 0:
        raise ValueError('--max-per-scene must be positive.')
    if args.skip_episodes < 0:
        raise ValueError('--skip-episodes must be non-negative.')
    if args.max_failures < 0:
        raise ValueError('--max-failures must be non-negative.')
    if not 0.0 <= args.max_near_black_ratio <= 1.0:
        raise ValueError('--max-near-black-ratio must be in [0, 1].')
    if args.min_distance_m < 0 or args.max_distance_m < args.min_distance_m:
        raise ValueError('Distance bounds must satisfy 0 <= min <= max.')
    return RenderConfig(episodes_path=args.episodes.expanduser().resolve(), scenes_dir=args.scenes_dir.expanduser().resolve(), out_root=args.out_root.expanduser().resolve(), max_episodes=args.max_episodes, output_index_offset=args.output_index_offset, max_per_scene=args.max_per_scene, skip_episodes=args.skip_episodes, max_failures=args.max_failures, max_near_black_ratio=args.max_near_black_ratio, min_distance_m=args.min_distance_m, max_distance_m=args.max_distance_m, gpu_device_id=args.gpu_device_id, overwrite=args.overwrite, resume=args.resume, selection_manifest=args.selection_manifest.expanduser().resolve() if args.selection_manifest else None)

def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

def load_episodes(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get('episodes', payload.get('data', []))
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported episode payload in {path}; expected a list or {{'episodes': [...]}}.")
    return [item for item in payload if isinstance(item, dict)]

def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None

def instruction_text(episode: dict[str, Any]) -> str | None:
    instruction = first_present(episode, 'instruction', 'instruction_text')
    if isinstance(instruction, dict):
        instruction = first_present(instruction, 'instruction_text', 'text')
    return str(instruction).strip() if isinstance(instruction, str) else None

def xyz(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} must be an XYZ list, got {value!r}')
    return np.asarray(value, dtype=np.float64)

def extract_reference_path(episode: dict[str, Any]) -> list[np.ndarray]:
    raw_path = first_present(episode, 'reference_path', 'reference_path_points', 'path')
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise ValueError('Episode is missing a reference_path with at least two points.')
    points: list[np.ndarray] = []
    for index, point in enumerate(raw_path):
        if isinstance(point, dict):
            point = first_present(point, 'position', 'position_xyz_m', 'point')
        points.append(xyz(point, f'reference_path[{index}]'))
    return points

def episode_distance_m(path: list[np.ndarray]) -> float:
    return float(sum((np.linalg.norm(right[[0, 2]] - left[[0, 2]]) for left, right in zip(path, path[1:]))))

def trajectory_key(episode: dict[str, Any], path: list[np.ndarray] | None=None) -> tuple[str, str]:
    """Identify physical routes independently of their instruction/episode variants."""
    scene = scan_id(str(first_present(episode, 'scene_id', 'scene') or ''))
    trajectory_id = first_present(episode, 'trajectory_id')
    if trajectory_id is not None and str(trajectory_id).strip():
        return (scene, f'trajectory:{trajectory_id}')
    if path is None:
        path = extract_reference_path(episode)
    normalized_path = [[round(float(component), 4) for component in point] for point in path]
    return (scene, 'path:' + json.dumps(normalized_path, separators=(',', ':')))

def scan_id(scene_id: str) -> str:
    """Extract the Matterport scan identifier from a VLN-CE scene path."""
    normalized = scene_id.replace('\\', '/')
    parts = [part for part in normalized.split('/') if part and part != 'mp3d']
    if not parts:
        raise ValueError(f'Cannot determine a Matterport scan id from {scene_id!r}')
    return Path(parts[-1]).stem

def scene_file(scenes_dir: Path, scene_id: str) -> Path:
    scan = scan_id(scene_id)
    direct = scenes_dir / 'mp3d' / scan / f'{scan}.glb'
    if direct.is_file():
        return direct
    nested = scenes_dir / scan / f'{scan}.glb'
    if nested.is_file():
        return nested
    raise FileNotFoundError(f'Missing Habitat scene for {scene_id}: expected {direct} or {nested}')

def require_runtime(config: RenderConfig, selected: list[dict[str, Any]]) -> Any:
    if not config.episodes_path.is_file():
        raise FileNotFoundError(f'Missing R2R-VLNCE episodes: {config.episodes_path}')
    if not config.scenes_dir.is_dir():
        raise FileNotFoundError(f'Missing Matterport3D scenes directory: {config.scenes_dir}')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required to encode controller RGB videos.')
    missing = []
    for row in selected:
        scene_id = str(first_present(row, 'scene_id', 'scene') or '')
        try:
            scene_file(config.scenes_dir, scene_id)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise FileNotFoundError('Cannot render continuous Habitat RGB.\n' + '\n'.join(missing[:10]))
    try:
        import habitat_sim
        import quaternion
    except ImportError as exc:
        raise RuntimeError('Habitat-Sim is required. Use the VLN-CE-compatible Habitat-Sim environment; do not substitute MatterSim skybox rendering.') from exc
    return habitat_sim

def select_episodes(config: RenderConfig, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select unique trajectories in scene round-robin order."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    seen_trajectories: set[tuple[str, str]] = set()
    for row in episodes:
        source_scene_id = str(first_present(row, 'scene_id', 'scene') or '')
        if not source_scene_id:
            continue
        try:
            path = extract_reference_path(row)
        except ValueError:
            continue
        distance = episode_distance_m(path)
        if not config.min_distance_m <= distance <= config.max_distance_m:
            continue
        key = trajectory_key(row, path)
        if key in seen_trajectories:
            continue
        seen_trajectories.add(key)
        scene_id = key[0]
        scene_rows = by_scene.setdefault(scene_id, [])
        if len(scene_rows) >= config.max_per_scene:
            continue
        scene_rows.append(row)
    round_robin = [scene_rows[index] for index in range(config.max_per_scene) for scene_rows in by_scene.values() if index < len(scene_rows)]
    selected = round_robin[config.skip_episodes:config.skip_episodes + config.max_episodes + config.max_failures]
    if not selected:
        raise RuntimeError('No eligible R2R-VLNCE episodes matched the requested distance range.')
    return selected

def select_manifest_episodes(manifest_path: Path, episodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    requested = payload.get('videos') if isinstance(payload, dict) else None
    if not isinstance(requested, list) or not requested:
        raise ValueError(f'Selection manifest has no videos: {manifest_path}')
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for episode in episodes:
        key = (scan_id(str(first_present(episode, 'scene_id', 'scene') or '')), str(first_present(episode, 'episode_id', 'id')), str(first_present(episode, 'trajectory_id')))
        if key in by_key:
            raise ValueError(f'Duplicate episode key in annotations: {key}')
        by_key[key] = episode
    selected = []
    missing = []
    for row in requested[:limit]:
        key = (scan_id(str(row.get('scene_id') or '')), str(row.get('episode_id')), str(row.get('trajectory_id')))
        episode = by_key.get(key)
        if episode is None:
            missing.append(key)
        else:
            selected.append(episode)
    if missing:
        raise KeyError(f'Selection manifest entries are absent from the annotations; examples: {missing[:5]}')
    if len(selected) != min(limit, len(requested)):
        raise RuntimeError('Failed to preserve the requested manifest selection')
    return selected

def make_simulator(habitat_sim: Any, scene: Path, gpu_device_id: int) -> Any:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.gpu_device_id = gpu_device_id
    sim_cfg.load_semantic_mesh = False
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = 'rgb'
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [HEIGHT, WIDTH]
    sensor.position = [0.0, SENSOR_HEIGHT_M, 0.0]
    sensor.hfov = HFOV_DEG
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.action_space = {'move_forward': habitat_sim.agent.ActionSpec('move_forward', habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP_M)), 'turn_left': habitat_sim.agent.ActionSpec('turn_left', habitat_sim.agent.ActuationSpec(amount=TURN_DEG)), 'turn_right': habitat_sim.agent.ActionSpec('turn_right', habitat_sim.agent.ActuationSpec(amount=TURN_DEG))}
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

def rotation_xyzw(rotation: Any) -> list[float]:
    return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]

def heading_deg(rotation: Any) -> float:
    import quaternion
    forward = quaternion.as_rotation_matrix(rotation) @ np.asarray([0.0, 0.0, -1.0])
    return float(math.degrees(math.atan2(forward[0], -forward[2])) % 360.0)

def target_heading_deg(position: np.ndarray, target: np.ndarray) -> float:
    delta = target[[0, 2]] - position[[0, 2]]
    return float(math.degrees(math.atan2(delta[0], -delta[1])) % 360.0)

def signed_angle_deg(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0

def macro_turn_action(heading_change_deg: float) -> str | None:
    magnitude = abs(heading_change_deg)
    if magnitude < TURN_DEG:
        return None
    if magnitude >= 135.0:
        return 'TURN_BACK'
    return 'TURN_RIGHT' if heading_change_deg > 0 else 'TURN_LEFT'

def macro_event(action_type: str, action_index: int, waypoint_index: int, heading_change: float, raw_actions: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [action for action in raw_actions if int(action['target_waypoint_index']) == waypoint_index and (action['action_type'] == 'MOVE_FORWARD' if action_type == 'GO_STRAIGHT' else action['action_type'] in {'TURN_LEFT', 'TURN_RIGHT'})]
    if not relevant:
        segment_actions = [action for action in raw_actions if int(action['target_waypoint_index']) == waypoint_index]
        relevant = segment_actions[:1]
    start_idx = min((int(action['start_idx']) for action in relevant))
    end_idx = max((int(action['end_idx']) for action in relevant))
    return {'action_type': action_type, 'from_waypoint_index': waypoint_index - 1, 'to_waypoint_index': waypoint_index, 'segment_indices': [waypoint_index], 'heading_change_deg': round(heading_change if action_type != 'GO_STRAIGHT' else 0.0, 4), 'controller_step_indices': [int(action['step']) for action in relevant], 'action_index': action_index, 'video_frame_start': start_idx, 'video_frame_end': end_idx, 'video_start_sec': round(start_idx / FPS, 6), 'video_end_sec': round(end_idx / FPS, 6)}

def canonical_route_metadata(start_rotation: list[float], reference_path: list[np.ndarray], raw_actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import quaternion
    start_quaternion = quaternion.quaternion(start_rotation[3], start_rotation[0], start_rotation[1], start_rotation[2])
    previous_heading = heading_deg(start_quaternion)
    waypoints = [{'waypoint_index': 0, 'position_xyz_m': reference_path[0].round(9).tolist(), 'reference_heading_deg': round(previous_heading, 4), 'segment_distance_xz_m': 0.0}]
    events: list[dict[str, Any]] = []
    for waypoint_index, (left, right) in enumerate(zip(reference_path, reference_path[1:]), start=1):
        segment_heading = target_heading_deg(left, right)
        heading_change = signed_angle_deg(previous_heading, segment_heading)
        segment_actions = [action for action in raw_actions if int(action['target_waypoint_index']) == waypoint_index]
        movement_actions = [action for action in segment_actions if action['action_type'] == 'MOVE_FORWARD']
        turn_actions = [action for action in segment_actions if action['action_type'] in {'TURN_LEFT', 'TURN_RIGHT'}]
        turn_action = macro_turn_action(heading_change)
        if turn_action and turn_actions:
            events.append(macro_event(turn_action, len(events) + 1, waypoint_index, heading_change, raw_actions))
        if movement_actions:
            events.append(macro_event('GO_STRAIGHT', len(events) + 1, waypoint_index, 0.0, raw_actions))
        waypoints.append({'waypoint_index': waypoint_index, 'position_xyz_m': right.round(9).tolist(), 'reference_heading_deg': round(segment_heading, 4), 'segment_distance_xz_m': round(float(np.linalg.norm(right[[0, 2]] - left[[0, 2]])), 6)})
        previous_heading = segment_heading
    return (events, waypoints)

def rgb(observation: Any) -> np.ndarray:
    image = np.asarray(observation)
    if image.ndim != 3 or image.shape[:2] != (HEIGHT, WIDTH):
        raise RuntimeError(f'Unexpected Habitat RGB observation shape: {image.shape}')
    return np.ascontiguousarray(image[..., :3], dtype=np.uint8)

class VideoWriter:

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.max_near_black_ratio = 0.0
        self.camera_frames = []
        self.process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray, camera_state: Any) -> None:
        assert self.process.stdin is not None
        self.max_near_black_ratio = max(self.max_near_black_ratio, float((frame.max(axis=2) <= 12).mean()))
        self.process.stdin.write(frame.tobytes())
        index = len(self.camera_frames)
        self.camera_frames.append({'frame_index': index, 'timestamp_sec': index / FPS, 'position_xyz_m': [float(v) for v in camera_state.position], 'rotation_xyzw': rotation_xyzw(camera_state.rotation)})

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f'ffmpeg failed while writing {self.path}')

def action_record(step: int, action_type: str, before: Any, after: Any, waypoint_index: int, target: np.ndarray) -> dict[str, Any]:
    before_position = np.asarray(before.position, dtype=np.float64)
    after_position = np.asarray(after.position, dtype=np.float64)
    return {'step': step, 'action_type': action_type, 'target_waypoint_index': waypoint_index, 'target_waypoint_position_xyz_m': target.round(9).tolist(), 'position_before_xyz_m': before_position.round(9).tolist(), 'position_after_xyz_m': after_position.round(9).tolist(), 'rotation_before_xyzw': rotation_xyzw(before.rotation), 'rotation_after_xyzw': rotation_xyzw(after.rotation), 'heading_before_deg': round(heading_deg(before.rotation), 4), 'heading_after_deg': round(heading_deg(after.rotation), 4), 'turn_degree': TURN_DEG if action_type == 'TURN_RIGHT' else -TURN_DEG if action_type == 'TURN_LEFT' else 0.0, 'distance_m': round(float(np.linalg.norm(after_position[[0, 2]] - before_position[[0, 2]])), 6), 'start_idx': INITIAL_HOLD_FRAMES + step - 1, 'end_idx': INITIAL_HOLD_FRAMES + step, 'start_sec': round((INITIAL_HOLD_FRAMES + step - 1) / FPS, 6), 'end_sec': round((INITIAL_HOLD_FRAMES + step) / FPS, 6)}

def execute_reference_path(sim: Any, start_position: np.ndarray, start_rotation: list[float], reference_path: list[np.ndarray], writer: VideoWriter) -> list[dict[str, Any]]:
    import quaternion
    agent = sim.initialize_agent(0)
    state = agent.get_state()
    state.position = start_position
    state.rotation = quaternion.quaternion(start_rotation[3], start_rotation[0], start_rotation[1], start_rotation[2])
    agent.set_state(state, reset_sensors=True)
    current_frame = rgb(sim.get_sensor_observations()['rgb'])
    for _ in range(INITIAL_HOLD_FRAMES):
        writer.write(current_frame, agent.get_state().sensor_states['rgb'])
    actions: list[dict[str, Any]] = []
    step = 0
    for waypoint_index, target in enumerate(reference_path[1:], start=1):
        stagnant_moves = 0
        for _ in range(1000):
            before = agent.get_state()
            before_position = np.asarray(before.position, dtype=np.float64)
            distance = float(np.linalg.norm(target[[0, 2]] - before_position[[0, 2]]))
            if distance <= WAYPOINT_GOAL_RADIUS_M:
                break
            delta = signed_angle_deg(heading_deg(before.rotation), target_heading_deg(before_position, target))
            action = 'turn_right' if delta > TURN_DEG / 2 else 'turn_left' if delta < -TURN_DEG / 2 else 'move_forward'
            observation = sim.step(action)
            after = agent.get_state()
            action_type = {'turn_right': 'TURN_RIGHT', 'turn_left': 'TURN_LEFT', 'move_forward': 'MOVE_FORWARD'}[action]
            if action == 'move_forward':
                moved = float(np.linalg.norm(np.asarray(after.position)[[0, 2]] - before_position[[0, 2]]))
                stagnant_moves = stagnant_moves + 1 if moved < 1e-05 else 0
                if stagnant_moves >= 3:
                    raise RuntimeError(f'Agent is blocked before reference waypoint {waypoint_index}; cannot render a faithful controller path.')
            step += 1
            actions.append(action_record(step, action_type, before, after, waypoint_index, target))
            current_frame = rgb(observation['rgb'])
            writer.write(current_frame, agent.get_state().sensor_states['rgb'])
        else:
            raise RuntimeError(f'Exceeded controller action budget while approaching reference waypoint {waypoint_index}.')
    for _ in range(FINAL_HOLD_EXTRA_FRAMES):
        writer.write(current_frame, agent.get_state().sensor_states['rgb'])
    return actions

def normalized_rotation(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Episode start_rotation must be an XYZW quaternion.')
    return [float(component) for component in value]

def output_stem(episode: dict[str, Any], index: int) -> str:
    scene_id = scan_id(str(first_present(episode, 'scene_id', 'scene')))
    episode_id = str(first_present(episode, 'episode_id', 'id') or index)
    return f'r2r_{index:04d}_{scene_id}_episode-{episode_id}'

def load_existing_outputs(out_root: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Load complete video/sidecar pairs so interrupted batches can resume safely."""
    emitted: list[dict[str, Any]] = []
    used_trajectories: set[tuple[str, str]] = set()
    for sidecar_path in sorted(out_root.glob('*.actions.json')):
        payload = read_json(sidecar_path)
        episode = payload.get('episode') or {}
        video = payload.get('video') or {}
        video_name = video.get('file')
        scene = str(episode.get('scene_id') or '')
        episode_id = str(episode.get('episode_id') or '')
        if not video_name or not scene or (not episode_id):
            raise ValueError(f'Incomplete resume sidecar: {sidecar_path}')
        video_path = out_root / str(video_name)
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f'Resume sidecar has no complete video: {sidecar_path}')
        actions = payload.get('actions') or []
        key = trajectory_key(episode, extract_reference_path(payload))
        emitted.append({'file': video_path.name, 'action_file': sidecar_path.name, 'scene_id': scene, 'episode_id': episode.get('episode_id'), 'trajectory_id': episode.get('trajectory_id'), 'action_count': len(actions)})
        if key in used_trajectories:
            raise ValueError(f'Duplicate trajectory in resume directory: {key}')
        used_trajectories.add(key)
    return (emitted, used_trajectories)

def render_episode(habitat_sim: Any, config: RenderConfig, episode: dict[str, Any], index: int) -> dict[str, Any]:
    source_scene_id = str(first_present(episode, 'scene_id', 'scene'))
    scene_id = scan_id(source_scene_id)
    reference_path = extract_reference_path(episode)
    start_position = xyz(first_present(episode, 'start_position', 'start_position_xyz_m'), 'start_position')
    start_rotation = normalized_rotation(first_present(episode, 'start_rotation', 'start_rotation_xyzw'))
    stem = output_stem(episode, index)
    video_path = config.out_root / f'{stem}.mp4'
    sidecar_path = config.out_root / f'{stem}.actions.json'
    if (video_path.exists() or sidecar_path.exists()) and (not config.overwrite):
        raise FileExistsError(f'Output exists for {stem}; pass --overwrite to replace it.')
    sim = make_simulator(habitat_sim, scene_file(config.scenes_dir, source_scene_id), config.gpu_device_id)
    writer = VideoWriter(video_path)
    render_failed = False
    try:
        actions = execute_reference_path(sim, start_position, start_rotation, reference_path, writer)
        if writer.max_near_black_ratio > config.max_near_black_ratio:
            raise RuntimeError(f'Rendered view contains excessive missing mesh or black pixels: max_near_black_ratio={writer.max_near_black_ratio:.4f} > threshold={config.max_near_black_ratio:.4f}.')
    except Exception:
        render_failed = True
        raise
    finally:
        try:
            writer.close()
        finally:
            sim.close()
            if render_failed:
                preserve(video_path)
                preserve(sidecar_path)
    payload = {'schema_version': 'r2r-canonical-macro-actions-v1', 'episode': {'episode_id': first_present(episode, 'episode_id', 'id'), 'scene_id': scene_id, 'trajectory_id': first_present(episode, 'trajectory_id'), 'instruction': instruction_text(episode), 'start_position_xyz_m': start_position.round(9).tolist(), 'start_rotation_xyzw': start_rotation, 'goal_position_xyz_m': reference_path[-1].round(9).tolist()}, 'reference_path': [], 'action_vocabulary': ['GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'TURN_BACK'], 'canonical_actions': [], 'ordered_action_sequence': [], 'controller_action_sequence': [row['action_type'] for row in actions], 'actions': actions, 'source': {'annotations': str(config.episodes_path), 'scene': str(scene_file(config.scenes_dir, source_scene_id))}, 'video': {'file': video_path.name, 'fps': FPS, 'width': WIDTH, 'height': HEIGHT}, 'rendering': {'simulator': 'Habitat-Sim', 'scene_dataset': 'Matterport3D', 'fps': FPS, 'width': WIDTH, 'height': HEIGHT, 'hfov_deg': HFOV_DEG, 'sensor_height_m': SENSOR_HEIGHT_M, 'move_forward_m': FORWARD_STEP_M, 'turn_angle_deg': TURN_DEG, 'waypoint_goal_radius_m': WAYPOINT_GOAL_RADIUS_M, 'initial_hold_frame_count': INITIAL_HOLD_FRAMES, 'final_hold_extra_frame_count': FINAL_HOLD_EXTRA_FRAMES, 'max_near_black_pixel_ratio': round(writer.max_near_black_ratio, 6), 'max_allowed_near_black_pixel_ratio': config.max_near_black_ratio, 'video': video_path.name}}
    canonical_actions, canonical_waypoints = canonical_route_metadata(start_rotation, reference_path, actions)
    payload['canonical_actions'] = canonical_actions
    payload['ordered_action_sequence'] = [event['action_type'] for event in canonical_actions]
    payload['reference_path'] = canonical_waypoints
    from videovera.data.navigation.actions import file_sha256
    payload['video']['frame_count'] = len(writer.camera_frames)
    payload['camera_trajectory'] = {'basis': 'measured_camera_pose', 'coordinate_system': 'habitat_y_up_camera_minus_z_xyzw', 'capture_source': 'Habitat-Sim rgb sensor state captured with every encoded RGB frame', 'video_sha256': file_sha256(video_path), 'frames': writer.camera_frames}
    write_json(sidecar_path, payload)
    return {'file': video_path.name, 'action_file': sidecar_path.name, 'scene_id': scene_id, 'episode_id': payload['episode']['episode_id'], 'trajectory_id': payload['episode']['trajectory_id'], 'action_count': len(actions)}

def main() -> None:
    config = parse_args()
    episodes = load_episodes(config.episodes_path)
    selected = select_manifest_episodes(config.selection_manifest, episodes, config.max_episodes) if config.selection_manifest else select_episodes(config, episodes)
    habitat_sim = require_runtime(config, selected)
    config.out_root.mkdir(parents=True, exist_ok=True)
    emitted, used_trajectories = load_existing_outputs(config.out_root) if config.resume else ([], set())
    if len(emitted) > config.max_episodes:
        raise ValueError(f'Resume directory contains {len(emitted)} records, exceeding --max-episodes={config.max_episodes}.')
    if emitted:
        print(f'Resuming from {len(emitted)}/{config.max_episodes} complete records.')
    failures = []
    for candidate_index, episode in enumerate(selected, start=1):
        if len(emitted) >= config.max_episodes:
            break
        candidate_key = trajectory_key(episode)
        if candidate_key in used_trajectories:
            continue
        output_index = config.output_index_offset + (candidate_index if config.selection_manifest else len(emitted) + 1)
        try:
            row = render_episode(habitat_sim, config, episode, output_index)
        except Exception as exc:
            failures.append({'episode_id': first_present(episode, 'episode_id', 'id'), 'trajectory_id': first_present(episode, 'trajectory_id'), 'scene_id': scan_id(str(first_present(episode, 'scene_id', 'scene') or '')), 'reason': f'{type(exc).__name__}: {exc}'})
            print(f'[{candidate_index}/{len(selected)}] skipped candidate: {type(exc).__name__}: {exc}')
            continue
        emitted.append(row)
        print(f"[{len(emitted)}/{config.max_episodes}] rendered {row['file']} ({row['action_count']} controller actions)")
    if len(emitted) != config.max_episodes:
        raise RuntimeError(f'Rendered only {len(emitted)}/{config.max_episodes} trajectories after {len(failures)} failures.')
    write_json(config.out_root / 'manifest.json', {'schema_version': 'r2r-habitat-controller-v1', 'episodes_source': str(config.episodes_path), 'scene_root': str(config.scenes_dir), 'videos': emitted, 'skipped_candidates': failures})
    print(json.dumps({'out_root': str(config.out_root), 'episodes': len(emitted)}, indent=2))
if __name__ == '__main__':
    main()
