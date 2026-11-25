"""Final data-construction component: data/navigation/navmesh.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import copy
import math
from typing import Any
import numpy as np
from data_generator.data.navigation.actions import wrap
from data_generator.data.navigation.renderer import SENSOR_HEIGHT_M, VideoWriter, rgb
START_HOLD_FRAMES = 4
END_HOLD_FRAMES = 6
TURN_FRAME_DEG = 15.0
TRAVEL_FRAME_M = 0.25

def yaw_quaternion(heading_deg: float):
    import quaternion
    half = math.radians(heading_deg) / 2.0
    return quaternion.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)

def emit_pose(sim: Any, agent: Any, writer: VideoWriter, camera_position: list[float], heading_deg: float) -> None:
    state = agent.get_state()
    state.position = np.asarray([camera_position[0], camera_position[1] - SENSOR_HEIGHT_M, camera_position[2]], dtype=float)
    state.rotation = yaw_quaternion(heading_deg)
    agent.set_state(state, reset_sensors=True)
    observation = rgb(sim.get_sensor_observations()['rgb'])
    camera_state = agent.get_state().sensor_states['rgb']
    measured = np.asarray(camera_state.position, dtype=float)
    expected = np.asarray(camera_position, dtype=float)
    if float(np.linalg.norm(measured - expected)) > 0.0001:
        raise RuntimeError(f'Camera pose mismatch: expected {expected.tolist()}, measured {measured.tolist()}')
    writer.write(observation, camera_state)

def render_route(sim: Any, route: dict[str, Any], writer: VideoWriter, speed_scale: float) -> None:
    """Render fixed-position turns followed by fixed-heading travel."""
    agent = sim.initialize_agent(0)
    boundaries = route['boundaries']
    current_heading = float(route['start_heading_deg'])
    current_position = list(boundaries[0]['position_xyz_m'])
    for _ in range(START_HOLD_FRAMES):
        emit_pose(sim, agent, writer, current_position, current_heading)
    for leg_index, leg in enumerate(route['legs']):
        target_heading = float(leg['heading_deg'])
        delta = wrap(target_heading - current_heading)
        turn_frames = int(math.ceil(abs(delta) / (TURN_FRAME_DEG * speed_scale)))
        for step in range(1, turn_frames + 1):
            emit_pose(sim, agent, writer, current_position, current_heading + delta * step / turn_frames)
        current_heading = target_heading
        start = np.asarray(boundaries[leg_index]['position_xyz_m'], dtype=float)
        end = np.asarray(boundaries[leg_index + 1]['position_xyz_m'], dtype=float)
        travel_frames = max(4, int(math.ceil(float(leg['distance_m']) / (TRAVEL_FRAME_M * speed_scale))))
        for step in range(1, travel_frames + 1):
            position = (start + (end - start) * step / travel_frames).tolist()
            emit_pose(sim, agent, writer, position, current_heading)
        current_position = end.tolist()
    target_heading = float(route.get('end_heading_deg', current_heading))
    delta = wrap(target_heading - current_heading)
    turn_frames = int(math.ceil(abs(delta) / (TURN_FRAME_DEG * speed_scale)))
    for step in range(1, turn_frames + 1):
        emit_pose(sim, agent, writer, current_position, current_heading + delta * step / turn_frames)
    current_heading = target_heading
    for _ in range(END_HOLD_FRAMES):
        emit_pose(sim, agent, writer, current_position, current_heading)

def reference_path(route: dict[str, Any]) -> list[dict[str, Any]]:
    distances = {int(event['to_waypoint_index']): float(event['distance_m']) for event in route['canonical_actions'] if event['action_type'] == 'GO_STRAIGHT'}
    return [{'waypoint_index': int(boundary['waypoint_index']), 'position_xyz_m': boundary['position_xyz_m'], 'segment_distance_xz_m': distances.get(int(boundary['waypoint_index']), 0.0)} for boundary in route['boundaries']]

def canonical_actions(route: dict[str, Any]) -> list[dict[str, Any]]:
    actions = copy.deepcopy(route['canonical_actions'])
    for event in actions:
        if event['action_type'] == 'GO_STRAIGHT':
            event['segment_indices'] = [int(event['to_waypoint_index'])]
    return actions
