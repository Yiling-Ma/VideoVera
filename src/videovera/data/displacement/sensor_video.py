"""Final data-construction component: data/displacement/sensor_video.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path
import numpy as np

def read_exact(source, length):
    value = source.read(length)
    if len(value) != length:
        raise ValueError('Truncated ScanNet .sens file')
    return value

def read_sensor(path, source_fps=None, force_frame_time=False):
    frames = []
    size = path.stat().st_size
    with path.open('rb') as source:

        def unpack(fmt):
            return struct.unpack('<' + fmt, read_exact(source, struct.calcsize('<' + fmt)))
        if unpack('I')[0] != 4:
            raise ValueError('Only ScanNet .sens version 4 is supported')
        read_exact(source, unpack('Q')[0])
        calibration = [list(unpack('16f')) for _ in range(4)]
        color, depth = unpack('ii')
        width, height, _, _ = unpack('4I')
        unpack('f')
        count = unpack('Q')[0]
        if color != 2:
            raise ValueError('Expected ScanNet JPEG color stream')
        if count < 2 or count > size // 96:
            raise ValueError('Invalid sensor frame count')
        for i in range(count):
            pose = list(unpack('16f'))
            stamp, _, color_size, depth_size = unpack('4Q')
            frames.append({'frame_id': i, 'timestamp_us': stamp, 'offset': source.tell(), 'size': color_size, 'camera_to_world': pose if np.isfinite(pose).all() else None})
            if source.tell() + color_size + depth_size > size:
                raise ValueError('Truncated sensor payload')
            source.seek(color_size + depth_size, 1)
    stamps = [f['timestamp_us'] for f in frames]
    missing = all((t == 0 for t in stamps))
    if force_frame_time or missing:
        if source_fps is None or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError('Missing/overridden timestamps require explicit --source-fps')
        reason = 'timestamps_explicitly_ignored' if force_frame_time else 'timestamps_missing'
        basis = f'assumed_source_fps_{source_fps:g}_{reason}'
        for f in frames:
            f['timestamp'] = f['frame_id'] / source_fps
    else:
        if any((b <= a for a, b in zip(stamps, stamps[1:]))):
            raise ValueError('Non-monotonic timestamps; use --source-fps and --force-frame-time only if justified')
        basis = 'sensor_color_timestamp_microseconds'
        for f in frames:
            f['timestamp'] = (f['timestamp_us'] - stamps[0]) / 1000000.0
    return (frames, {'width': width, 'height': height, 'time_basis': basis, 'intrinsic_color': calibration[0], 'extrinsic_color': calibration[1], 'camera_axes': '+X right, +Y down, +Z forward', 'world_up': '+Z'})

def render_clip(sensor, frames, output, fps=10, width=960):
    """Retain both endpoint frames and source timing, including intermediate views."""
    selected = [frames[0]]
    for f in frames[1:-1]:
        if f['timestamp'] - selected[-1]['timestamp'] >= 1 / fps - 1e-06:
            selected.append(f)
    selected.append(frames[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temp_output = output.with_name(output.stem + '.partial.mp4')
    with tempfile.TemporaryDirectory(prefix='etd_rgb_') as cache:
        cache = Path(cache)
        concat = ['ffconcat version 1.0']
        with sensor.open('rb') as source:
            for i, f in enumerate(selected):
                source.seek(f['offset'])
                (cache / f'{i:06d}.jpg').write_bytes(read_exact(source, f['size']))
                dt = selected[i + 1]['timestamp'] - f['timestamp'] if i + 1 < len(selected) else 1 / fps
                concat += [f"file '{i:06d}.jpg'", f'duration {dt:.6f}']
        concat.append(f"file '{len(selected) - 1:06d}.jpg'")
        listing = cache / 'frames.ffconcat'
        listing.write_text('\n'.join(concat) + '\n')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-safe', '0', '-i', str(listing), '-vf', f'scale={width}:-2', '-vsync', 'vfr', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp_output)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=width,height', '-of', 'json', str(temp_output)]))
    duration = frames[-1]['timestamp'] - frames[0]['timestamp']
    if abs(float(probe['format']['duration']) - duration) > 0.5:
        raise ValueError('Rendered clip duration differs from pose interval')
    temp_output.replace(output)
    return {'source_start_sec': frames[0]['timestamp'], 'source_end_sec': frames[-1]['timestamp'], 'source_start_frame': frames[0]['frame_id'], 'source_end_frame': frames[-1]['frame_id'], 'duration_sec': duration, 'encoded_duration_sec': float(probe['format']['duration']), 'sampled_frames': len(selected), 'sampling_fps': fps, 'probe': probe, 'render_timestamp_tolerance_sec': 0.04, 'timeline': 'continuous_source_interval_with_timestamp_preserving_sampling'}
