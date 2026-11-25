"""Final data-construction component: validation/media.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import hashlib
import json
import subprocess
from videovera.io import sha256 as sha

def decoded(path):
    import re
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,avg_frame_rate,width,height:format=duration', '-of', 'json', str(path)]))
    stream = info['streams'][0]
    try:
        n = int(stream['nb_frames'])
    except (KeyError, ValueError):
        count = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-threads', '1', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)]))
        n = int(count['streams'][0]['nb_read_frames'])
    for attempt in range(2):
        if n < 2:
            raise ValueError('Invalid decoded frame count: ' + str(path))
        indices = sorted({round(i * (n - 1) / 23) for i in range(24)})
        expression = '+'.join((f'eq(n\\,{i})' for i in indices))
        graph = f'[0:v]split=2[all][s];[s]select={expression},scale=64:64[thumb]'
        result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-nostats', '-threads', '1', '-i', str(path), '-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[all]', '-vsync', '0', '-f', 'null', '/dev/null', '-map', '[thumb]', '-vsync', '0', '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', 'pipe:1', '-progress', 'pipe:2'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise ValueError('Decode failed: ' + str(path) + ' ' + result.stderr.decode(errors='replace')[-500:])
        counts = re.findall(b'^frame=(\\d+)\\s*$', result.stderr, re.M)
        if not counts:
            raise ValueError('Missing full-decode frame count: ' + str(path))
        actual = int(counts[-1])
        if actual != n:
            n = actual
            continue
        payload = result.stdout
        if len(payload) != len(indices) * 64 * 64 * 3:
            raise ValueError('Incomplete uniform-frame decode: ' + str(path))
        num, den = map(int, stream['avg_frame_rate'].split('/'))
        return dict(decode_version=2, frames=n, sample_count=len(indices), decoded_sha256=hashlib.sha256(payload).hexdigest(), file_sha256=sha(path), width=stream['width'], height=stream['height'], fps=num / den, duration_sec=float(info['format']['duration']))
    raise ValueError('Unstable media frame count: ' + str(path))
