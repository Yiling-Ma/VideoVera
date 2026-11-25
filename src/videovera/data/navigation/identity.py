"""Original reference-geometry identity, separate from measured navmesh identity."""
import copy
import hashlib
import json
from pathlib import Path


def adapt_episode(row):
    result = copy.deepcopy(row)
    if 'trajectory_id' not in result:
        result['trajectory_id'] = 'follower_episode_' + str(result['episode_id'])
    return result


def reference_key(path):
    d = json.loads(Path(path).read_text())
    points = [[round(float(v), 3) for v in p['position_xyz_m']] for p in d['reference_path']]
    points = min(points, list(reversed(points)))
    return hashlib.sha256(json.dumps([d['episode']['scene_id'], points], sort_keys=True).encode()).hexdigest()


def unique_sources(paths, used=()):
    seen = {reference_key(path) for path in used}
    result = []
    for path in paths:
        key = reference_key(path)
        if key not in seen:
            seen.add(key); result.append(path)
    return result
