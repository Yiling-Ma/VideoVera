"""Dense continuous-window search with unchanged landmark visibility gates."""
import math
from dataclasses import replace
from types import SimpleNamespace
from data_generator.data.landmark_routes import generate as A
FROZEN_BOUNDS=A.optimized_continuous_clip_bounds
FROZEN_ROUTES=A.candidate_routes
FROZEN_QUALITY=A.candidate_video_quality
def representatives(anchors, rows, camera_path, start, end, phase):
    """One strictly observed instance per unique named object; never synthesize frames."""
    result = []
    for anchor in anchors:
        hits = [i for i in anchor.visible_frame_indices if start <= rows[i].timestamp <= end]
        if len(hits) < 3:
            continue
        index = hits[int(round((len(hits) - 1) * phase))]
        xy = camera_path[index]
        if not all((math.isfinite(v) for v in xy)):
            continue
        result.append(replace(anchor, frame_index=index, timestamp=rows[index].timestamp, camera_xy=tuple(xy), visible_frame_count=len(hits), visible_frame_indices=tuple(hits)))
    return sorted(result, key=lambda x: (x.frame_index, x.label))
windows=SimpleNamespace(representatives=representatives)
def context(rows, poses):
    matched = [A.nearest_pose(r.timestamp, poses) for r in rows]
    pitch = [abs(A.camera_pitch_deg(p)) if p else 180 for p in matched]
    rot = [A.sky_up_rotation_deg(p) if p else None for p in matched]
    prefix = lambda v: A.np.r_[0, A.np.cumsum(v)]
    return (A.np.array([r.timestamp for r in rows]), prefix([p is None for p in matched]), prefix([False] + [b.timestamp - a.timestamp <= 0 or b.timestamp - a.timestamp > 0.2 for a, b in zip(rows, rows[1:])]), prefix([False] + [a != b for a, b in zip(rot, rot[1:])]), prefix([p <= 45 for p in pitch]), pitch)

def strict(candidate, q):
    return q['valid'] and all((sum((q['start_frame_index'] <= i <= q['end_frame_index'] for i in a.visible_frame_indices)) >= 3 for a in [*candidate['route'], candidate['facing']]))

def quality(candidate, rows, poses, pre, scene, blocked):
    q = FROZEN_QUALITY(candidate, rows, poses, 0.5, 8, 40, 0.2, 0.75, 70, 24, 1)
    if strict(candidate, q) and (scene, q['start_frame_index'], q['end_frame_index']) not in blocked:
        return q
    return alternate(candidate, rows, poses, pre, scene, blocked)

def alternate(c, rows, poses, pre, scene, blocked):
    anchors = [*c['route'], c['facing']]
    ix = [a.frame_index for a in anchors]
    bs, be = (min(ix), max(ix))
    dt = float(A.np.median(A.np.diff([r.timestamp for r in rows])))
    extra = max(0, math.ceil(8 / dt) + 1 - (be - bs + 1))
    margin = max(round(0.5 / dt), round(2 / dt))
    left = max(0, bs - margin - extra)
    right = min(len(rows) - 1, be + margin + extra)
    stamps, invalid, gaps, rotchanges, horizontal, pitch = pre
    visible = [set(a.visible_frame_indices) for a in anchors]
    for start in range(left, bs + 1):
        for end in range(max(be, start + math.ceil(8 / dt)), min(right, start + math.floor(40 / dt)) + 1):
            duration = stamps[end] - stamps[start]
            if not 8 <= duration <= 40:
                continue
            if invalid[end + 1] - invalid[start] or gaps[end + 1] - gaps[start + 1] or rotchanges[end + 1] - rotchanges[start + 1]:
                continue
            if (horizontal[end + 1] - horizontal[start]) / (end - start + 1) < 0.75 or float(A.np.percentile(pitch[start:end + 1], 90)) > 70:
                continue
            if any((sum((start <= i <= end for i in vs)) < 3 for vs in visible)):
                continue
            sampled = A.select_evidence_frame_indices(c, start, end, 24, 1)
            if any((not any((i in vs for i in sampled)) for vs in visible)):
                continue
            if (scene, start, end) in blocked:
                continue
            A.optimized_continuous_clip_bounds = lambda *args: (start, end)
            try:
                q = FROZEN_QUALITY(c, rows, poses, 0.5, 8, 40, 0.2, 0.75, 70, 24, 1)
            finally:
                A.optimized_continuous_clip_bounds = FROZEN_BOUNDS
            assert q['valid']
            return q
    return None

def candidates(anchors, rows, path):
    if not rows:
        return
    seen = set()
    sets = [anchors]
    start = rows[0].timestamp
    while start <= rows[-1].timestamp - 8:
        for phase in [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
            opts = windows.representatives(anchors, rows, path, start, start + 40, phase)
            if len(opts) >= 4:
                sets.append(opts)
        start += 5
    for opts in sets:
        for c in FROZEN_ROUTES(opts, path):
            if len(c['turns']) != 2:
                continue
            key = (tuple(((x.object_key, x.frame_index) for x in c['route'])), c['facing'].object_key, c['facing'].frame_index)
            if key not in seen:
                seen.add(key)
                yield c

def main():
    collect=A.collect_visible_anchors
    captured={}
    def capture(*args,**kwargs):
        anchors,rows,path=collect(*args,**kwargs)
        captured['rows']=rows
        return anchors,rows,path
    def routes(anchors,path,include_reverse_routes=False,reverse_routes_only=False):
        if include_reverse_routes or reverse_routes_only:
            raise ValueError('Use forward landmark routes; reverse navigation is a separate task')
        return list(candidates(anchors,captured['rows'],path))
    def gate(candidate,rows,poses,*args,**kwargs):
        return quality(candidate,rows,poses,context(rows,poses),'',set()) or {'valid':False,'rejection_reasons':['no_valid_continuous_window']}
    A.collect_visible_anchors=capture
    A.candidate_routes=routes
    A.candidate_video_quality=gate
    return A.main()

if __name__=='__main__':main()
