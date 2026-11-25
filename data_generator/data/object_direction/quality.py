"""Final data-construction component: data/object_direction/quality.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import math
ROLE_KEYS = ('observer_object', 'facing_object', 'target_object')

def role_signature(row):
    grounding = row.get('grounding') or row.get('metadata', {}).get('grounding', {})
    return (row['scene_id'], *(str(grounding[k]['uid']) for k in ROLE_KEYS))

def target_extent_check(grounding, margin=0.1):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Extent margin must be finite and non-negative')
    origin, facing, target = [grounding[k] for k in ROLE_KEYS]
    dx, dy = [facing['centroid'][i] - origin['centroid'][i] for i in (0, 1)]
    length = math.hypot(dx, dy)
    dims = target['axes_lengths_m'][:2]
    coordinates = [v for obj in (origin, facing, target) for v in obj['centroid']]
    if not all((math.isfinite(v) for v in coordinates + list(dims))) or length < 1e-08 or min(dims) <= 0:
        raise ValueError('Invalid horizontal object geometry')
    forward = (dx / length, dy / length)
    right = (forward[1], -forward[0])
    offset = [target['centroid'][i] - origin['centroid'][i] for i in (0, 1)]
    intervals = {}
    clearances = []
    for name, axis in [('front', forward), ('right', right)]:
        center = sum((a * b for a, b in zip(axis, offset)))
        radius = sum((abs(a) * d / 2 for a, d in zip(axis, dims)))
        intervals[name] = [center - radius, center + radius]
        clearances.append(abs(center) - radius)
    clearance = min(clearances)
    return dict(passed=clearance > margin, min_boundary_clearance_m=clearance, intervals_m=intervals, required_margin_m=margin, basis='Conservative projection of horizontal world-axis-aligned target box; not exact occupied mesh')

def context_metrics(role_hits):
    sets = [{h.get('frame_id', h['timestamp']) for h in seen} for seen in role_hits]
    counts = [len(sets[a] & sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    edges = sum((n >= 2 for n in counts))
    shared = set.intersection(*sets) if sets else set()
    return dict(pair_shared_frame_counts=counts, connected=edges >= 2, co_visible_pair_count=edges, observer_facing_co_visible=counts[0] >= 2, triple_shared_frame_count=len(shared), all_roles_co_visible=bool(shared))
