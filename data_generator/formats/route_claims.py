from __future__ import annotations

def canonical_gold_trace_claim_type(claim_type: str) -> str:
    mapping = {'route_start': 'route_start', 'route_segment': 'route_segment', 'reverse_segment': 'route_segment', 'route_destination': 'route_destination', 'turn_pattern': 'turn_pattern', 'shape_resolution': 'shape_resolution', 'answer_grounding': 'answer_grounding'}
    return mapping.get(claim_type, claim_type or 'other')
