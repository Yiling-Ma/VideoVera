from __future__ import annotations

def canonical_gold_trace_claim_type(claim_type: str) -> str:
    mapping = {'object_identification': 'anchor_object', 'spatial_frame': 'facing_anchor', 'spatial_relation': 'spatial_resolution', 'answer_grounding': 'final_quadrant'}
    return mapping.get(claim_type, claim_type or 'other')
