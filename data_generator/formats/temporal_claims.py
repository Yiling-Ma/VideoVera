from __future__ import annotations

def canonical_gold_trace_claim_type(claim_type: str) -> str:
    mapping = {'first_event': 'first_event', 'middle_event': 'middle_event', 'last_event': 'last_event', 'visual_observation': 'visual_observation', 'temporal_relation': 'temporal_relation', 'answer_grounding': 'answer_grounding'}
    return mapping.get(claim_type, claim_type or 'other')
