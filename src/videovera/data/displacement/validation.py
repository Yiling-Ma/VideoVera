"""Final data-construction component: data/displacement/validation.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from videovera.data.displacement.geometry import TASK_TYPE, Policy, compare_claims, direction_facts, gold_steps

def require(condition, message):
    if not condition:
        raise ValueError(message)

def audit(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, videos, failures, video_bytes = (set(), set(), [], 0)
    for row in rows:
        try:
            require(row['id'] not in ids, 'duplicate_id')
            require(row['video_path'] not in videos, 'duplicate_video')
            ids.add(row['id'])
            videos.add(row['video_path'])
            require(row['task_type'] == TASK_TYPE, 'wrong_task')
            frames = json.loads(Path(row['metadata']['trajectory_path']).read_text())['frames']
            facts = direction_facts(frames, row['grounding']['reference'], Policy(**row['grounding']['policy']))
            require(set(row['choices']) == set('ABCD') and len(set(row['choices'].values())) == 4, 'four_distinct_choices_required')
            require(facts['direction'] == row['answer'] == row['choices'][row['answer_key']], 'label_mismatch')
            result = compare_claims(row, {'answer_key': row['answer_key'], 'reasoning_steps': row['gold_reasoning_steps']}, frames)
            require(not result['reasoning_error_detected'], 'gold_claim_contradiction')
            require(all((c['status'] == 'supported' for c in result['claim_results'])), 'gold_claim_not_supported')
            if row['metadata'].get('slot_index') is not None:
                require(row['gold_reasoning_steps'] == gold_steps(frames, facts, row['answer_key']), 'generated_gold_differs_from_measured_route')
            video = Path(row['video_path'])
            if row.get('dataset_source') == 'r2r':
                import numpy as np
                from videovera.data.displacement.source import normalized_frames, direct_endpoint_components
                from videovera.data.displacement.route_policy import route_policy
                stored_policy = row['metadata'].get('route_policy', {})
                actual_policy = route_policy(row['gold_reasoning_steps'], stored_policy.get('min_turns', 1), stored_policy.get('max_turns', 3))
                require(actual_policy == stored_policy, 'route_policy_mismatch')
                measured_policy = route_policy(gold_steps(frames, facts, row['answer_key']), stored_policy.get('min_turns', 1), stored_policy.get('max_turns', 3))
                require(actual_policy == measured_policy, 'route_policy_disagrees_with_measured_trajectory')
                annotation = Path(row['metadata']['annotation_path'])
                require(hashlib.sha256(annotation.read_bytes()).hexdigest() == row['metadata']['annotation_sha256'], 'source_annotation_hash_mismatch')
                payload = json.loads(annotation.read_text())
                normalized, _ = normalized_frames(payload, video)
                require(len(normalized) == len(frames), 'normalized_frame_count_mismatch')
                require(np.allclose([f['camera_to_world'] for f in normalized], [f['camera_to_world'] for f in frames], atol=1e-08), 'normalized_pose_mismatch')
                require(np.allclose([f['timestamp'] for f in normalized], [f['timestamp'] for f in frames], atol=1e-08), 'normalized_timestamp_mismatch')
                require(np.allclose(direct_endpoint_components(payload, facts['reference']), [facts['front_m'], facts['right_m']], atol=1e-06), 'native_habitat_direction_mismatch')
            video_bytes += video.stat().st_size
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type,width,height', '-of', 'json', str(video)]))
            require(abs(float(probe['format']['duration']) - facts['duration_sec']) < 0.5, 'duration_mismatch')
            require(any((s.get('codec_type') == 'video' for s in probe['streams'])), 'no_video_stream')
            metadata = row['metadata']
            pending = metadata['human_review_required'] and (not metadata['training_ready'])
            released = metadata['training_ready'] and metadata.get('human_review_status') == 'not_individually_reviewed' and (metadata.get('outcome') == 'automated_audit_passed_pilot_design_human_accepted') and metadata.get('quality_gate')
            require(pending or released, 'review_provenance_missing')
        except (AssertionError, KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
            failures.append({'id': row.get('id'), 'error': str(exc)})
    return {'records': len(rows), 'unique_ids': len(ids), 'unique_videos': len(videos), 'answer_counts': dict(Counter((r['answer'] for r in rows))), 'answer_key_counts': dict(Counter((r['answer_key'] for r in rows))), 'reference_counts': dict(Counter((r['grounding']['reference'] for r in rows))), 'physical_scenes': len({r['metadata']['physical_scene_id'] for r in rows}), 'dataset_source_counts': dict(Counter((r.get('dataset_source') for r in rows))), 'turn_count_distribution': dict(Counter((r['metadata']['route_policy']['turn_count'] for r in rows if 'route_policy' in r['metadata']))), 'video_total_bytes': video_bytes, 'failures': failures, 'jsonl_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'geometry_and_asset_checks_passed': bool(rows) and (not failures), 'visual_correctness_verified': False, 'live_model_api_tested': False}
