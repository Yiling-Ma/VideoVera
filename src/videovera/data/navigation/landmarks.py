"""Final data-construction component: data/navigation/landmarks.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any
from videovera.data.navigation.actions import extract_route, UncertainRoute
DEFAULT_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
DEFAULT_BASE_URL = 'http://127.0.0.1:8000/v1'

def landmark_core_label(label: str) -> str:
    normalized = ' '.join(label.lower().split())
    return re.split('\\s+(?:beside|next to|near|under|above|behind|in front of|with|by|against|on)\\s+', normalized, maxsplit=1)[0]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--model', default=os.environ.get('ROUTE_LANDMARK_MODEL', DEFAULT_MODEL))
    parser.add_argument('--base-url', default=os.environ.get('ROUTE_LANDMARK_BASE_URL', DEFAULT_BASE_URL))
    parser.add_argument('--api-key', default=os.environ.get('ROUTE_LANDMARK_API_KEY', 'EMPTY'))
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=220)
    parser.add_argument('--timeout', type=int, default=90)
    parser.add_argument('--min-confidence', type=float, default=0.7)
    parser.add_argument('--multi-view-offset-sec', type=float, default=0.0, help='Also inspect frames this many seconds before and after each waypoint.')
    parser.add_argument('--max-videos', type=int, default=0, help='0 processes every video.')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--error-mode', choices=['fail', 'skip'], default='fail')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error('--num-shards must be positive')
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('--shard-index must be in [0, --num-shards)')
    return args

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def assert_service_available(base_url: str, timeout: int) -> None:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", method='GET')
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 8)) as response:
            if response.status != 200:
                raise RuntimeError(f'landmark service returned HTTP {response.status}')
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f'landmark service is unavailable at {base_url}: {exc}') from exc

def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path), '-vf', f'select=eq(n\\,{max(0, frame_index)})', '-frames:v', '1', '-q:v', '2', str(output_path)], check=True)

def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'

def extract_json_object(text: str) -> dict[str, Any]:
    value = str(text or '').strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1] if '\n' in value else ''
        if value.rstrip().endswith('```'):
            value = value.rstrip()[:-3]
    start = value.find('{')
    end = value.rfind('}')
    if start < 0 or end <= start:
        raise RuntimeError(f'landmark model returned no JSON object: {text!r}')
    try:
        parsed = json.loads(value[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'landmark model returned invalid JSON: {text!r}') from exc
    if not isinstance(parsed, dict):
        raise RuntimeError('landmark response must be a JSON object')
    return parsed

def response_text(response_payload: dict[str, Any]) -> str:
    content = response_payload.get('choices', [{}])[0].get('message', {}).get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join((str(block.get('text') or '') for block in content if isinstance(block, dict)))
    return str(content or '')

def normalize_label(value: str) -> str:
    normalized = ''.join((char.lower() if char.isalnum() or char.isspace() else ' ' for char in value))
    return ' '.join(normalized.split())

def landmark_family(candidate: dict[str, Any]) -> str:
    """Map paraphrases of the same landmark kind to one identity family."""
    label = normalize_label(str(candidate.get('label') or ''))
    token_groups = {'artwork': {'painting', 'paintings', 'art', 'artwork', 'picture', 'portrait'}, 'doorway': {'door', 'doors', 'doorway', 'passageway', 'archway'}, 'stairs': {'stairs', 'staircase', 'steps'}, 'bed': {'bed', 'beds', 'headboard'}, 'seating': {'chair', 'chairs', 'armchair', 'sofa', 'couch', 'lounge', 'ottoman'}, 'storage': {'cabinet', 'cabinets', 'wardrobe', 'shelf', 'shelves', 'bookcase', 'bookshelf'}, 'appliance': {'refrigerator', 'microwave', 'dishwasher', 'washer', 'washing'}, 'table': {'table', 'countertop', 'counter'}, 'plant': {'plant', 'plants', 'planter'}, 'window': {'window', 'windows'}}
    tokens = set(label.split())
    for family, keywords in token_groups.items():
        if tokens & keywords:
            return family
    category = normalize_label(str(candidate.get('category') or ''))
    if category and category not in {'object', 'furniture', 'architectural feature'}:
        return category
    ignored = {'large', 'small', 'white', 'black', 'wooden', 'framed', 'with', 'beside', 'near', 'above', 'under', 'on', 'left', 'right'}
    informative = [token for token in label.split() if token not in ignored]
    return informative[-1] if informative else label

def request_landmarks(image_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = 'Inspect this egocentric indoor-navigation frame. Return JSON only:\n{"landmarks":[{"label":"2-6 word visible landmark phrase","category":"object category","screen_region":"left|center|right","confidence":0.0}]}\n\nList at most three concrete, stable, visibly grounded navigation landmarks. Prefer distinctive objects or object-plus-context phrases,\nsuch as "dining table beside mirror" or "washing machine under shelf". Do not name rooms, walls, floors, generic locations,\ncoordinates, frame numbers, route actions, or unseen objects. Do not use any route instruction. Do not guess.'
    payload = {'model': args.model, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': image_data_url(image_path)}}]}], 'temperature': args.temperature, 'max_tokens': args.max_tokens}
    request = urllib.request.Request(f"{args.base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode('utf-8'), headers={'Authorization': f'Bearer {args.api_key}', 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response_payload = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'landmark model HTTP {exc.code}: {detail}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'landmark model unavailable: {exc}') from exc
    landmarks = extract_json_object(response_text(response_payload)).get('landmarks')
    if not isinstance(landmarks, list):
        return []
    clean = []
    seen = set()
    for landmark in landmarks[:3]:
        if not isinstance(landmark, dict):
            continue
        label = ' '.join(str(landmark.get('label') or '').split())
        normalized = normalize_label(label)
        if not normalized or normalized in seen or (not 2 <= len(normalized.split()) <= 6):
            continue
        seen.add(normalized)
        try:
            confidence = float(landmark.get('confidence') or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        clean.append({'label': label, 'category': ' '.join(str(landmark.get('category') or '').split()) or None, 'screen_region': str(landmark.get('screen_region') or 'unknown').strip().lower(), 'confidence': round(max(0.0, min(1.0, confidence)), 3)})
    return clean

def waypoint_frame(sidecar: dict[str, Any], waypoint_index: int) -> int:
    events = [event for event in sidecar.get('canonical_actions') or [] if isinstance(event, dict)]
    if waypoint_index == 0:
        return min((int(event.get('video_frame_start', 0)) for event in events), default=0)

    def event_waypoint_index(event: dict[str, Any], key: str) -> int | None:
        value = event.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    matching = [event for event in events if event_waypoint_index(event, 'to_waypoint_index') == waypoint_index and event.get('action_type') == 'GO_STRAIGHT']
    if not matching:
        matching = [event for event in events if event_waypoint_index(event, 'to_waypoint_index') == waypoint_index]
    if not matching:
        straight_events = [event for event in events if event.get('action_type') == 'GO_STRAIGHT']
        straight_offset = waypoint_index - 1
        if 0 <= straight_offset < len(straight_events):
            matching = [straight_events[straight_offset]]
    if not matching:
        raise ValueError(f'No canonical video interval reaches waypoint {waypoint_index}')
    return max(0, int(matching[-1]['video_frame_end']) - 1)

def enrich_sidecar(sidecar: dict[str, Any], video_path: Path, frame_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    try:
        pose_route = extract_route(sidecar, video_path)
    except UncertainRoute as exc:
        return {'metadata_version': 'route-plan-landmarks-pose-v1', 'status': 'blocked', 'reason': str(exc), 'anchors': []}
    fps = float((sidecar.get('video') or {}).get('fps') or 8.0)
    frame_count = int((sidecar.get('video') or {}).get('frame_count') or 0)
    anchors = []
    for waypoint in pose_route['boundaries']:
        waypoint_index = int(waypoint['waypoint_index'])
        try:
            frame_index = waypoint['frame_index']
        except ValueError as exc:
            anchors.append({'waypoint_index': waypoint_index, 'role': str(waypoint.get('role') or ('start' if waypoint_index == 0 else 'intermediate')), 'canonical_video_frame': None, 'timestamp_sec': None, 'frame_path': None, 'landmark_candidates': [], 'error': str(exc)})
            continue
        offset = max(0, int(round(args.multi_view_offset_sec * fps)))
        observation_frames = sorted({max(0, min(frame_count - 1, frame_index + delta)) for delta in (-offset, 0, offset)}) if frame_count > 0 else [max(0, frame_index)]
        candidates = []
        errors = []
        observation_paths = []
        for observation_frame in observation_frames:
            frame_path = frame_root / f'waypoint_{waypoint_index:03d}_frame_{observation_frame:06d}.jpg'
            extract_frame(video_path, observation_frame, frame_path)
            observation_paths.append(str(frame_path))
            try:
                candidates.extend(request_landmarks(frame_path, args))
            except Exception as exc:
                if args.error_mode == 'fail':
                    raise
                errors.append(str(exc))
        deduplicated = {}
        for candidate in candidates:
            key = normalize_label(candidate['label'])
            if key not in deduplicated or candidate['confidence'] > deduplicated[key]['confidence']:
                deduplicated[key] = candidate
        candidates = list(deduplicated.values())
        error = '; '.join(errors) or None
        anchors.append({'waypoint_index': waypoint_index, 'role': str(waypoint.get('role') or ('start' if waypoint_index == 0 else 'intermediate')), 'canonical_video_frame': frame_index, 'timestamp_sec': round(frame_index / fps, 3), 'frame_path': observation_paths[len(observation_paths) // 2], 'observation_frames': observation_frames, 'observation_frame_paths': observation_paths, 'landmark_candidates': candidates, 'error': error})
    label_counts = Counter((normalize_label(candidate['label']) for anchor in anchors for candidate in anchor['landmark_candidates']))
    usable = 0
    selected_families: set[str] = set()
    selected_cores: set[str] = set()
    for anchor in anchors:
        selected = None
        for candidate in anchor['landmark_candidates']:
            normalized = normalize_label(candidate['label'])
            family = landmark_family(candidate)
            core = landmark_core_label(candidate['label'])
            candidate['identity_family'] = family
            candidate['identity_core'] = core
            candidate['unique_within_route'] = label_counts[normalized] == 1
            candidate['distinct_identity_within_route'] = family not in selected_families
            candidate['usable_for_route_plan'] = candidate['confidence'] >= args.min_confidence and candidate['unique_within_route'] and candidate['distinct_identity_within_route'] and (core not in selected_cores)
            if selected is None and candidate['usable_for_route_plan']:
                selected = candidate
        anchor['selected_landmark'] = selected
        if selected is not None:
            selected_families.add(str(selected['identity_family']))
            selected_cores.add(str(selected['identity_core']))
            usable += 1
    required_usable = len(anchors)
    status = 'ok' if usable >= required_usable else 'insufficient_usable_landmarks'
    spans = [(event.get('action_type'), event.get('video_frame_start'), event.get('video_frame_end')) for event in sidecar.get('canonical_actions') or [] if isinstance(event, dict)]
    return {'metadata_version': 'route-plan-landmarks-pose-v1', 'pose_signature': pose_route['pose_signature'], 'macro_action_signature': hashlib.sha256(json.dumps(spans).encode()).hexdigest(), 'status': status, 'provider': 'openai_compatible', 'model': args.model, 'source': 'visual inspection of canonical RGB waypoint frames', 'minimum_confidence': args.min_confidence, 'anchors': anchors, 'validation': {'landmarks_are_visual_only': True, 'landmarks_do_not_use_r2r_instruction': True, 'usable_anchor_count': usable}}

def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / 'manifest.json'
    manifest = read_json(manifest_path)
    videos = [row for row in manifest.get('videos') or [] if isinstance(row, dict)]
    videos = [row for global_index, row in enumerate(videos) if global_index % args.num_shards == args.shard_index]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    assert_service_available(args.base_url, args.timeout)
    frame_root = dataset_dir / 'route_plan_landmark_frames'
    statuses: Counter[str] = Counter()
    for index, row in enumerate(videos, start=1):
        sidecar_name = str(row.get('action_file') or row.get('canonical_action_file') or '')
        video_name = str(row.get('file') or '')
        sidecar_path = dataset_dir / sidecar_name
        video_path = dataset_dir / video_name
        if not sidecar_path.is_file() or not video_path.is_file():
            raise FileNotFoundError(f'Missing canonical pair: {sidecar_path}, {video_path}')
        sidecar = read_json(sidecar_path)
        if sidecar.get('route_plan_landmarks', {}).get('pose_signature') and (not args.overwrite):
            try:
                current = extract_route(sidecar, video_path)['pose_signature']
            except UncertainRoute:
                current = None
            if sidecar['route_plan_landmarks']['pose_signature'] != current:
                sidecar.pop('route_plan_landmarks')
        if sidecar.get('route_plan_landmarks', {}).get('pose_signature') and (not args.overwrite):
            status = str(sidecar['route_plan_landmarks'].get('status') or 'existing')
            statuses[status] += 1
            print(f'[{index}/{len(videos)}] retained {video_name}: {status}', flush=True)
            continue
        sidecar['route_plan_landmarks'] = enrich_sidecar(sidecar, video_path, frame_root / sidecar_path.name.removesuffix('.actions.json'), args)
        write_json(sidecar_path, sidecar)
        status = str(sidecar['route_plan_landmarks']['status'])
        statuses[status] += 1
        print(f'[{index}/{len(videos)}] enriched {video_name}: {status}', flush=True)
    manifest['route_plan_landmark_metadata'] = {'metadata_version': 'route-plan-landmarks-v3', 'enabled': True, 'model': args.model, 'frame_directory': str(frame_root), 'status_counts': dict(sorted(statuses.items()))}
    write_json(manifest_path, manifest)
    print(json.dumps({'dataset_dir': str(dataset_dir), 'videos': len(videos), 'status_counts': dict(statuses)}, indent=2))
if __name__ == '__main__':
    main()
