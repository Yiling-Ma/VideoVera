"""ARKitScenes Object Direction: measured depth support and original continuous RGB."""
import ast, argparse, collections, copy, functools, inspect, itertools, json, math, random, subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
from videovera.io import atomic_json as write, sha256 as sha, read_jsonl, write_jsonl
from videovera.data.object_direction import scannet as spatial
from videovera.data.landmark_routes import generate as native
OUT=Path('work/object_direction_arkit')
POLICY = dict(min_component_margin_m=0.25, target_extent_margin_m=0.1, minimum_visible_samples_per_role=3, minimum_support_pixel_area_ratio=0.02, minimum_peak_support_pixel_area_ratio=0.03, max_evidence_span_sec=30, minimum_clip_duration_sec=8, require_connected_role_coview_graph=True, visibility_sampling_min_interval_sec=0.5, additional_native_visibility=dict(bbox_area_min=0.04, bbox_area_max=0.45, min_in_frame_fraction=0.6, min_corner_depth_m=0.1, min_depth_coverage=0.5, min_depth_support_ratio=0.1, depth_tolerance_m=0.12), overlapping_object_support_pixels_excluded=True, visibility_basis='Measured depth consistent with front surface of official 3D OBB; NOT official 2D instance GT', human_approval_required=True, bulk_generation_allowed=False, formal_admission_allowed=False)
def instrument_support():
    """Expose the existing supported pixels without changing its computation."""
    tree = ast.parse(inspect.getsource(native.evaluate_obb_depth_support))
    function = tree.body[0]
    function.name = 'depth_support_with_pixels'
    original_return = function.body[-1]
    assert isinstance(original_return, ast.Return)
    function.body[-1] = ast.Return(value=ast.Tuple(elts=[original_return.value, ast.parse('supported.reshape(iy1-iy0, ix1-ix0)', mode='eval').body, ast.parse('(ix0, iy0, ix1, iy1)', mode='eval').body], ctx=ast.Load()))
    namespace = dict(native.evaluate_obb_depth_support.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<expose-original-depth-support>', 'exec'), namespace)
    return namespace[function.name]

def extent_pass(grounding):
    try:
        return spatial.target_extent_check(grounding, 0.1)['passed']
    except ValueError:
        return False

def mask_for(depth, obj, pose, intrinsic, projection):
    result = support_pixels(depth, obj, pose.world_to_camera, intrinsic, projection, 0.12)
    mask = np.zeros(depth.shape, dtype=bool)
    if isinstance(result, tuple):
        metrics, crop, (x0, y0, x1, y1) = result
        assert int(crop.sum()) == metrics['support_depth_pixels']
        mask[y0:y1, x0:x1] = crop
    else:
        metrics = result
    return (metrics, mask)

@functools.lru_cache(maxsize=48)
def timeline(directory, extension):
    return native.list_timestamped_files(directory, extension)

def frame_support(scene, row, objects, poses):
    paths = native.scene_paths(scene)
    pose = native.nearest_pose(row.timestamp, poses)
    depth_row = native.nearest_timestamped(row.timestamp, timeline(paths['depth'], '.png'), 0.06)
    intr_row = native.nearest_timestamped(row.timestamp, timeline(paths['intrinsics'], '.pincam'), 0.06)
    if pose is None or depth_row is None or intr_row is None:
        return None
    depth = native.load_depth_png_meters(depth_row.path)
    intrinsic = native.scale_intrinsics(native.parse_pincam(intr_row.path, intr_row.timestamp), depth.shape[1], depth.shape[0])
    supports = {}
    votes = np.zeros(depth.shape, dtype=np.uint16)
    for obj in objects:
        projection = native.project_obb(native.obb_corners(obj), pose.world_to_camera, intrinsic, depth.shape[1], depth.shape[0])
        if projection is None:
            continue
        metrics, mask = mask_for(depth, obj, pose, intrinsic, projection)
        supports[obj.object_key] = (metrics, mask, projection)
        votes += mask.astype(np.uint16)
    result = {}
    for key, (metrics, mask, projection) in supports.items():
        unique = mask & (votes == 1)
        count = int(unique.sum())
        ratio = count / max(1, metrics['ray_box_pixels'])
        if not metrics['visible'] or metrics['valid_depth_coverage'] < 0.5 or ratio < 0.1 or (not 0.04 <= projection['bbox_area_ratio'] <= 0.45) or (projection['in_frame_fraction'] < 0.6) or (not projection['all_corners_in_front']) or (projection['min_corner_depth_m'] < 0.1) or (count / depth.size < 0.02):
            continue
        ys, xs = np.where(unique)
        result[key] = dict(pixel_area_ratio=count / depth.size, bbox_area_ratio=count / depth.size, bbox_2d=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], mask_size=[depth.shape[1], depth.shape[0]], support_pixels=count, native_depth_support=metrics, exclusive_depth_support_ratio=ratio, projection=projection, rgb_path=str(row.path), depth_path=str(depth_row.path), intrinsic_path=str(intr_row.path), world_to_camera=pose.world_to_camera.tolist())
    return (result, supports, votes, pose, depth_row, intrinsic)

def scan_scene(scene, meta, return_all=False, cached_hits=None):
    paths = native.scene_paths(scene)
    objects = native.load_arkitscenes_objects(paths['annotation'])
    labels = collections.Counter((obj.label for obj in objects))
    eligible = [obj for obj in objects if labels[obj.label] == 1 and obj.label not in native.EXCLUDED_LABELS]
    if len(eligible) < 3:
        return (None, dict(reason='fewer_than_three_unique_eligible_categories'))
    aligned = [replace(obj, axes_lengths=tuple(np.ptp(native.obb_corners(obj), axis=0)), normalized_axes=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) for obj in eligible]
    geometry_possible = False
    for triple in itertools.combinations(aligned, 3):
        plans = spatial.generate_object_relative_direction(scene.name, paths['annotation'], triple, 6, random.Random(20261108), 0.25, require_role_frame_evidence=False)
        if any((extent_pass(plan['grounding']) for plan in plans)):
            geometry_possible = True
            break
    if not geometry_possible:
        return (None, dict(reason='no_original_geometry_triple'))
    poses = native.parse_trajectory(paths['trajectory'])
    rows = native.list_timestamped_files(paths['rgb'], '.png')
    if len(rows) < 2 or rows[-1].timestamp - rows[0].timestamp < 8:
        return (None, dict(reason='short_source'))
    scene_out = OUT / 'search' / scene.name
    scene_out.mkdir(parents=True, exist_ok=True)
    hits = {obj.object_key: [] for obj in eligible}
    if cached_hits:
        saved = json.loads(Path(cached_hits).read_text())
        if set(saved) != set(hits):
            raise ValueError('Cached object identity differs')
        for values in saved.values():
            for hit in values:
                row = rows[hit['frame_id']]
                pose = native.nearest_pose(row.timestamp, poses)
                if str(row.path) != hit['rgb_path'] or abs(row.timestamp - hit['timestamp']) > 1e-06 or pose is None or (not np.allclose(pose.world_to_camera, hit['world_to_camera'], atol=1e-06)):
                    raise ValueError('Cached calibrated source frame differs')
        hits = saved
    else:
        previous = -math.inf
        for index, row in enumerate(rows):
            if row.timestamp - previous < 0.5:
                continue
            previous = row.timestamp
            frame = frame_support(scene, row, objects, poses)
            if frame is None:
                continue
            for key, value in frame[0].items():
                if key in hits:
                    hits[key].append(dict(value, frame_id=index, timestamp=row.timestamp, visibility_basis=POLICY['visibility_basis']))
    write(scene_out / 'visibility.json', hits)
    selected = []
    for triple in itertools.combinations(aligned, 3):
        window = spatial.common_window([hits[o.object_key] for o in triple], rows[-1].timestamp, 30, 8, 3, 0.03)
        if window is None:
            continue
        _, start, end, subsets = window
        evidence = {obj.object_key: dict(max(seen, key=lambda h: h['pixel_area_ratio']), num_visible_frames=len(seen), visible_frame_samples=seen) for obj, seen in zip(triple, subsets)}
        qa = spatial.generate_object_relative_direction(scene.name, paths['annotation'], triple, 6, random.Random(20261108), 0.25, display_names={obj.object_key: obj.label.replace('_', ' ') for obj in eligible}, frame_evidence=evidence, min_role_bbox_area_ratio=0.02)
        for record in qa:
            if not extent_pass(record['grounding']):
                continue
            extent = spatial.target_extent_check(record['grounding'], 0.1)
            role_keys = [record['grounding'][role]['object_key'] for role in spatial.ROLE_KEYS]
            context = spatial.context_metrics([evidence[key]['visible_frame_samples'] for key in role_keys])
            record['grounding'].update(target_extent_check=extent, visual_context=context)
            interval = [row for row in rows if start <= row.timestamp <= end]
            gaps = np.diff([row.timestamp for row in interval])
            if len(interval) < 2 or max(gaps) > max(0.25, 1.5 * float(np.median(gaps))):
                continue
            selected.append((context['triple_shared_frame_count'], context['co_visible_pair_count'], min((max((h['pixel_area_ratio'] for h in evidence[key]['visible_frame_samples'])) for key in role_keys)), record, window))
    if not selected:
        return (None, dict(reason='no_original_quality_triple', strict_visible_counts={k: len(v) for k, v in hits.items()}))
    selected.sort(key=lambda x: x[:3], reverse=True)
    candidates = [(item[3], item[4], rows, objects, poses, paths, meta) for item in selected]
    return (candidates if return_all else candidates[0], dict(eligible_triples=len(selected), strict_visible_counts={k: len(v) for k, v in hits.items()}))

def make_pack(candidate, ordinal, folder_override=None):
    record, window, rows, objects, poses, paths, meta = candidate
    _, start, end, subsets = window
    folder = Path(folder_override) if folder_override else OUT / ('example_%02d_%s' % (ordinal, meta['video_id']))
    folder.mkdir(parents=True, exist_ok=False)
    interval = [(i, row) for i, row in enumerate(rows) if start <= row.timestamp <= end]
    timestamps = [row.timestamp for _, row in interval]
    dt = float(np.median(np.diff(timestamps)))
    listing = folder / 'source_frames.ffconcat'
    lines = ['ffconcat version 1.0']
    for j, (_, row) in enumerate(interval):
        duration = timestamps[j + 1] - row.timestamp if j + 1 < len(interval) else dt
        lines.extend(["file '%s'" % row.path, 'duration %.9f' % duration])
    lines.append(lines[-2])
    listing.write_text('\n'.join(lines) + '\n')
    clip = folder / 'continuous_source.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-safe', '0', '-i', str(listing), '-vf', 'scale=960:-2', '-r', '10', '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(clip)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height:format=duration', '-of', 'json', str(clip)]))
    duration = float(probe['format']['duration'])
    if duration < 8 or abs(duration - (timestamps[-1] - timestamps[0] + dt)) > 0.5:
        raise ValueError('Continuous clip duration mismatch')
    sample_indices = [round(i * (int(probe['streams'][0]['nb_read_frames']) - 1) / 23) for i in range(24)]
    model_hits = {record['grounding'][role]['object_key']: [] for role in spatial.ROLE_KEYS}
    for sample_index in sample_indices:
        row = min((row for _, row in interval), key=lambda row: abs(row.timestamp - (timestamps[0] + sample_index / 10)))
        support = frame_support(paths['annotation'].parent, row, objects, poses)
        if support:
            for key in model_hits:
                if key in support[0]:
                    model_hits[key].append(dict(video_frame_index=sample_index, source_timestamp=row.timestamp))
    if min((len(values) for values in model_hits.values())) < 3:
        write(folder / 'MODEL_VIEW_REJECTED.json', model_hits)
        raise ValueError('Fewer than three visible samples per role in original fixed24 model view')
    write(folder / 'MODEL_24FRAME_VISIBILITY.json', model_hits)
    record = copy.deepcopy(record)
    record['video_path'] = str(clip)
    record['dataset_source'] = 'arkitscenes'
    record['metadata'] = {'source_split': 'Training', 'physical_scene_id': 'arkitscenes:visit:' + str(meta['visit_id']), 'source_scene_id': str(meta['video_id']), 'quality_policy': POLICY, 'grounding': record['grounding'], 'model_24frame_visibility': model_hits, 'human_review_required': True, 'human_review_complete': False, 'evaluation_valid': False}
    record['human_review_required'] = True
    record['evaluation_valid'] = False
    write(folder / 'qa.json', record)
    write(folder / 'geometry.json', record['grounding'])
    write(folder / 'provenance.json', {'annotation_sha256': sha(paths['annotation']), 'trajectory_sha256': sha(paths['trajectory']), 'clip_sha256': sha(clip), 'source_split': 'Training', 'visit_id': str(meta['visit_id']), 'source_frame_ids': [i for i, _ in interval], 'source_timestamps': timestamps, 'synthetic_views': False})
    return record
support_pixels=instrument_support()

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path,required=True,help='JSON list: directory, scene_id, visit_id, split=Training')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--max-per-physical-scene',type=int,choices=range(1,11),default=10)
    args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
    sources=json.loads(args.source_manifest.read_text())
    benchmark=read_jsonl(args.benchmark)
    excluded={str(r.get('scene_id',r.get('video_id'))) for r in benchmark if 'arkit' in str(r.get('dataset_source','')).lower()}
    excluded_visits={str(r.get('metadata',{}).get('physical_scene_id','')).removeprefix('arkitscenes:visit:') for r in benchmark}
    config={'source_manifest_sha256':sha(args.source_manifest),'benchmark_sha256':sha(args.benchmark),'count':args.count,'max_per_physical_scene':args.max_per_physical_scene,'policy':POLICY}
    prior=OUT/'config.json'
    if prior.exists() and json.loads(prior.read_text())!=config:raise ValueError('Output configuration changed; use a new output directory')
    write(prior,config)
    rows=[];counts=collections.Counter();seen=set()
    for item in sources:
        if item.get('split')!='Training':raise ValueError('Only official Training scans are allowed')
        visit=str(item['visit_id']);scene_id=str(item['scene_id'])
        if not visit or visit=='None':raise ValueError('Missing physical visit identity')
        if scene_id in excluded or visit in excluded_visits:continue
        scene=Path(item['directory'])
        candidates,report=scan_scene(scene,{'video_id':scene_id,'visit_id':visit},return_all=True)
        write(OUT/'search'/scene_id/'screening.json',report)
        for candidate in candidates or []:
            if len(rows)>=args.count or counts[visit]>=args.max_per_physical_scene:break
            record,window,*_=candidate
            semantic=(visit,tuple(sorted(record['grounding'][key]['label'] for key in spatial.ROLE_KEYS)))
            if semantic in seen:continue
            folder=OUT/'candidates'/str(record['id'])
            if (folder/'qa.json').exists():row=json.loads((folder/'qa.json').read_text())
            elif folder.exists():raise RuntimeError('Partial output retained; inspect before resuming: '+str(folder))
            else:
                folder.parent.mkdir(exist_ok=True)
                try:row=make_pack(candidate,len(rows)+1,folder)
                except ValueError as e:
                    folder.mkdir(parents=True,exist_ok=True);write(folder/'rejected.json',{'reason':str(e),'partial_files_retained':True});continue
            rows.append(row);seen.add(semantic);counts[visit]+=1
        if len(rows)>=args.count:break
    final=OUT/'candidates.jsonl'
    if final.exists():
        if read_jsonl(final)!=rows:raise ValueError('Final output already exists with different content')
    else:write_jsonl(final,rows)
    write(OUT/'summary.json',{'candidate_count':len(rows),'requested':args.count,'accepted_count':0,'human_review_required':True,'per_physical_visit':dict(counts)})

if __name__=='__main__':main()
