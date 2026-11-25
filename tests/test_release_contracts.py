"""Synthetic fixtures only; no training data, model calls or claimed human reviews."""
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from videovera.packaging import group, reviewed, convert_reference
from videovera.io import atomic_json
from videovera.data.routes.entrypoint import scoped
from videovera.data.routes import generate
from videovera.validation.reference import (RewardWeights, gold_completion, expected_trace, score_completion)
from videovera.validation.media import decoded
from tests.test_pose_actions import trajectory
from videovera.data.navigation.questions import build_record
from videovera.data.navigation.actions import extract_route


def test_physical_visit_registry_closes_rescans_and_requires_training():
    row = {'id':'one','task_type':'relative_direction','dataset_source':'arkitscenes','scene_id':'scan-a'}
    registry = {'arkitscenes':{'scan-a':{'physical_scene_id':'visit-1','split':'Training'},
                              'scan-b':{'physical_scene_id':'visit-1','split':'Training'}}}
    a = group(row, registry); row['scene_id'] = 'scan-b'
    assert group(row, registry) == a
    registry['arkitscenes']['scan-b']['split'] = 'Validation'
    with pytest.raises(ValueError, match='Non-Training'):
        group(row, registry)
    with pytest.raises(ValueError, match='Missing official'):
        group(row, {})


def test_review_requires_all_five_decisions():
    review = dict(decision='accept', video_quality='pass', question_unique='yes',
                  gold_answer_correct='yes', gold_reasoning_correct='yes')
    assert reviewed(review)
    for key in review:
        missing = dict(review); missing.pop(key)
        assert not reviewed(missing)


def test_scoping_does_not_require_shape_pool_to_select_reverse():
    pool = {}
    for label in generate.REVERSE:
        pool['reverse_route_plan',label] = [dict(parent=f'{label}_{i}',start=i,end=i+1,scene='s',task='reverse_route_plan',answer=label) for i in range(10)]
    selection = scoped(generate.select, 'reverse_route_plan')
    plan = selection(pool, 6, 20261003)
    assert len(plan) == 6
    assert {p['task'] for p in plan} == {'reverse_route_plan'}
    assert len({p['id'] for p in plan}) == 6


def test_reference_reward_and_reordered_claim_penalty(tmp_path):
    payload = trajectory([('go',3),('turn',90),('go',3)])
    make_video(tmp_path/'video.mp4')
    with patch('videovera.data.navigation.questions.extract_route',return_value=extract_route(payload)):
        raw = build_record(payload,tmp_path/'video.mp4',tmp_path/'poses.json','scene','route','reverse_route_plan')
    source = tmp_path/'source.json';source.write_text(json.dumps(raw))
    row = convert_reference(raw,source)
    gold = row['structured_gold']; parsed = json.loads(gold_completion(gold))
    weights = RewardWeights(answer=.40,process=.45,joint=.10)
    good = score_completion(json.dumps(parsed),gold,weights=weights)
    assert good.total == pytest.approx(1.) and good.trace_exact
    parsed['steps'][0],parsed['steps'][1] = parsed['steps'][1],parsed['steps'][0]
    shuffled = score_completion(json.dumps(parsed),gold,weights=weights)
    assert shuffled.process_f1 < 1 and not shuffled.trace_exact


def make_video(path,color='testsrc2'):
    source = f'{color}=size=160x120:rate=10' if color=='testsrc2' else f'color=c={color}:s=160x120:r=10'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',source,'-t','15',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)


def test_full_decode_detects_content_copies_and_corruption(tmp_path):
    video = tmp_path/'one.mp4';make_video(video)
    result = decoded(video)
    copyfile = tmp_path/'alias.mp4';copyfile.write_bytes(video.read_bytes())
    assert decoded(copyfile)['decoded_sha256'] == result['decoded_sha256']
    assert result['frames'] == 150 and result['sample_count'] == 24
    copyfile.write_bytes(b'not valid video')
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        decoded(copyfile)


def test_temporal_generation_and_packaging_preserve_warmup(tmp_path):
    videos = tmp_path/'videos';videos.mkdir();make_video(videos/'P01_test.MP4')
    fields=['video_id','narration','verb','verb_class','noun','noun_class','start_frame','stop_frame','start_timestamp','stop_timestamp']
    ann = tmp_path/'EPIC_100_train.csv'
    with ann.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,(verb,noun) in enumerate([('open','door'),('turn-on','light'),('close','door'),('open','fridge')]):
            start=1+i*3;end=start+2
            writer.writerow(dict(video_id='P01_test',narration=verb+' '+noun,verb=verb,verb_class=i,noun=noun,noun_class=i,
                                 start_frame=start*10,stop_frame=end*10,start_timestamp=f'00:00:{start:02d}.00',stop_timestamp=f'00:00:{end:02d}.00'))
    benchvideo=tmp_path/'benchmark.mp4';make_video(benchvideo,'yellow')
    bench=tmp_path/'benchmark.jsonl';bench.write_text(json.dumps(dict(id='test-benchmark',task_type='event_order',dataset_source='epic_kitchens_100',video_id='P02_test',scene_id='P02_test',video_path=str(benchvideo)))+'\n')
    out=tmp_path/'temporal'
    subprocess.run([sys.executable,'-m','videovera','temporal','--task','event_order','--annotations',str(ann),
                    '--video-root',str(videos),'--benchmark',str(bench),'--output-dir',str(out),'--count','1'],check=True)
    candidates=out/'candidates.jsonl';rows=[json.loads(x) for x in candidates.read_text().splitlines()]
    assert len(rows)==1
    assert Path(rows[0]['video_path']).is_relative_to(out)
    converted=convert_reference(rows[0],candidates)
    warm=tmp_path/'warmup.jsonl';warm.write_text(json.dumps(converted,ensure_ascii=False,indent=None)+'\n')
    registry=tmp_path/'registry.json';registry.write_text(json.dumps({'epic':{'P01_test':{'split':'train','physical_scene_id':'P01_test'},'P02_test':{'split':'train','physical_scene_id':'P02_test'}}}))
    reviews=tmp_path/'reviews.jsonl';reviews.write_text(json.dumps(dict(id=rows[0]['id'],decision='accept',video_quality='pass',question_unique='yes',gold_answer_correct='yes',gold_reasoning_correct='yes'))+'\n')
    export=tmp_path/'export'
    subprocess.run([sys.executable,'-m','videovera','package','--candidates',str(candidates),'--benchmark',str(bench),
                    '--source-registry',str(registry),'--review-labels',str(reviews),'--warmup',str(warm),'--output-dir',str(export)],check=True)
    assert (export/'warmup.jsonl').read_bytes()==warm.read_bytes()
    manifest=json.loads((export/'manifest.json').read_text())
    assert manifest['total']==1 and manifest['warmup']==1 and manifest['rl']==0

    # Renaming an already accepted semantic window must not create new data.
    alias=copy.deepcopy(rows[0]);alias['id']='renamed_candidate'
    aliases=tmp_path/'aliases.jsonl';aliases.write_text(json.dumps(alias)+'\n')
    reviews.write_text(json.dumps(dict(id=alias['id'],decision='accept',video_quality='pass',question_unique='yes',gold_answer_correct='yes',gold_reasoning_correct='yes'))+'\n')
    second=tmp_path/'second_export'
    subprocess.run([sys.executable,'-m','videovera','package','--candidates',str(aliases),'--benchmark',str(bench),
                    '--source-registry',str(registry),'--review-labels',str(reviews),'--warmup',str(warm),
                    '--existing',str(export/'corpus.jsonl'),'--output-dir',str(second)],check=True)
    report=json.loads((second/'manifest.json').read_text())
    assert report['accepted_new']==0 and report['rejected']==1
    assert 'Duplicate source trajectory' in json.loads((second/'rejections.json').read_text())[0]['reason']
    assert (second/'corpus.jsonl').read_bytes()==(export/'corpus.jsonl').read_bytes()


def test_archive_keeps_failed_media_and_original_bytes(tmp_path):
    from videovera.io import preserve
    path=tmp_path/'partial.mp4';path.write_bytes(b'preserved partial byte sequence')
    archived=preserve(path)
    assert not path.exists() and archived.read_bytes()==b'preserved partial byte sequence'
    assert json.loads((archived.parent/'receipt.json').read_text())['data_deleted'] is False


def test_follower_identity_and_reference_novelty_do_not_alter_geometry(tmp_path):
    from videovera.data.navigation.identity import adapt_episode, reference_key, unique_sources
    row=dict(episode_id='42',reference_path=[[1,2,3],[4,5,6]])
    original=copy.deepcopy(row);adapted=adapt_episode(row)
    assert row==original and adapted['reference_path']==original['reference_path']
    assert adapted['trajectory_id']=='follower_episode_42'
    a=tmp_path/'one.json';b=tmp_path/'reverse_alias.json'
    raw=dict(episode={'scene_id':'scene'},reference_path=[{'position_xyz_m':x} for x in row['reference_path']])
    a.write_text(json.dumps(raw));raw['reference_path'].reverse();b.write_text(json.dumps(raw))
    assert reference_key(a)==reference_key(b)
    assert unique_sources([a,b])==[a]
    assert unique_sources([b],[a])==[]


def test_source_protocol_requires_human_sample_decisions_and_hash():
    from videovera.packaging import approved_protocol
    review=dict(decision='accept',video_quality='pass',question_unique='yes',gold_answer_correct='yes',gold_reasoning_correct='yes')
    approval=dict(family='arkitscenes',task_type='relative_direction',protocol_sha256='policy-hash',
                  decision='approve_pipeline',allow_bulk_generation=True,allow_admission_after_checks=True,
                  reviewer='synthetic_test_only',statement='synthetic fixture; no real review claimed',reviewed_sample_ids=['sample'])
    assert approved_protocol(approval,'arkitscenes','relative_direction',{'sample':review},'policy-hash')
    assert not approved_protocol(approval,'arkitscenes','relative_direction',{},'policy-hash')
    assert not approved_protocol(approval,'arkitscenes','relative_direction',{'sample':review},'changed-policy')
    assert not approved_protocol(approval,'arkitscenes','route_plan',{'sample':review},'policy-hash')


def test_stationary_shape_semantics_keep_parent_window_identity():
    from videovera.packaging import semantic
    row=dict(id='stationary',task_type='route_shape',question='What is the route shape?',
             grounding={'pose_route':None},metadata={'parent_video_sha256':'bound-parent','parent_frame_range':[20,50]})
    before=copy.deepcopy(row)
    a=semantic(row,('r2r','scene'))
    row['id']='renamed'
    assert semantic(row,('r2r','scene'))==a
    row['metadata']['parent_frame_range']=[60,90]
    assert semantic(row,('r2r','scene'))!=a
    assert before['grounding']==row['grounding']
