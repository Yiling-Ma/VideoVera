import hashlib
import json
from pathlib import Path
import sys

import pytest

from videovera.data.routes.finalize import finalize, publish


def test_published_snapshots_are_immutable_and_idempotent(tmp_path):
    path = tmp_path/'data.jsonl'
    publish(path, [{'id':'one'}])
    before = path.read_bytes()
    publish(path, [{'id':'one'}])
    with pytest.raises(FileExistsError):
        publish(path, [{'id':'two'}])
    assert path.read_bytes() == before


def test_identical_video_with_conflicting_labels_is_never_backfilled(tmp_path):
    (tmp_path/'completed').mkdir()
    video = tmp_path/'fixture.mp4'
    video.write_bytes(b'synthetic hash fixture, not real media')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    (tmp_path/'plan.json').write_text(json.dumps([{'id':'a'},{'id':'b'}]))
    for key, answer in [('a','L-shape'),('b','U-shape')]:
        row = dict(id=key, task_type='route_shape', video_path=str(video),
                   answer=answer, answer_key='A', metadata={'video_sha256':digest})
        (tmp_path/'completed'/(key+'.json')).write_text(json.dumps(row))
    with pytest.raises(ValueError, match='conflicting gold'):
        finalize(tmp_path/'unused_source', tmp_path, 42, 2)
    assert not (tmp_path/'route_shape.jsonl').exists()
