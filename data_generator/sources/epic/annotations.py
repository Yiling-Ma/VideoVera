from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ActionSegment:
    uid: str
    video_id: str
    participant_id: str
    narration: str
    verb: str
    verb_class: int
    noun: str
    noun_class: int
    start_frame: int
    stop_frame: int
    start_timestamp: str
    stop_timestamp: str
    start_sec: float
    stop_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.stop_sec - self.start_sec)

@dataclass
class VideoInfo:
    video_id: str
    duration_sec: float
    fps: float
    resolution: str

def load_segments(csv_path: Path, video_id: str) -> list[ActionSegment]:
    rows: list[ActionSegment] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if row['video_id'] != video_id:
                continue
            rows.append(ActionSegment(uid=f'{video_id}_{idx:05d}', video_id=video_id, participant_id=video_id.split('_')[0], narration=row['narration'].strip(), verb=row['verb'].strip(), verb_class=int(row['verb_class']), noun=row['noun'].strip(), noun_class=int(row['noun_class']), start_frame=int(row['start_frame']), stop_frame=int(row['stop_frame']), start_timestamp=row['start_timestamp'].strip(), stop_timestamp=row['stop_timestamp'].strip(), start_sec=timestamp_to_seconds(row['start_timestamp']), stop_sec=timestamp_to_seconds(row['stop_timestamp'])))
    rows.sort(key=lambda seg: seg.start_sec)
    return rows

def load_video_info(csv_path: Path, video_id: str) -> VideoInfo:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id:
                return VideoInfo(video_id=video_id, duration_sec=float(row['duration']), fps=float(row['fps']), resolution=row['resolution'].strip())
    raise KeyError(f'video_id not found in video info: {video_id}')

def timestamp_to_seconds(value: str) -> float:
    hh, mm, ss = value.strip().split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)
