from __future__ import annotations
from pathlib import Path
from .annotations import load_segments, load_video_info
DEFAULT_EPIC_ROOT = Path('inputs/EPIC_KITCHENS')

def resolve_epic_paths(data_root: Path) -> tuple[Path, Path, Path]:
    """Return (annotations_csv, video_info_csv, videos_root)."""
    annotations_dir = data_root / 'annotations'
    train_csv = annotations_dir / 'EPIC_100_train.csv'
    info_csv = annotations_dir / 'EPIC_100_video_info.csv'
    videos_root = data_root / 'EPIC-KITCHENS'
    if not train_csv.exists():
        raise FileNotFoundError(f'Missing annotations CSV: {train_csv}')
    if not info_csv.exists():
        raise FileNotFoundError(f'Missing video info CSV: {info_csv}')
    if not videos_root.exists():
        raise FileNotFoundError(f'Missing videos root: {videos_root}')
    return (train_csv, info_csv, videos_root)

def find_video_path(videos_root: Path, video_id: str) -> Path | None:
    participant_id = video_id.split('_')[0]
    for ext in ('.MP4', '.mp4', '.mov', '.mkv'):
        candidate = videos_root / participant_id / 'videos' / f'{video_id}{ext}'
        if candidate.exists():
            return candidate
    matches = list(videos_root.glob(f'*/videos/{video_id}.*'))
    return matches[0] if matches else None

def list_available_videos(data_root: Path, *, max_videos: int | None=None, video_ids: list[str] | None=None) -> list[tuple[str, Path]]:
    train_csv, _info_csv, videos_root = resolve_epic_paths(data_root)
    if video_ids:
        selected = video_ids
    else:
        selected = sorted({p.stem for p in videos_root.glob('*/videos/*.*')})
    out: list[tuple[str, Path]] = []
    for video_id in selected:
        video_path = find_video_path(videos_root, video_id)
        if video_path is None:
            continue
        segments = load_segments(train_csv, video_id)
        if not segments:
            continue
        out.append((video_id, video_path))
        if max_videos is not None and len(out) >= max_videos:
            break
    return out

def load_video_bundle(data_root: Path, video_id: str) -> tuple[Path, list, object]:
    train_csv, info_csv, videos_root = resolve_epic_paths(data_root)
    video_path = find_video_path(videos_root, video_id)
    if video_path is None:
        raise FileNotFoundError(f'No local video file for {video_id} under {videos_root}')
    segments = load_segments(train_csv, video_id)
    if not segments:
        raise RuntimeError(f'No segments found for {video_id} in {train_csv}')
    video_info = load_video_info(info_csv, video_id)
    return (video_path, segments, video_info)
