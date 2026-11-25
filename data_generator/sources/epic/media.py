from __future__ import annotations
import subprocess
from pathlib import Path
from .annotations import ActionSegment

def extract_clip(*, video_path: Path, segment: ActionSegment, out_path: Path, pad_before: float, pad_after: float, duration_limit: float | None=None) -> dict[str, float]:
    start_sec = max(0.0, segment.start_sec - pad_before)
    end_sec = segment.stop_sec + pad_after
    if duration_limit is not None and end_sec - start_sec > duration_limit:
        end_sec = start_sec + duration_limit
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return {'clip_start_sec': start_sec, 'clip_end_sec': end_sec}
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{start_sec:.3f}', '-to', f'{end_sec:.3f}', '-i', str(video_path), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(out_path)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return {'clip_start_sec': start_sec, 'clip_end_sec': end_sec}

def extract_clip_range(*, video_path: Path, start_sec: float, end_sec: float, out_path: Path) -> dict[str, float]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return {'clip_start_sec': start_sec, 'clip_end_sec': end_sec}
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{max(0.0, start_sec):.3f}', '-to', f'{max(0.0, end_sec):.3f}', '-i', str(video_path), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(out_path)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return {'clip_start_sec': start_sec, 'clip_end_sec': end_sec}

def extract_support_frames(*, video_path: Path, segment: ActionSegment, out_dir: Path, num_frames: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [out_dir / f'frame_{idx + 1:02d}.jpg' for idx in range(num_frames)]
    if all((p.exists() and p.stat().st_size > 0 for p in existing)):
        return [str(p) for p in existing]
    duration = max(0.05, segment.stop_sec - segment.start_sec)
    frame_paths: list[str] = []
    for idx in range(num_frames):
        t = segment.start_sec + duration * ((idx + 0.5) / num_frames)
        out_path = out_dir / f'frame_{idx + 1:02d}.jpg'
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{t:.3f}', '-i', str(video_path), '-frames:v', '1', '-vf', 'scale=640:-1', '-q:v', '4', str(out_path)]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        frame_paths.append(str(out_path))
    return frame_paths
