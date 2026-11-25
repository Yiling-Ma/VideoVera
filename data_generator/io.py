"""Durable local I/O. Successful outputs are never silently overwritten."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_jsonl(path, rows):
    """Create a new finalized artifact; refuse an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def preserve(path):
    """Archive an owned partial/output instead of deleting it, including on failure."""
    import uuid
    path = Path(path)
    if not path.exists():
        return None
    archive = path.parent / 'preserved' / uuid.uuid4().hex
    archive.mkdir(parents=True)
    destination = archive / path.name
    path.rename(destination)
    atomic_json(archive / 'receipt.json', dict(original_name=path.name,
                preserved_name=destination.name, data_deleted=False))
    return destination
