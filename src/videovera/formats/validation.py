"""Final data-construction component: formats/validation.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from videovera.formats.options import option_block, ANSWER_REFERENCE_RE
EGO = 'egocentric_temporal_direction'
ARKIT = 'object_grounded_route_planning'
ANSWER_REF = re.compile('\\b(?:the (?:correct )?answer is|the correct option is|select option)\\s+([A-D])\\b', re.I)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_messages(rows, expected_counts):
    require(Counter((r['task_type'] for r in rows)) == expected_counts, 'Unexpected task counts')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate IDs in merged corpus')
    for row in rows:
        require(Path(row['video_path']).is_file(), 'Missing video: ' + row['video_path'])
        labels, values, _, _ = option_block(row)
        require(dict(zip(labels, values)).get(row['answer_key']) == row['answer'], 'Option mapping mismatch: ' + row['id'])
        require([m['role'] for m in row['messages']] == ['user', 'assistant'], 'Wrong chat roles')
        videos = [v for v in row['messages'][0]['content'] if v['type'] == 'video']
        require(len(videos) == 1 and videos[0]['video'] == row['video_path'], 'Chat video path mismatch')
        text = row['messages'][1]['content'][0]['text']
        finals = re.findall('(?m)^Answer:\\s*([A-Z])\\s*$', text)
        require(finals == [row['answer_key']] and text.rstrip().endswith('Answer: ' + row['answer_key']), 'Assistant final answer mismatch: ' + row['id'])
        require(all((m['letter'] == row['answer_key'] for m in ANSWER_REFERENCE_RE.finditer(text))), 'Stale reasoning answer letter: ' + row['id'])
        if row['task_type'] in (EGO, ARKIT):
            references = re.findall('\\bchoice\\s+([A-D])\\b', text, re.I)
            require(all((letter.upper() == row['answer_key'] for letter in references)), 'Stale reasoning answer letter: ' + row['id'])
            require(not ANSWER_REF.search(text), 'Unnormalized reasoning answer letter: ' + row['id'])
