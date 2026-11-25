"""Final data-construction component: data/object_direction/descriptions.

Task rules are ported from the production generator. Machine paths and imports
are normalized for this package; use the documented public CLI."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
ATTRIBUTES = set('white black gray grey brown beige red blue green yellow orange pink purple silver golden wooden wood metal metallic plastic leather'.split())
COLORS = set('white black gray grey brown beige red blue green yellow orange pink purple silver golden'.split())
EXCLUDED = {'wall', 'floor', 'ceiling', 'object', 'doorframe', 'windowsill', 'blanket', 'clothes', 'clothing'}
GROUPED_LABELS = {'cabinets', 'kitchen cabinets', 'bathroom cabinets', 'cupboards', 'shelves', 'bookshelves', 'drawers', 'chairs', 'tables', 'desks', 'beds', 'sofas', 'couches', 'stools', 'pillows', 'windows', 'doors', 'curtains', 'blinds'}
ALIASES = {'kitchen cabinet': 'cabinet', 'kitchen cabinets': 'cabinet', 'bathroom cabinet': 'cabinet', 'bathroom cabinets': 'cabinet', 'cupboard': 'cabinet', 'cupboards': 'cabinet', 'kitchen counter': 'counter', 'kitchen counters': 'counter', 'bookshelves': 'bookshelf', 'shelves': 'shelf'}

def canonical_label(label):
    label = re.sub('\\s+', ' ', label.replace('_', ' ').strip().lower())
    return ALIASES.get(label, label)

def stable_reference(label, phrases):
    """Return a conservative visual name, or None for conflicting colors."""
    if not phrases:
        return 'the ' + label
    colors = {word for phrase in phrases for word in phrase.split() if word in COLORS}
    colors.discard('grey') if 'gray' in colors else None
    colors.discard('gray') if 'grey' in colors else None
    if len(colors) > 1:
        return None
    return 'the ' + sorted(phrases, key=lambda value: (-len(value.split()), value))[0]

def intrinsic_phrases(description, label):
    words = '|'.join(sorted(ATTRIBUTES))
    pattern = f'\\b((?:(?:{words})\\s+){{1,3}}{re.escape(label)})\\b'
    return sorted(set(re.findall(pattern, description.lower())))

def audit(rows, aggregation):
    groups = aggregation['segGroups']
    by_id = {str(g['objectId']): g for g in groups}
    counts = Counter((canonical_label(g['label']) for g in groups))
    matched = defaultdict(list)
    mismatches = []
    for row in rows:
        group = by_id.get(str(row['object_id']))
        if group is None or row['object_name'].replace('_', ' ').lower() != group['label'].lower():
            mismatches.append({'object_id': row['object_id'], 'ann_id': row['ann_id']})
            continue
        matched[str(row['object_id'])].append(row)
    candidates = []
    for oid, descriptions in sorted(matched.items()):
        label = by_id[oid]['label'].strip().lower()
        canonical = canonical_label(label)
        if counts[canonical] != 1 or label in EXCLUDED or label in GROUPED_LABELS:
            continue
        phrases = defaultdict(list)
        for row in descriptions:
            for phrase in intrinsic_phrases(row['description'], label):
                phrases[phrase].append(row['ann_id'])
        reference = stable_reference(label, phrases)
        if reference is None:
            continue
        candidates.append({'object_id': oid, 'label': label, 'canonical_label': canonical, 'scene_category_count': counts[canonical], 'default_reference': reference, 'appearance_candidates': [{'text': 'the ' + phrase, 'source_ann_ids': ids} for phrase, ids in sorted(phrases.items())], 'source_descriptions': [{'ann_id': r['ann_id'], 'description': r['description']} for r in descriptions], 'human_review_required': True, 'video_visibility_verified': False, 'appearance_verified': False})
    return {'category_counts': dict(counts), 'id_or_label_mismatches': mismatches, 'unique_category_candidates': candidates, 'training_ready': False, 'note': 'Scene annotation uniqueness only; video visibility and appearance remain unverified.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene', default='scene0000_00')
    args = parser.parse_args()
    split_stats = {}
    training = []
    for split in ('train', 'val'):
        rows = json.loads((args.root / 'scanrefer' / f'ScanRefer_filtered_{split}.json').read_text())
        split_stats[split] = dict(descriptions=len(rows), scenes=len({r['scene_id'] for r in rows}), objects=len({(r['scene_id'], r['object_id']) for r in rows}))
        if split == 'train':
            training = rows
    rows = [r for r in training if r['scene_id'] == args.scene]
    if not rows:
        raise ValueError('Pilot scene must be present in the ScanRefer training split')
    aggregation = json.loads((args.root / 'scans' / args.scene / f'{args.scene}.aggregation.json').read_text())
    report = dict(scene_id=args.scene, split='train', split_stats=split_stats, **audit(rows, aggregation))
    out = args.root / 'reports'
    out.mkdir(exist_ok=True)
    (out / f'{args.scene}_description_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(split_stats=split_stats, scene=args.scene, matched_descriptions=len(rows) - len(report['id_or_label_mismatches']), mismatches=len(report['id_or_label_mismatches']), unique_candidates=len(report['unique_category_candidates']), appearance_candidates=sum((bool(r['appearance_candidates']) for r in report['unique_category_candidates']))), indent=2))
if __name__ == '__main__':
    main()
