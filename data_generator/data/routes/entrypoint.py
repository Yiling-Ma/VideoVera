"""Generate task-scoped route candidates without changing selection or media gates."""
import argparse
import ast
import inspect
import json
import sys
from pathlib import Path
from collections import Counter
from data_generator.io import atomic_json, read_jsonl, write_jsonl, sha256
from data_generator.data.routes import generate, finalize
from data_generator.data.navigation.questions import build_record


def scoped(function, task):
    """Filter only scheduling loops; preserve the original task validators."""
    class Scope(ast.NodeTransformer):
        def visit_For(self, node):
            self.generic_visit(node)
            if not isinstance(node.iter, (ast.Tuple, ast.List)):
                return node
            target = ast.unparse(node.target)
            if target == 'task' and ast.literal_eval(node.iter) in [
                    ('route_shape', 'reverse_route_plan')]:
                node.iter = ast.Tuple(elts=[ast.Constant(task)], ctx=ast.Load())
            elif target == '(task, labels)' or target == 'task, labels':
                items = node.iter.elts
                if len(items) == 2 and all(isinstance(x, ast.Tuple) for x in items):
                    if [x.elts[0].value for x in items] == ['route_shape', 'reverse_route_plan']:
                        node.iter.elts = [x for x in items if x.elts[0].value == task]
            return node
    tree = Scope().visit(ast.parse(inspect.getsource(function)))
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(function), 'exec'), namespace)
    return namespace[function.__name__]


def forward(source, output, count):
    rows = []; failed = []; seen = set()
    for item in json.loads((source / 'manifest.json').read_text())['videos']:
        video = source / item['file']; annotation = source / item['action_file']
        payload = json.loads(annotation.read_text())
        try:
            row = build_record(payload, video, annotation, item.get('scene_id', payload.get('scene_id', '')),
                               video.stem, 'route_plan')
            key = row['metadata']['pose_signature']
            if key in seen:
                continue
            seen.add(key); rows.append(row)
        except ValueError as exc:
            failed.append({'video': str(video), 'reason': str(exc)})
        if len(rows) >= count:
            break
    atomic_json(output / 'rejections.json', failed)
    if len(rows) < count:
        write_jsonl(output / 'candidates.partial.jsonl', rows)
        raise RuntimeError(f'Insufficient valid route parents: {len(rows)}/{count}; partial outputs retained')
    write_jsonl(output / 'route_plan.jsonl', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--task', required=True, choices=['route_plan', 'route_shape', 'reverse_route_plan'])
    if any(x in sys.argv for x in ('-h','--help')) and '--task' not in sys.argv:
        parser.print_help()
        sys.argv=[sys.argv[0],'--help']
        return generate.main()
    args, rest = parser.parse_known_args()
    if '-h' in rest or '--help' in rest:
        parser.print_help()
        sys.argv = [sys.argv[0], '--help']
        return generate.main()
    if '--output' not in rest or '--source' not in rest:
        raise SystemExit('--source and --output are required')
    output = Path(rest[rest.index('--output') + 1]); output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(rest[rest.index('--source') + 1])
    scope = {'task': args.task, 'source_manifest_sha256': sha256(source / 'manifest.json')}
    receipt = output / 'task_scope.json'
    if receipt.exists() and json.loads(receipt.read_text()) != scope:
        raise ValueError('Task or source changed; use a new output directory')
    atomic_json(receipt, scope)
    if args.task == 'route_plan':
        count = int(rest[rest.index('--count-per-task') + 1]) if '--count-per-task' in rest else 1000
        return forward(source, output, count)
    generate.select = scoped(generate.select, args.task)
    finalize.finalize = scoped(finalize.finalize, args.task)
    sys.argv = [sys.argv[0], *rest]
    return generate.main()

if __name__ == '__main__':
    main()
