"""Data-construction commands. Heavy rendering dependencies are loaded lazily."""
import argparse
import importlib
import sys

COMMANDS = {
    'annotation-routes': ('videovera.data.navigation.annotation_routes', 'Enumerate routes from official Training annotations'),
    'render-routes': ('videovera.data.navigation.render', 'Render continuous navmesh RGB videos and measured camera poses'),
    'landmarks': ('videovera.data.navigation.landmarks', 'Add visual landmarks using a local model endpoint'),
    'route-tasks': ('videovera.data.routes.entrypoint', 'Build Forward Route Planning, Route-Shape or Reverse Route Planning'),
    'displacement': ('videovera.data.displacement.generate', 'Build Egocentric Displacement candidates'),
    'object-direction-scannet': ('videovera.data.object_direction.build_scannet', 'Build ScanNet/ScanRefer Object Direction candidates'),
    'object-direction-arkit': ('videovera.data.object_direction.arkitscenes', 'Build depth-supported ARKitScenes Object Direction candidates'),
    'landmark-routes': ('videovera.data.landmark_routes.windows', 'Build continuous Landmark Route Planning candidates'),
    'temporal': ('videovera.data.temporal.generate', 'Build Event-Sequence Ordering or Adjacent Action candidates'),
    'package': ('videovera.packaging', 'Validate, deduplicate and export reference/policy records'),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', nargs='?', choices=COMMANDS)
        parser.print_help()
        print('\nCommands:')
        for name, (_, description) in COMMANDS.items():
            print(f'  {name:28s} {description}')
        return 0
    command, *rest = argv
    if command not in COMMANDS:
        raise SystemExit(f'Unknown command: {command}. Use videovera --help.')
    sys.argv = [f'videovera {command}', *rest]
    module = importlib.import_module(COMMANDS[command][0])
    result = module.main()
    return result if isinstance(result, int) else 0
