"""Data-construction commands. Heavy rendering dependencies are loaded lazily."""
import argparse
import importlib
import sys

COMMANDS = {
    'annotation-routes': ('data_generator.data.navigation.annotation_routes', 'Enumerate routes from official Training annotations'),
    'render-routes': ('data_generator.data.navigation.render', 'Render continuous navmesh RGB videos and measured camera poses'),
    'landmarks': ('data_generator.data.navigation.landmarks', 'Add visual landmarks using a local model endpoint'),
    'route-tasks': ('data_generator.data.routes.entrypoint', 'Build Forward Route Planning, Route-Shape or Reverse Route Planning'),
    'displacement': ('data_generator.data.displacement.generate', 'Build Egocentric Displacement candidates'),
    'object-direction-scannet': ('data_generator.data.object_direction.build_scannet', 'Build ScanNet/ScanRefer Object Direction candidates'),
    'object-direction-arkit': ('data_generator.data.object_direction.arkitscenes', 'Build depth-supported ARKitScenes Object Direction candidates'),
    'landmark-routes': ('data_generator.data.landmark_routes.windows', 'Build continuous Landmark Route Planning candidates'),
    'temporal': ('data_generator.data.temporal.generate', 'Build Event-Sequence Ordering or Adjacent Action candidates'),
    'package': ('data_generator.packaging', 'Validate, deduplicate and export reference/policy records'),
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
