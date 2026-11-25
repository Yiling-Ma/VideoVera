# Installation checks

The Python package is `data_generator`; the installed command is `videovera`.
After installation, check both entrypoints and compile the Python modules:

```bash
videovera --help
python -m data_generator --help
python -m compileall -q data_generator
```

Inspect task-specific options before running a generator:

```bash
videovera route-tasks --task route_shape --help
videovera temporal --help
videovera package --help
```

These commands check installation and expose the available arguments. Actual
generation additionally requires the relevant official source assets. Continuous
navigation rendering requires Habitat-Sim and the navigation dependencies;
landmark observation uses a separately configured local model endpoint.

See [source preparation](sources.md) for runtime inputs and
[review and export](acceptance.md) for candidate acceptance and output formats.
