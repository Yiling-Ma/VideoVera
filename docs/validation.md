# Installation and code checks

The public source package is `data_generator`, with `videovera` retained as the
installed command. GitHub Actions installs the package on Python 3.10 / Ubuntu,
checks undefined-name errors, compiles the Python modules and exercises both
command entrypoints.

Regression fixtures and internal release-review records are retained privately
rather than shipped in this repository. The private checks cover geometry,
forward/inverse actions, shape boundaries, class/letter balance, physical-source
identity, source-protocol approval, reference validation, media corruption,
temporal candidate generation, deduplication and frozen warm-up bytes.

Habitat-Sim GPU rendering, calibrated source-data screening at corpus scale and
the local landmark-model endpoint require their original runtime and licensed
inputs. They were not rerun during the directory reorganization. Installation
checks do not establish complete real-data or GPU reproduction.
