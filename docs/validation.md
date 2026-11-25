# Release checks

The portable release was checked with Python 3.10 and FFmpeg on a CPU environment.
The regression suite passes 45 tests and three parametrized subtests. All ten
public command help entrypoints load successfully, package imports resolve, and
undefined-name/static syntax checks pass.

The tests include actual encoding and decoding of small synthetic MP4 fixtures,
not model inference or real source-data generation. They cover pose/video hash
binding, forward/inverse actions, terminal-look handling, stationary turns,
shape boundaries, label/letter balance, single-task selection, immutable
publication, physical rescans, Training membership, source-protocol approval,
ordered reference agreement, invalid media, temporal candidate generation,
policy/reference export, decoded-QA deduplication and frozen warm-up bytes.
Failed media preservation and follower reference identity are also checked.

Habitat-Sim GPU rendering, calibrated source-data screening at corpus scale and
the external local landmark-model endpoint require their original runtime and
licensed inputs. They were not rerun as part of code organization. No checkpoint,
training job, source-data downloader or recurring monitor was started for this
release.
