# VideoVera

**Verifiable Process Reward for Video Reasoning**

This repository contains the data-construction pipeline for VideoVera. Existing
video annotations determine questions, answer choices, correct answers and
reference reasoning claims. The generators retain continuous source video and
record the geometry, poses or temporal segments used to derive each example.

This release focuses on data generation and acceptance. It does not include
training runs, checkpoints, experimental results, benchmark result analysis,
error analysis, paper figures or cluster-specific job controllers.

## Tasks and annotation sources

| Task | Command | Source annotations |
|---|---|---|
| Object Direction | `object-direction-scannet`, `object-direction-arkit` | ScanNet/ScanRefer objects and instance masks; ARKitScenes object boxes, RGB-D and poses |
| Egocentric Displacement | `displacement` | R2R/RxR routes, continuous Matterport3D rendering and measured camera poses |
| Forward Route Planning | `route-tasks --task route_plan` | Measured forward actions and visually grounded route landmarks |
| Landmark Route Planning | `landmark-routes` | ARKitScenes object boxes, depth support, camera trajectory and continuous RGB |
| Route-Shape Classification | `route-tasks --task route_shape` | Complete measured motion/turn sequence |
| Reverse Route Planning | `route-tasks --task reverse_route_plan` | Forward route, inverse actions and reversed temporal order |
| Event-Sequence Ordering | `temporal --task event_order` | EPIC-KITCHENS-100 Training action segments |
| Adjacent Action | `temporal --task before_or_after_action` | EPIC-KITCHENS-100 Training action segments and their neighbors |

Public module and command names describe their purpose rather than historical
experiment versions. Stable task keys and serialized schema identifiers are
retained so existing dataset records remain readable.

## Install

Python 3.10 or later, FFmpeg and FFprobe are required.

```bash
git clone https://github.com/Yiling-Ma/VideoVera.git
cd VideoVera
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[geometry,dev]'
videovera --help
```

Continuous Matterport3D rendering additionally requires a compatible
Habitat-Sim installation and locally licensed scene assets. Install the
`navigation` extra in that environment. Habitat-Sim is a separate, platform-
dependent installation; it is not replaced by a generic panorama renderer.

```bash
python -m pip install -e '.[navigation]'
```

Source datasets, their access credentials and model weights are **not** included.
Acquire each source through its official distribution under its own terms.
See [source preparation](docs/sources.md) and the complete
[generation recipe](docs/generation.md).

## Pipeline

1. Prepare official Training annotations and continuous source media.
2. Generate task-specific candidate questions and annotation-derived references.
3. Retain media, source provenance, completed checkpoints and rejection records.
4. Review candidates or approve a source/task protocol after concrete sample review,
   retaining explicit human decisions and protocol hashes.
5. Run `videovera package` against the fixed benchmark and prior corpus to check
   source isolation, decoded content, semantics, scene caps and reference targets.
6. Export separate policy inputs and hidden reference targets, preserving a
   supplied warm-up JSONL byte-for-byte.

Candidates are not automatically accepted training data. If valid source
capacity is insufficient, the generators report a shortfall rather than weaken
quality gates or duplicate examples. New output directories are used for new
runs; completed outputs and failure records are preserved.

## Layout

```text
src/videovera/
  data/
    navigation/        # Annotation routes, rendering, pose actions and landmarks
    routes/            # Forward, shape and reverse task generation
    displacement/      # Endpoint-frame geometry and balanced route selection
    object_direction/  # ScanNet and ARKitScenes spatial candidates
    landmark_routes/   # Depth-supported objects and continuous route windows
    temporal/          # Event ordering and adjacent-action candidates
  sources/             # Official annotation readers and geometry utilities
  formats/             # Messages, options and typed reference construction
  validation/          # Full decode, schema and reference agreement checks
  packaging.py         # Acceptance, isolation, deduplication and split export
configs/               # Source/review templates and quality policy
docs/                  # Installation, generation and export contracts
tests/                 # Synthetic geometry/media and regression tests
provenance/            # Production-component hashes
```

## Quality and split contracts

- Object Direction uses explicit observer/facing/target roles, signed horizontal
  projections, referent uniqueness, target-extent margins and visual support.
- Route labels come from measured camera poses. Initial or terminal looks are
  distinguished from actual path bends. Shape labels and reverse actions are
  computed without relabeling clips to satisfy quotas.
- Landmark routes retain one continuous original RGB interval, calibrated poses,
  depth-supported visibility and uniform model-frame support. Dense window
  search expands coverage without changing candidate quality tests.
- Temporal examples use distinct annotated actions, unambiguous time order and
  at least two uniform model samples per required action.
- Packaging checks identifiers, physical source groups, full media decoding,
  uniform decoded-frame content, QA slots, source semantics and reference validity.
  Physical scene IDs are required; rescans do not create additional scene capacity.
- Object Direction is capped at 10 examples per physical scene; Landmark Route
  Planning at 14. Shape/reverse selection retains at most five clips per parent.
- Frozen warm-up membership and bytes are preserved. Policy exports contain no
  answers, reference traces, verifier state or assistant messages.

See [acceptance and formats](docs/acceptance.md) for exact inputs and outputs.
No model training or recurring background process is launched by these commands.

## Checks

```bash
python -m pytest -q
python -m ruff check src tests
```

The test fixtures are synthetic. They check geometry, inverse actions, shape
boundaries, class/letter balancing, immutable publication, physical-source
identity, reference scoring, corruption detection, and temporal generation
through packaging. Real-data generation additionally requires the licensed
source assets; GPU rendering must be checked in a Habitat-Sim environment.

## Source acknowledgements

The pipeline reads annotations from ScanNet, ScanRefer, ARKitScenes,
Matterport3D, R2R, RxR and EPIC-KITCHENS-100 and uses Habitat-Sim for continuous
navigation rendering. These datasets and dependencies retain their own licenses
and citation requirements. Dataset assets and third-party model weights are not
redistributed by this code repository.
