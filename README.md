<h1 align="center">
  <img src="assets/videovera-wordmark.svg" alt="VideoVera:" width="168" height="40" align="top"> Verifiable Process Reward<br>
  for Video Reasoning
</h1>

<p align="center">
  <a href="https://yiling-ma.github.io/VideoVera/">
    <img src="https://img.shields.io/badge/Project-Website-75539B?style=flat-square" alt="VideoVera project website">
  </a>
  <a href="https://github.com/Yiling-Ma/VideoVera">
    <img src="https://img.shields.io/badge/Code-GitHub-53658A?style=flat-square&amp;logo=github&amp;logoColor=white" alt="Code: GitHub">
  </a>
  <a href="https://huggingface.co/datasets/YilingMa/VideoVera-6.4k">
    <img src="https://img.shields.io/badge/Dataset-6.4k-A599B5?style=flat-square&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face dataset: VideoVera-6.4k">
  </a>
  <a href="https://huggingface.co/datasets/YilingMa/VideoVera-12k">
    <img src="https://img.shields.io/badge/Dataset-12k-8FA69B?style=flat-square&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face dataset: VideoVera-12k">
  </a>
  <a href="https://huggingface.co/datasets/YilingMa/VideoVera-27k">
    <img src="https://img.shields.io/badge/Dataset-27k-C0A587?style=flat-square&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face dataset: VideoVera-27k">
  </a>
</p>

<p align="center">
  <a href="assets/eight-tasks.png">
    <img src="assets/eight-tasks.png" width="100%" alt="VideoVera's eight reasoning tasks: Object Direction, Egocentric Displacement, Forward Route Planning, Landmark Route Planning, Route-Shape Classification, Reverse Route Planning, Event-Sequence Ordering, and Adjacent Action.">
  </a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> &nbsp;&middot;&nbsp;
  <a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
  <a href="#data">Data</a> &nbsp;&middot;&nbsp;
  <a href="#data-generation">Data Generation</a> &nbsp;&middot;&nbsp;
  <a href="#documentation">Documentation</a>
</p>

## Introduction

VideoVera constructs video-reasoning examples with verifiable process
supervision. Each example pairs a continuous video and a multiple-choice
question with a correct option and ordered reference claims, derived from
geometry, measured camera poses or annotated action intervals.

This repository provides the final data-construction pipeline for eight spatial,
route and temporal tasks. It includes source readers, task generators, review
templates and export tools for SFT and RL data, with policy inputs kept separate
from reference targets.

<a id="tasks-and-annotation-sources"></a>

### Eight reasoning tasks

| Task | Annotation sources |
|---|---|
| Object Direction | ScanNet/ScanRefer objects and instance masks; ARKitScenes object boxes, RGB-D and poses |
| Egocentric Displacement | R2R/RxR routes, continuous Matterport3D rendering and measured camera poses |
| Forward Route Planning | Measured forward actions and visually grounded route landmarks |
| Landmark Route Planning | ARKitScenes object boxes, depth support, camera trajectory and continuous RGB |
| Route-Shape Classification | Complete measured motion/turn sequence |
| Reverse Route Planning | Forward route, inverse actions and reversed temporal order |
| Event-Sequence Ordering | EPIC-KITCHENS-100 Training action segments |
| Adjacent Action | EPIC-KITCHENS-100 Training action segments and their neighbors |

## Highlights

- **Eight reasoning tasks.** Spatial direction and displacement, forward and
  landmark-based routes, trajectory shape, reverse routes and temporal actions.
- **Annotation-derived reference traces.** Questions, answers and ordered claims
  retain the geometry, measured motion or action timings used to construct them.
- **Three frozen training scales.** 6,400, 12,431 and 27,229 examples, with explicit
  warm-up and RL partitions. The two larger scales share the same warm-up data.
- **Continuous video and portable exports.** Releases include video payloads,
  SFT records, answer-free RL inputs, separate reference targets and checksums.

<a id="layout"></a>

## Repository Structure

```text
VideoVera/
├── assets/                    # README figures and editable SVGs
├── configs/                   # Source/review templates and quality policy
├── data_generator/
│   ├── data/                  # Generators for the eight tasks
│   │   ├── navigation/        # Route annotations, continuous rendering and poses
│   │   ├── routes/            # Forward, shape and reverse route tasks
│   │   ├── displacement/      # Egocentric displacement
│   │   ├── object_direction/  # ScanNet and ARKitScenes Object Direction
│   │   ├── landmark_routes/   # Continuous landmark-defined route windows
│   │   └── temporal/          # Event ordering and adjacent actions
│   ├── sources/               # Official annotation readers and geometry utilities
│   ├── formats/               # Chat records and typed reference construction
│   ├── validation/            # Media, schema and reference validation
│   └── packaging.py           # Acceptance, deduplication and partition export
├── docs/                      # Source preparation, generation and export guides
└── pyproject.toml             # Package configuration and dependencies
```

<a id="install"></a>

## Installation

Python 3.10 or later, FFmpeg and FFprobe are required.

```bash
git clone https://github.com/Yiling-Ma/VideoVera.git
cd VideoVera
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[geometry]'
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

<a id="training-datasets"></a>

## Data

The frozen training releases are available on Hugging Face:

| Release | Total examples | Warm-up SFT | RL prompts |
|---|---:|---:|---:|
| [VideoVera-6.4k](https://huggingface.co/datasets/YilingMa/VideoVera-6.4k) | 6,400 | 800 | 5,600 |
| [VideoVera-12k](https://huggingface.co/datasets/YilingMa/VideoVera-12k) | 12,431 | 1,600 | 10,831 |
| [VideoVera-27k](https://huggingface.co/datasets/YilingMa/VideoVera-27k) | 27,229 | 1,600 | 25,629 |

Each release includes video payloads, frozen training partitions, separate RL
inputs and reference targets, checksums and a download/preparation guide. The
12k and 27k releases share the same 1,600-example warm-up file byte-for-byte and
unchanged warm-up videos. The 6.4k release retains its original 800-example
warm-up partition. See each dataset card for task counts and exact file formats.

### Download a training release

With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
installed, choose one scale. For example, download and prepare the 27k release:

```bash
hf download YilingMa/VideoVera-27k \
  --repo-type dataset \
  --local-dir work/downloads/videovera-27k

python work/downloads/videovera-27k/tools/prepare_data.py \
  --root work/downloads/videovera-27k \
  --output work/prepared/videovera-27k
```

Use `YilingMa/VideoVera-6.4k` or `YilingMa/VideoVera-12k` for the other scales,
with corresponding local directory names. The preparation helper verifies
checksums, materializes the original videos and resolves local video paths.
Use a new preparation output directory.

The prepared `all.jsonl` is the full SFT corpus; `sft_warmup.jsonl` is the frozen
warm-up split. Use `grpo_policy.jsonl` for RL model inputs and keep
`grpo_targets.jsonl` separate from the model context. Each dataset card describes
its files and preparation steps in detail.

<a id="how-to-use-data_generator"></a>

## Data Generation

Run the commands below from the repository root after installation. The examples
use `python -m data_generator`; the installed `videovera` command accepts the same
subcommands and arguments. Run only the branches required for your tasks.

<p align="center">
  <a href="assets/data-construction-pipeline.svg">
    <img src="assets/data-construction-pipeline.png" width="100%" alt="VideoVera data construction: three annotation sources generate eight tasks, followed by human review, validation and deduplication, and export of the frozen training partitions.">
  </a>
</p>

### 1. Prepare local inputs

Acquire the required official Training assets and edit the source manifests using
[the source preparation guide](docs/sources.md). A typical local layout is:

<details>
<summary>Expected local input layout</summary>

```text
inputs/
  benchmark.jsonl                  # Fixed held-out benchmark; includes usable video paths
  official_source_registry.json    # Official split and physical source identities
  r2r_train.json.gz                # Official R2R Training route annotations
  matterport3d/                    # Licensed scenes and navigation meshes
  scannet/
    scans/
    scanrefer/ScanRefer_filtered_train.json
  arkitscenes/Training/             # RGB, depth, intrinsics, trajectories and 3D boxes
  arkit_train_sources.json          # Scan directories and official physical visit IDs
  EPIC_100_train.csv
  epic/                            # Original recordings named by video ID
  displacement_pilot_reviews.jsonl # At least five accepted displacement pilot reviews
  human_reviews.jsonl              # Actual candidate/sample review decisions
  accepted_corpus.jsonl            # Optional previously accepted corpus
  accepted_reference_targets.jsonl # Original targets for that corpus, when stored separately
  frozen_warmup.jsonl              # Warm-up partition to preserve exactly
work/                              # Generated candidates, media and export artifacts
```

</details>

Not every task needs every source: temporal tasks only need EPIC inputs, while
Object Direction can use ScanNet/ScanRefer or ARKitScenes. The benchmark and
source registry used for final packaging must cover the relevant source groups.

Use `configs/arkit_sources.example.json` and
`configs/source_registry.example.json` as format templates; replace all
placeholders with official identities. ARKitScenes scans of the same physical
visit must share a visit ID. These templates are not pre-approved source lists.

### 2. Prepare navigation parents

This branch supplies the continuous videos and measured poses used by
Egocentric Displacement, Forward Route Planning, Route-Shape Classification and
Reverse Route Planning. It requires the navigation dependencies and Habitat-Sim.

<details>
<summary>Navigation parent preparation: annotations, rendering and landmarks</summary>

First enumerate routes from the official Training annotations:

```bash
python -m data_generator annotation-routes \
  --episodes inputs/r2r_train.json.gz \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/annotation_routes \
  --shards 1 --max-legs 6
```

Then render valid navigation parents:

```bash
python -m data_generator render-routes \
  --input-dir work/annotation_routes/shard_00 \
  --scenes-dir inputs/matterport3d \
  --output-dir work/navigation_parents \
  --count 5000 --speed-scale 0.35 --max-legs 6 \
  --max-near-black-ratio 0.10 --seed 20260928
```

The annotation command writes into shard directories, including `shard_00` when
`--shards 1` is used. The renderer reads `.actions.json` files directly from the
specified shard, not from its parent directory. Its output includes
`manifest.json`, continuous videos and corresponding measured-pose sidecars.

The requested count is an example production budget, not a guarantee of source
capacity or an accepted-data count. The later task selectors may require more
valid parents. Use a new output directory for a different rendering run and do
not change labels or relax gates to fill a shortfall. RxR guide/follower Training
annotations use the same annotation command with the appropriate
`--dataset-name` and separate output directories.

Forward Route Planning additionally needs visually grounded landmark labels.
With a local Qwen3-VL-8B-Instruct endpoint already running:

```bash
python -m data_generator landmarks \
  --dataset-dir work/navigation_parents \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --min-confidence 0.70
```

This command is a model client; it does not launch a model server. The observation
model supplies landmark labels, while measured pose rules determine route
answers and reference actions.

</details>

### 3. Generate candidates for the selected tasks

<details>
<summary>Generation commands for all eight tasks</summary>

**Forward Route Planning**

```bash
python -m data_generator route-tasks --task route_plan \
  --source work/navigation_parents --output work/route_plan \
  --count-per-task 1000
```

**Route-Shape Classification and Reverse Route Planning**

Check the original balanced selection before producing clips:

```bash
for task in route_shape reverse_route_plan; do
  python -m data_generator route-tasks --task "$task" \
    --source work/navigation_parents --output "work/$task" \
    --count-per-task 4000 --seed 20261003 --workers 2 --plan-only
done
```

If both plans pass, run the same commands without `--plan-only` to generate and
finalize the candidates:

```bash
for task in route_shape reverse_route_plan; do
  python -m data_generator route-tasks --task "$task" \
    --source work/navigation_parents --output "work/$task" \
    --count-per-task 4000 --seed 20261003 --workers 2
done
```

Each task retains the original class/answer-letter balancing, unique source
windows and cap of five clips per parent. The 4,000-record buffers are candidates;
formal acceptance and deduplication can reduce the final count.

**Egocentric Displacement**

```bash
python -m data_generator displacement \
  --source-dir work/navigation_parents \
  --scenes-dir inputs/matterport3d \
  --pilot-labels inputs/displacement_pilot_reviews.jsonl \
  --output-dir work/displacement --count 3200 --seed 20261002
```

The pilot file must contain at least five actual accepted reviews, including
correct-answer and correct-reasoning decisions. The planner requires distinct
measured routes and balances directions, reference endpoints and answer letters.
It reports insufficient capacity rather than counting variants as unique routes.

**Object Direction — ScanNet/ScanRefer**

```bash
python -m data_generator object-direction-scannet pool \
  --root inputs/scannet --benchmark inputs/benchmark.jsonl \
  --output-dir work/scannet_direction --target 300 \
  --per-scene 10 --min-unique-categories 3 --max-scenes 100
```

**Object Direction — ARKitScenes**

```bash
python -m data_generator object-direction-arkit \
  --source-manifest inputs/arkit_train_sources.json \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/arkit_direction --count 300 \
  --max-per-physical-scene 10
```

These branches retain the original observer/facing/target geometry, referent
uniqueness, visual support, connected co-visibility and target-extent checks.
ARKitScenes additionally uses calibrated measured-depth support.

**Landmark Route Planning**

```bash
python -m data_generator landmark-routes \
  --data-root inputs/arkitscenes/Training \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/landmark_routes --target 1000 \
  --max-per-scene 14 --video-fps 10 --seed 20261002
```

The generator searches strictly observed objects and continuous source windows.
Final packaging applies the cap of 14 across scans of the same physical visit,
including previously accepted examples.

**Event-Sequence Ordering and Adjacent Action**

```bash
python -m data_generator temporal --task event_order \
  --annotations inputs/EPIC_100_train.csv --video-root inputs/epic \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/event_order --count 2000

python -m data_generator temporal --task before_or_after_action \
  --annotations inputs/EPIC_100_train.csv --video-root inputs/epic \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/adjacent_action --count 2400
```

When supplementing a temporal pool, pass earlier raw candidates through
`--existing` to exclude used event windows or anchor/neighbor pairs before
regeneration. Required actions must survive the uniform 24-frame model view.

</details>

### 4. Review candidates

Inspect the original continuous videos, questions, choices, answers and retained
geometry/temporal evidence. Record each review as one JSON object per line in
`inputs/human_reviews.jsonl`; `configs/review.template.json` shows the fields.
Individual acceptance requires `decision=accept`, `video_quality=pass`,
`question_unique=yes`, `gold_answer_correct=yes` and `gold_reasoning_correct=yes`.
Blank or incomplete decisions do not grant approval.

Bulk admission can instead use an explicitly approved source/task protocol after
concrete sample review. The approval must identify the reviewed samples, match
the exact quality-policy hash and authorize bulk generation and admission after
checks. It does not claim that every generated example was individually reviewed.
See [the approval format](docs/acceptance.md#reviewed-source-protocol-approval).

### 5. Validate, deduplicate and export

The example below extends an existing corpus while preserving its original
warm-up partition. Include only candidate files produced by your selected
branches; `--candidates` can be repeated.

<details>
<summary>Complete packaging command and required inputs</summary>

```bash
python -m data_generator package \
  --candidates work/scannet_direction/scannet_spatial_candidates.jsonl \
  --candidates work/arkit_direction/candidates.jsonl \
  --candidates work/route_plan/route_plan.jsonl \
  --candidates work/route_shape/route_shape.jsonl \
  --candidates work/reverse_route_plan/reverse_route_plan.jsonl \
  --candidates work/displacement/qa_candidates.jsonl \
  --candidates work/landmark_routes/qa_candidates.jsonl \
  --candidates work/event_order/candidates.jsonl \
  --candidates work/adjacent_action/candidates.jsonl \
  --benchmark inputs/benchmark.jsonl \
  --source-registry inputs/official_source_registry.json \
  --review-labels inputs/human_reviews.jsonl \
  --existing inputs/accepted_corpus.jsonl \
  --existing-targets inputs/accepted_reference_targets.jsonl \
  --warmup inputs/frozen_warmup.jsonl \
  --output-dir work/dataset_release
```

For a source-protocol approval, additionally supply
`--source-approvals inputs/source_approvals.json` and
`--quality-policy configs/quality_policy.json`. The source registry must cover
candidates, prior corpus records and the benchmark using official source/physical
identities. Existing records need their original reference targets, either in
the records themselves or in `--existing-targets`.

For a fresh corpus, omit `--existing` and `--existing-targets`, and supply a frozen
warm-up file selected from the reviewed records in that build. Warm-up IDs must
be a unique subset of the resulting corpus, and their task, answer and message
content must match. The exporter preserves that file's bytes; it does not
silently choose or reshuffle a warm-up split.

Packaging checks reference schema and reward-one validity, official Training
membership, benchmark physical-source isolation, full media decode, decoded
content/QA slots, source semantics and the combined existing/new caps.

</details>

### 6. Inspect the actual outputs

```text
work/dataset_release/
  corpus.jsonl               # Existing prefix plus accepted new records
  accepted_new.jsonl         # Newly accepted records with typed references
  warmup.jsonl               # Byte-identical copy of the supplied frozen warm-up
  policy.jsonl               # Non-warm-up model inputs, without answers or targets
  references.jsonl           # Hidden reference targets in the same order as policy.jsonl
  rejections.json            # Rejected candidate IDs and reasons
  acceptance_progress.json   # Durable acceptance progress
  manifest.json              # Actual accepted counts, per-task counts and input hashes
```

Use `manifest.json` to report dataset size. Requested counts, downloaded scans,
rendered parent videos and generated candidates are not accepted-example counts.
The full export contract is described in [acceptance and formats](docs/acceptance.md).

Keep the same inputs and configuration when resuming a supported stage. Use new
output directories for changed tasks or configurations, retain rejection and
partial-output records, and use a new directory for each final package export.
These commands do not start model training or upload the dataset.

## Documentation

| Guide | Contents |
|---|---|
| [Source preparation](docs/sources.md) | Official assets, local layouts, physical identities and source registries |
| [Generation recipes](docs/generation.md) | Task-specific commands, geometry/temporal rules and resume behavior |
| [Review and export](docs/acceptance.md) | Review formats, source-protocol approval and exported file contracts |
| [Installation checks](docs/validation.md) | Package entrypoints, syntax checks and runtime requirements |

## Acknowledgements

The pipeline reads annotations from [ScanNet](https://github.com/ScanNet/ScanNet),
[ScanRefer](https://github.com/daveredrum/ScanRefer),
[ARKitScenes](https://github.com/apple-aiml-research/ARKitScenes),
[Matterport3D](https://niessner.github.io/Matterport/),
[R2R](https://bringmeaspoon.org/),
[RxR](https://github.com/google-research-datasets/RxR) and
[EPIC-KITCHENS-100](https://epic-kitchens.github.io/2026) and uses
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for continuous
navigation rendering. These datasets and dependencies retain their own licenses
and citation requirements. Dataset assets and third-party model weights are not
redistributed by this code repository.

## Contact

For questions about the datasets or data-generation pipeline, please contact
[yiling.ma@yale.edu](mailto:yiling.ma@yale.edu).
