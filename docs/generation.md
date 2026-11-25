# Generation recipe

All paths below are user-supplied local paths. Commands generate candidates;
formal acceptance is a separate step. `--help` shows task-specific options.

## Navigation parents and Forward Route Planning

```bash
videovera annotation-routes \
  --episodes inputs/r2r_train.json.gz \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/annotation_routes --max-legs 6

videovera render-routes \
  --input-dir work/annotation_routes \
  --scenes-dir inputs/matterport3d \
  --output-dir work/navigation_parents \
  --count 2000 --speed-scale 0.35 --max-legs 6 \
  --max-near-black-ratio 0.10 --seed 20260928

videovera landmarks \
  --dataset-dir work/navigation_parents \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1 --min-confidence 0.70

videovera route-tasks --task route_plan \
  --source work/navigation_parents --output work/forward_routes \
  --count-per-task 1000
```

Rendering requires Habitat-Sim, navigation extras and the scene assets. Landmark
labels must be grounded in the actual waypoint views before forward questions
are constructed. Models produce landmark observations; annotation/pose rules
still determine the question's answer and reference actions.

## Shape and reverse planning

```bash
videovera route-tasks --task route_shape \
  --source work/navigation_parents --output work/route_shapes \
  --count-per-task 4000 --seed 20261003 --workers 2

videovera route-tasks --task reverse_route_plan \
  --source work/navigation_parents --output work/reverse_routes \
  --count-per-task 4000 --seed 20261003 --workers 2
```

Each task is selected independently using the original class/letter balance,
unique windows and maximum five clips per parent. The shape vocabulary is
straight line, L-shape, U-shape, Manhattan-style piecewise linear and in-place
orientation. Reverse action classes remain unchanged. Selection failure is a
capacity shortfall; do not lower class quotas or relabel a trajectory to proceed.

`--plan-only` performs selection without producing clips. The finalizer checks
media hashes and replaces duplicate media using valid unused windows from the
same pool. Completed clips and old checkpoints remain available.

## Egocentric Displacement

```bash
videovera displacement \
  --source-dir work/navigation_parents \
  --scenes-dir inputs/matterport3d \
  --pilot-labels inputs/displacement_pilot_reviews.jsonl \
  --output-dir work/displacement --count 3200 --seed 20261002
```

The original five accepted pilot-review requirement is retained. The planner
balances direction labels, question reference endpoints and answer letters. It
requires distinct measured route keys; combinatorial variants are not counted
as independent routes. Additional geometry is rendered in Habitat-Sim, not by
reversing compressed RGB frames or changing an answer label.

## Object Direction

```bash
videovera object-direction-scannet pool \
  --root inputs/scannet --benchmark inputs/benchmark.jsonl \
  --output-dir work/scannet_direction --target 300 \
  --per-scene 10 --min-unique-categories 3 --max-scenes 100

videovera object-direction-arkit \
  --source-manifest inputs/arkit_train_sources.json \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/arkit_direction --count 300 \
  --max-per-physical-scene 10
```

The ScanNet search can use scenes with three distinct referent categories; its
per-example geometry, visibility, uniqueness, connected co-visibility and
continuous-video gates remain unchanged. ARKitScenes uses calibrated measured
depth, exclusive supported pixels, referent uniqueness and the same spatial
coordinate and target-extent rules. Both outputs require review before packaging.

## Landmark Route Planning

```bash
videovera landmark-routes \
  --data-root inputs/arkitscenes/Training \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/landmark_routes --target 1000 \
  --max-per-scene 14 --video-fps 10 --seed 20261002
```

Dense window search considers additional strictly observed frames and continuous
boundaries, then applies the same original quality function. Each named object
needs at least three strict source-frame hits and at least one uniform model
sample. Clips retain their original continuous interval, calibrated poses and
orientation; duration is 8–40 seconds with the original continuity and camera
pitch gates. Packaging applies the cap of 14 across all scans of a physical visit,
including previously accepted records.

## Temporal tasks

```bash
videovera temporal --task event_order \
  --annotations inputs/EPIC_100_train.csv --video-root inputs/epic \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/event_order --count 2000

videovera temporal --task before_or_after_action \
  --annotations inputs/EPIC_100_train.csv --video-root inputs/epic \
  --benchmark inputs/benchmark.jsonl \
  --output-dir work/adjacent_action --count 2400
```

Use `--existing` with earlier raw candidates when supplementing a pool. Event
windows and anchor/neighbor identities are excluded before regeneration. Uniform
24-frame support and original temporal ambiguity checks apply. Caps are 45
ordering and 50 adjacent-action examples per source recording; packaging counts
existing records toward the same caps.

## Resume and failure handling

Keep output configurations immutable. Successful individual results, partial
media, rejected candidates and failure records remain on disk. Do not rerun a
failed stage into a directory with changed input manifests or a different task.
Do not use `--overwrite` for normal production. New snapshots belong in a new
output directory. Generators do not create recurring jobs, stop other users'
processes, publish a dataset or start model training.
