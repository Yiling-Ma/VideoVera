# Source preparation

Obtain assets from the original dataset providers. The code does not accept
licenses, store credentials or download gated data on a user's behalf.

## ScanNet / ScanRefer

Provide `scans/<scene_id>/` with the official `.sens`, aggregation JSON, mesh,
segmentation and filtered instance masks; place the official
`ScanRefer_filtered_train.json` in `scanrefer/`. Scene identity must be resolved
at the physical scene level across scan suffixes.

The spatial scene builder can resume transfers of required files from the
original ScanNet endpoint. Access must already have been approved. No access
token or credential belongs in a config committed to Git.

## ARKitScenes

Provide official 3DOD **Training** scans containing the object annotation JSON,
calibrated RGB/depth frames, intrinsics and camera trajectory. The reader
supports the original scan layout via `scene_paths` in the landmark generator.
Depth support is measured against official oriented 3D boxes; it is not
represented as official 2D instance ground truth.

Supply the official scan-to-visit mapping. Multiple scans of the same visit share
a single physical scene and a single cap. A missing visit ID must be resolved;
it must not be replaced with the scan ID. The Object Direction source-manifest
format is illustrated in `configs/arkit_sources.example.json`.

## Matterport3D / R2R / RxR

Provide the locally licensed Matterport3D scene assets and official Training
route annotations. `annotation-routes` enumerates original reference geometries;
`render-routes` projects them onto the navmesh, emits continuous RGB video and
records actual per-frame camera poses. Do not use Val/Test route annotations as
Training inputs. Supply the fixed benchmark manifest to exclude held-out sources.

Visual landmark extraction uses the local Qwen3-VL-8B-Instruct endpoint configured
by `--model` and `--base-url`. The production recipe uses BF16, temperature zero,
confidence threshold 0.70, max model length 8192 and serialized requests. The
client does not start a server or allocate GPUs. Model deployment is independent
of the data generator; no paid API is required.

## EPIC-KITCHENS-100

Provide official Training recordings and `EPIC_100_train.csv`. `temporal`
recursively finds local MP4 recordings named by their official video IDs. The
benchmark's source recordings are excluded before generation. Continuous output
clips are rendered from original videos, not from a collection of selected
support images.

## Benchmark and physical-source registry

`package` needs the fixed benchmark JSONL and a source registry. Registry entries
must be derived from official source metadata and cover every candidate, prior
corpus source and benchmark source. Example structure:

```json
{
  "scannet": {"scene0001_00": {"split": "train", "physical_scene_id": "scene0001"}},
  "arkitscenes": {"12345678": {"split": "Training", "physical_scene_id": "visit_42"}},
  "r2r": {"train_scene_id": {"split": "train", "physical_scene_id": "train_scene_id"}},
  "epic": {"P01_01": {"split": "train", "physical_scene_id": "P01_01"}}
}
```

The values above illustrate the format; they are not an approved split registry.
Never infer independence from different filenames or substitute a scan identifier
for a physical visit. Source datasets and benchmark labels are supplied locally
and are not included in this repository.
