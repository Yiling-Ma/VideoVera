# Candidate acceptance and export formats

## Inputs

`videovera package` accepts raw candidate JSONL files, the fixed benchmark,
previously accepted records, official source identities and review decisions.
It does not replace source geometry with model-generated labels.

```bash
videovera package \
  --candidates work/scannet_direction/scannet_spatial_candidates.jsonl \
  --candidates work/arkit_direction/candidates.jsonl \
  --candidates work/forward_routes/route_plan.jsonl \
  --candidates work/route_shapes/route_shape.jsonl \
  --candidates work/reverse_routes/reverse_route_plan.jsonl \
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

Omit `--existing` and `--existing-targets` for a new corpus. Existing records must
already contain `structured_gold` or be accompanied by their original reference
targets; the exporter does not infer missing ground truth from a model response.
The provided warm-up IDs must be a unique subset of the resulting corpus and the
protected task, answer and message content must agree.

A review row requires every decision below. A blank, missing or partial decision
is not approval. Store actual review decisions locally; no completed review is
included as a repository example.

```json
{
  "id": "candidate_identifier",
  "decision": "accept",
  "video_quality": "pass",
  "question_unique": "yes",
  "gold_answer_correct": "yes",
  "gold_reasoning_correct": "yes"
}
```

## Checks

For each new candidate the exporter checks explicit review, official Training
source membership, physical source separation from the benchmark, unique IDs,
source semantics, question/content slots and combined source caps. ARKitScenes
rescans share their official physical visit identity. ScanNet scans share their
physical scene. Temporal groups are official source recordings, not clip IDs.

Task-specific target constructors derive typed references from the retained
geometry, measured pose actions or action segments. Schema-valid reference
completions must attain reward one with a correct answer and exact trace under
the ordered verifier. Text/claim-count mismatches are rejected, not padded with
invented claims. Shape and displacement use the finalized minimal canonical
claim conventions. Full media decode and the original 24 uniformly sampled,
64×64 decoded-frame content fingerprint detect corrupted inputs and content
aliases. The fingerprint is a content-deduplication check, not a claim to solve
all possible perceptual near-duplicate detection.

The original task selectors also enforce their own geometry, visibility,
continuous-window, class/letter balance and parent caps. Packaging complements
these generators; a source registry or review file is not a replacement for
running the task generator and retaining its source evidence.

## Outputs

| File | Contents |
|---|---|
| `corpus.jsonl` | Existing corpus prefix preserved verbatim, followed by accepted new records |
| `accepted_new.jsonl` | Only newly accepted records with annotation-derived typed references |
| `warmup.jsonl` | Supplied frozen warm-up file copied byte-for-byte |
| `policy.jsonl` | Non-warm-up user messages and video inputs; no answers, assistant messages or verifier state |
| `references.jsonl` | Matching hidden reference targets, in policy order |
| `rejections.json` | Candidate identifiers, source files and rejection reasons |
| `acceptance_progress.json` | Durable acceptance/rejection progress |
| `manifest.json` | Actual counts, per-task counts and protected input hashes |

The output directory must be new. Input candidates, source clips, rejected
records and earlier outputs are never deleted. Counts describe actual accepted
records; a requested generator count is not presented as a completed dataset.
The exporter neither uploads files nor launches training.

## Protocol compatibility

Task keys such as `relative_direction`, `route_shape` and
`object_grounded_route_planning` are the serialized dataset keys. The paper uses
Object Direction, Route-Shape Classification and Landmark Route Planning for
these tasks. Historical schema identifiers remain readable even though public
filenames, module names and commands have no experiment-version suffixes.

This release contains only the reference-validation subset needed by data
construction. Model evaluation reports, aggregate benchmark analysis and error
analysis are intentionally outside this repository's current scope.

## Reviewed source-protocol approval

The production workflow also supports approval of a source/task protocol after
reviewing concrete sample videos, questions, answers, visibility and geometry.
It does not require every later candidate to be individually reviewed when the
human has explicitly approved bulk generation and admission under that protocol.

For this workflow provide `--source-approvals inputs/source_approvals.json` and
`--quality-policy configs/quality_policy.json` in addition to `--review-labels`.
The approval is a JSON list. Each entry must specify `family`, `task_type`,
`decision: "approve_pipeline"`, `allow_bulk_generation: true`,
`allow_admission_after_checks: true`, a nonempty `reviewer` and `statement`,
`reviewed_sample_ids` whose five review fields all pass, and `protocol_sha256`
matching the exact quality-policy file. No approval file is generated by the CLI.

These candidates still undergo the source, geometry, media, schema, deduplication
and cap checks. Output metadata explicitly distinguishes individual review from
source-protocol approval; it never claims that a bulk-admitted example received
individual human review. Silence, a blank sample review or a changed protocol
hash cannot authorize bulk admission.
