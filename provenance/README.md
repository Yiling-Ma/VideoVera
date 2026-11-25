# Production-rule provenance

`components.json` records the current public Python component hashes and the
hashes of the production source components used for each port. New portable
integration modules are marked separately. Machine-specific paths, source
assets, experimental outputs and private source-directory mappings are not
included.

`core_rules.json` records AST fingerprints for ten representative production
functions: pose extraction, route questions, shape rules, stationary-turn rules,
route selection, egocentric direction classification, continuous landmark-video
quality, spatial evidence windows, target-extent checks and ordered claim
matching. These functions were compared with the production implementations
without source-location attributes. Renaming imports and separating entrypoints
does not authorize label or threshold changes.

The portable integration additionally retains failed/partial outputs instead of
deleting them, resolves RxR follower identity without changing geometry, excludes
already used reference geometry, binds reviewed source approvals to a policy
hash, and preserves frozen warm-up bytes. Reference geometry identity is not a
substitute for validation of the actual rendered navmesh trajectory.

Only code provenance is recorded here. This directory contains no experimental
results, error-case analysis, dataset records or human-review decisions.
