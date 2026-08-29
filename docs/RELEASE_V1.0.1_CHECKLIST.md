# v1.0.1 release checklist

This checklist closes the gap between the historical `v1.0.0` repository snapshot and the final JBI submission package.

## Already prepared on this branch

- [x] Clarify the executed ClinicalTrials.gov cutoff as `startDate >= 2017-01-01` and mirror that boundary in the fetcher documentation.
- [x] Distinguish the 137 external Disease query labels from the 134 connected model-side Disease tails.
- [x] Synchronize the entity-alignment report with the audited v4 protocol wording.
- [x] Synchronize `analysis/MASTER_REPORT.md` with the audited v4 result summaries and current frozen JSON exports.
- [x] Preserve full candidate p-value/score precision in `external_validation/export_candidates.py` so stored KEEP/WITHHOLD flags remain reproducible at threshold ties.
- [x] Add `requirements-core.txt` for dependency-light CI checks.
- [x] Add `requirements.txt` for the PyKEEN/Torch model environment.
- [x] Expand CI to compile released Python sources and run the dependency-light statistical/baseline tests while retaining read-only permissions and immutable action pins.
- [x] Synchronize the manuscript Figure 5 generator to the collision-free v5 layout and record canonical manuscript asset hashes in `docs/FIG5_PROVENANCE.md`.
- [x] Remove stale pre-hardening `analysis/master_pipeline.{pdf,png}` binaries from the patch candidate; they remain preserved in the immutable `v1.0.0` tag. The patch regenerates Figure 5 from frozen result JSONs rather than committing a PDF whose byte hash varies with generation metadata.
- [x] Review the other manuscript-facing result figures. Their v4/public binary differences are rendering metadata only (pixel-identical at the audited raster comparison) or exact matches, so no scientific replacement is required.
- [x] Audit the candidate tree for release-boundary exclusions: no API cache directory, `.mimosa` session trace, QA tree, checkpoint/model binary, or raw third-party dataset is present.
- [x] Add `code/generate_release_manifest.py` so the checksum manifest can be rebuilt deterministically from the exact tracked release tree.
- [x] Extend CI to generate/export a candidate release manifest and verify it against the checked-in `RELEASE_MANIFEST.sha256`.
- [x] Incorporate the exact archival environment-capture policy from current `main`.
- [x] Update `CITATION.cff` to `version: 1.0.1` with release date `2026-08-29`.

## Required immediately before publishing `v1.0.1`

- [ ] Generate the final candidate manifest from the exact tracked tree, review it, and replace `RELEASE_MANIFEST.sha256` as the final content change.
- [ ] Confirm CI passes on that exact manifest commit, including the manifest equality check.
- [ ] Merge the reviewed PR and create tag/release `v1.0.1` from the merged release commit; do not move or rewrite `v1.0.0`.
- [ ] Change the manuscript Code/Data Availability release URL from `v1.0.0` to `v1.0.1` only after the new release exists.
- [ ] If an archival DOI is minted, archive the exact `v1.0.1` tag and add the DOI to citation metadata without changing the tagged research artifacts afterward.

## Release boundary

`v1.0.1` is a patch-level reproducibility synchronization release, not a new experiment. The audited protocol addendum documents the executed boundary and the 137/134 universe distinction; it does not promote post-hoc sensitivity analyses to primary results or alter the frozen result JSONs.

## Why the candidate remains a draft

The branch remains a draft only until the deterministic `RELEASE_MANIFEST.sha256` is synchronized with this exact tree and CI confirms equality. All other release-specific metadata has been frozen for the planned `2026-08-29` patch release.
