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
- [x] Remove stale pre-hardening `analysis/master_pipeline.{pdf,png}` binaries from the patch candidate; they remain preserved in the immutable `v1.0.0` tag.
- [x] Review the other manuscript-facing result figures. Their v4/public binary differences are rendering metadata only (pixel-identical at the audited raster comparison) or exact matches, so no scientific replacement is required.
- [x] Audit the candidate tree for release-boundary exclusions: no API cache directory, `.mimosa` session trace, QA tree, checkpoint/model binary, or raw third-party dataset is present.
- [x] Add `code/generate_release_manifest.py` so the checksum manifest can be rebuilt deterministically from the exact tracked release tree.
- [x] Add a CI dry-run of the release-manifest generator.

## Required immediately before publishing `v1.0.1`

- [ ] Confirm CI passes on the final candidate commit after the last text/code change.
- [ ] Update `CITATION.cff` to `version: 1.0.1` and the actual publication date.
- [ ] Run `python code/generate_release_manifest.py --write`, review the diff, and commit the regenerated `RELEASE_MANIFEST.sha256` as the final content change.
- [ ] Reconfirm CI on that exact manifest commit.
- [ ] Create tag/release `v1.0.1` from that exact commit; do not move or rewrite `v1.0.0`.
- [ ] Change the manuscript Code/Data Availability release URL from `v1.0.0` to `v1.0.1` only after the new release exists.
- [ ] If an archival DOI is minted, archive the exact `v1.0.1` tag and add the DOI to citation metadata without changing the tagged research artifacts afterward.

## Release boundary

`v1.0.1` is a patch-level reproducibility synchronization release, not a new experiment. The audited protocol addendum documents the executed boundary and the 137/134 universe distinction; it does not promote post-hoc sensitivity analyses to primary results or alter the frozen result JSONs.

## Why the candidate remains a draft

The branch is intentionally kept as a draft until the last two release-specific files are finalized: `CITATION.cff` and `RELEASE_MANIFEST.sha256`. Those values depend on the actual release date and the exact final tracked tree, so freezing them earlier would make the release metadata stale again.
