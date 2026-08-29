# v1.0.1 release checklist

This checklist closes the gap between the historical `v1.0.0` repository snapshot and the final JBI submission package.

## Already prepared on this branch

- [x] Clarify the executed ClinicalTrials.gov cutoff as `startDate >= 2017-01-01`.
- [x] Distinguish the 137 external Disease query labels from the 134 connected model-side Disease tails.
- [x] Synchronize the entity-alignment report with the audited v4 protocol wording.
- [x] Add `requirements-core.txt` for dependency-light CI checks.
- [x] Add `requirements.txt` for the PyKEEN/Torch model environment.
- [x] Expand CI to compile released Python sources and run the dependency-light statistical/baseline tests.

## Required before publishing `v1.0.1`

- [ ] Synchronize the final manuscript-facing pipeline figure (`fig5_pipeline`) with the public release figure used for the article. Keep the old `v1.0.0` asset immutable.
- [ ] Review any other binary figure differences between the final submission package and the reproducibility tree; only replace files that are actually manuscript-facing or release evidence.
- [ ] Update `CITATION.cff` to `version: 1.0.1` and the actual publication date immediately before tagging.
- [ ] Regenerate `RELEASE_MANIFEST.sha256` from the exact candidate tree after every text/binary change is complete.
- [ ] Confirm the candidate tree contains no API caches, raw third-party datasets, credentials, QA/session traces, or model/checkpoint binaries without cleared redistribution rights.
- [ ] Confirm CI passes on the final candidate commit.
- [ ] Create tag/release `v1.0.1` from that exact commit; do not move or rewrite `v1.0.0`.
- [ ] Change the manuscript Code/Data Availability release URL from `v1.0.0` to `v1.0.1` only after the new release exists.
- [ ] If an archival DOI is minted, archive the exact `v1.0.1` tag and add the DOI to citation metadata without changing the tagged research artifacts afterward.

## Release boundary

`v1.0.1` is a patch-level reproducibility synchronization release, not a new experiment. The audited protocol addendum documents the executed boundary and the 137/134 universe distinction; it does not promote post-hoc sensitivity analyses to primary results or alter the frozen result JSONs.
