# Release notes

## v1.0.1 — reproducibility synchronization patch (candidate)

This patch prepares the repository to match the final JBI submission evidence boundary without rewriting the historical `v1.0.0` release.

Changes prepared for the patch:

- documents the executed inclusive ClinicalTrials.gov boundary, `startDate >= 2017-01-01`;
- distinguishes 137 external Disease query labels from 134 connected model-side Disease tails;
- synchronizes the entity-alignment report with the audited v4 protocol wording;
- adds explicit dependency files for lightweight CI and the PyKEEN/Torch model environment;
- expands CI to compile the released Python sources and exercise the dependency-light statistical/baseline tests.

Before the tag is published, the final manuscript-facing pipeline figure must be synchronized, `CITATION.cff` must be updated to the actual `v1.0.1` publication date, and `RELEASE_MANIFEST.sha256` must be regenerated from the exact final tree. See `docs/RELEASE_V1.0.1_CHECKLIST.md`.

## v1.0.0

Initial JBI-specific public release containing code, the frozen protocol, compact derived external-validation artifacts, derived JSON results, analysis reports, generated figures, and citation metadata.

Excluded by design: session traces, QA directories, private paths, credentials, API caches, large candidate-level exports, raw third-party datasets, and model/checkpoint binaries whose redistribution terms were not cleared.

This release has a GitHub version URL. No DOI is claimed until an author-controlled archival service publishes this exact release.
