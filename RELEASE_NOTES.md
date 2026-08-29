# Release notes

## v1.0.1 — reproducibility synchronization patch (candidate)

This patch prepares the repository to match the final JBI submission evidence boundary without rewriting the historical `v1.0.0` release.

Changes prepared for the patch:

- documents the executed inclusive ClinicalTrials.gov boundary, `startDate >= 2017-01-01`, including the fetcher-side boundary note;
- distinguishes 137 external Disease query labels from 134 connected model-side Disease tails;
- synchronizes the entity-alignment report and master result report with the audited v4 evidence state;
- preserves full candidate p-value/score precision so exported KEEP/WITHHOLD flags remain reproducible at threshold ties;
- synchronizes the manuscript Figure 5 generator to the final collision-free v5 layout and records the canonical manuscript asset hashes in `docs/FIG5_PROVENANCE.md`;
- removes the stale pre-hardening pipeline binaries from the patch candidate while preserving them in the immutable `v1.0.0` tag;
- adds explicit dependency files for lightweight CI and the PyKEEN/Torch model environment;
- expands CI to compile released Python sources, exercise dependency-light statistical/baseline tests, and dry-run deterministic release-manifest generation;
- adds a deterministic `code/generate_release_manifest.py` helper for final release checksums;
- incorporates the archival environment-capture policy for the exact validated workstation/container state.

Before the tag is published, `CITATION.cff` must be updated to the actual `v1.0.1` publication date and `RELEASE_MANIFEST.sha256` must be regenerated from the exact final tracked tree. See `docs/RELEASE_V1.0.1_CHECKLIST.md`.

## v1.0.0

Initial JBI-specific public release containing code, the frozen protocol, compact derived external-validation artifacts, derived JSON results, analysis reports, generated figures, and citation metadata.

Excluded by design: session traces, QA directories, private paths, credentials, API caches, large candidate-level exports, raw third-party datasets, and model/checkpoint binaries whose redistribution terms were not cleared.

This release has a GitHub version URL. No DOI is claimed until an author-controlled archival service publishes this exact release.
