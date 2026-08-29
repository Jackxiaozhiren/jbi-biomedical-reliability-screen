# Reproducibility and Release Integrity

## Scope

The repository is a bounded reproducibility release: it contains analysis code, derived evidence, external-validation summaries, figures, and model metadata, while intentionally excluding third-party or sensitive materials whose redistribution is not appropriate.

## Existing release

`v1.0.0` is an immutable historical release state for citation/audit purposes. Its `RELEASE_MANIFEST.sha256` should be interpreted against that release snapshot and should not be rewritten merely because `main` later receives documentation, licensing, CI, protocol-clarification, or figure-generation improvements.

## v1.0.1 synchronization patch

The patch release is intended to align the public reproducibility materials with the final JBI submission evidence boundary without changing the frozen research results. Its preparation includes the audited ClinicalTrials.gov boundary clarification, the 137-external-query/134-model-tail distinction, synchronized result reporting, full-precision candidate exports, the collision-free Figure 5 generator, explicit dependency files, and expanded CI.

The exact manuscript-facing Figure 5 asset hashes and regeneration verification are recorded in `FIG5_PROVENANCE.md`.

## Release-integrity procedure

Before publishing a new versioned release:

1. finish all scientific/documentation changes on the candidate branch;
2. confirm excluded third-party assets, API caches, model/checkpoint binaries, QA/session traces, and credentials are absent;
3. run the statistical-core CI checks;
4. update `CITATION.cff` to the version and actual release date;
5. run `python code/generate_release_manifest.py --write` and commit the regenerated manifest as the final content change;
6. confirm CI again on that exact commit;
7. tag that exact commit and do not subsequently move or rewrite the tag;
8. if an archival service is used, archive that exact tag before adding any DOI claim to downstream manuscript text.

## Environment boundary

A generic checkout is not claimed to recreate every upstream API response or retrain every external model. Exact reruns requiring third-party data/models/API access must follow the frozen protocol and upstream terms. The released derived evidence remains the verification boundary where those upstream materials cannot be redistributed.

## Archival environment capture

For the next verified archival release, capture the exact execution environment from the workstation or container used for the final validated run. Do not infer, backfill, or fabricate historical dependency versions.

Record at minimum:

- Python version;
- operating system and architecture;
- exact installed package versions (for example, `python -m pip freeze --all` or an exact Conda export);
- Git commit SHA for the verified release tree;
- external dataset, model, and API identifiers plus retrieval dates where applicable;
- checksums for frozen inputs and released derived evidence where redistribution and licensing permit.

Keep this environment snapshot paired with the release tag and its SHA-256 manifest. If an upstream dependency or API cannot be frozen, document the provider, identifier/version, retrieval procedure, and the reproducibility boundary rather than substituting guessed versions.
