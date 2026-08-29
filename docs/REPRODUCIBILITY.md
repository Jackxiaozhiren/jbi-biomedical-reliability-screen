# Reproducibility and Release Integrity

## Scope

The repository is a bounded reproducibility release: it contains analysis code, derived evidence, external-validation summaries, figures, and model metadata, while intentionally excluding third-party or sensitive materials whose redistribution is not appropriate.

## Existing release

`v1.0.0` is an immutable historical release state for citation/audit purposes. Its `RELEASE_MANIFEST.sha256` should be interpreted against that release snapshot and should not be rewritten merely because `main` later receives documentation, licensing, or CI improvements.

## Next metadata release

Changes that add the explicit software license, rights clarification, third-party notices, or CI should be published as a new patch release (recommended `v1.0.1`) rather than retroactively changing the meaning of `v1.0.0`.

Before publishing that patch release:

1. regenerate a SHA-256 manifest for the exact release tree;
2. verify citation metadata and release notes;
3. run the statistical-core CI checks;
4. confirm that excluded third-party assets remain excluded;
5. archive the exact tag with a persistent DOI if an archival service is used.

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
