# JBI Biomedical Reliability Screen — Code and Derived Results

This repository contains the JBI-specific reproducibility materials for:

> **An externally validated reliability audit for knowledge-graph-based drug repurposing screens**

The release covers a conditional keep/withhold reliability screen for biomedical knowledge-graph link predictions. It includes the frozen protocol, analysis code, derived result exports, compact external-validation summaries, and generated analysis figures.

## Scope and interpretation

The reported quantities describe reliability and enrichment under the declared sampling, alignment, and time-sliced validation protocols. They are not evidence of treatment efficacy, causal effect, clinical deployment success, or universal validity across knowledge graphs and model architectures.

## Contents

- `code/` — analysis, audit, evaluation, deterministic test scripts, the synchronized manuscript Figure 5 generator, and release-manifest helper;
- `external_validation/` — frozen protocol, fetch/alignment scripts, compact derived evidence tables, and summary reports;
- `results/` — derived JSON result exports used by the manuscript;
- `analysis/` and `figures/` — derived reports and generated figures;
- `models/manifest.json` — model metadata and result-to-model path manifest;
- `docs/FIG5_PROVENANCE.md` — manuscript Figure 5 hash/provenance and v5 layout synchronization record;
- `RELEASE_MANIFEST.sha256` — checksum manifest for the corresponding frozen release state;
- `CITATION.cff` — citation metadata.

## Environment

For the dependency-light statistical and baseline checks used by CI:

```bash
python -m pip install -r requirements-core.txt
```

For scripts that load or train PyKEEN models:

```bash
python -m pip install -r requirements.txt
```

The full environment records the model-stack compatibility pair documented in `code/env_patch.py` (`pykeen==1.10.1` with PyTorch 2.8.x). The public repository remains a bounded reproducibility release: model/checkpoint binaries and upstream data are not redistributed unless their terms permit it.

To regenerate the manuscript-facing pipeline figure from the frozen result JSONs:

```bash
python code/generate_pipeline_fig.py
```

The old pre-hardening pipeline binaries remain preserved in the immutable `v1.0.0` tag; the `v1.0.1` candidate uses the synchronized v5 generator instead. See `docs/FIG5_PROVENANCE.md` for the canonical manuscript asset hashes and verification record.

Immediately before publishing a versioned release, regenerate the tracked-tree checksum manifest with:

```bash
python code/generate_release_manifest.py --write
```

## Deliberate exclusions

This public repository excludes `.mimosa/` session traces, QA directories, private paths, credentials, API caches, large candidate-level exports, raw third-party datasets, and model/checkpoint binaries whose redistribution terms have not been independently cleared. The excluded large candidate exports are retained locally only and are not part of the public release.

The source public datasets and external APIs remain subject to their own terms. Readers should obtain source data from the original providers and use the frozen protocol and derived exports for interpretation. Model binaries can be added only after their licenses permit redistribution.

## Reproduction boundary

The derived JSON exports are the frozen evidence used for the manuscript. A generic checkout is not claimed to retrain every model or recreate every external API response without the corresponding permitted data, model files, API access, and environment. No cached API response is redistributed in this release. Candidate exports are written at full floating-point precision so stored threshold decisions can be audited without avoidable formatting loss.

See `docs/REPRODUCIBILITY.md` for release-integrity guidance.

## Citation

Please cite the JBI article and the versioned release of this repository. The version-specific citation record is described in `CITATION.cff`.

## License and third-party terms

Project-authored source code is licensed under the BSD 3-Clause License; see `LICENSE`. This does not relicense third-party datasets, APIs, models, checkpoints, or external software. See `RIGHTS_AND_TERMS.md` and `THIRD_PARTY_NOTICES.md`.

## Version status

The existing `v1.0.0` release remains an immutable historical release snapshot. Licensing/rights/CI metadata and the audited protocol clarification added after that tag should be published as a new patch release (recommended `v1.0.1`) with a regenerated checksum manifest rather than modifying the old release in place.

This JBI-specific repository is separate from the earlier `fdr-kg` repository and does not contain that repository's historical submission materials.
