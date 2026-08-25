# JBI Biomedical Reliability Screen — Code and Derived Results

This repository contains the JBI-specific reproducibility materials for:

> **An externally validated reliability audit for knowledge-graph-based drug repurposing screens**

The release covers a conditional keep/withhold reliability screen for biomedical knowledge-graph link predictions. It includes the frozen protocol, analysis code, derived result exports, compact external-validation summaries, and generated analysis figures.

## Scope and interpretation

The reported quantities describe reliability and enrichment under the declared sampling, alignment, and time-sliced validation protocols. They are not evidence of treatment efficacy, causal effect, clinical deployment success, or universal validity across knowledge graphs and model architectures.

## Contents

- `code/` — analysis, audit, evaluation, and deterministic test scripts;
- `external_validation/` — frozen protocol, fetch/alignment scripts, compact derived evidence tables, and summary reports;
- `results/` — derived JSON result exports used by the manuscript;
- `analysis/` and `figures/` — derived reports and generated figures;
- `models/manifest.json` — model metadata and result-to-model path manifest.

## Deliberate exclusions

This public candidate excludes `.mimosa/` session traces, QA directories, private paths, credentials, API caches, large candidate-level exports, raw third-party datasets, and model/checkpoint binaries whose redistribution terms have not been independently cleared. The excluded large candidate exports are retained locally only and are not part of this repository.

The source public datasets and external APIs remain subject to their own terms. Readers should obtain source data from the original providers and use the frozen protocol and derived exports for interpretation. Model binaries can be added only after their licenses permit redistribution.

## Reproduction boundary

The derived JSON exports are the frozen evidence used for the manuscript. A generic checkout is not claimed to retrain every model or recreate every external API response without the corresponding permitted data, model files, API access, and environment. No cached API response is redistributed in this release.

## Citation

Please cite the JBI article and the versioned release of this repository. The version-specific citation record is described in `CITATION.cff`.

## Status

This is the JBI-specific public release `v1.0.0`. It is separate from the earlier `fdr-kg` repository and does not contain that repository's KBS/ESWA historical materials.
