# Contributing to jbi-biomedical-reliability-screen

Thanks for helping make the research **auditable, reproducible, and accurate**. This repository focuses on external-validation and reliability analysis for biomedical knowledge-graph screening.

## Before proposing a change

1. Read the [README](README.md) for the active research scope and follow the existing reproducibility instructions.
2. Check whether your proposal corrects a **documentation issue**, a **software defect**, or a **scientific-method issue**. They have different evidence requirements.
3. Open an issue with a minimal reproducible example when the expected behavior or validity of a scientific claim is uncertain.

## Validation

Use the repository's GitHub Actions workflow as the canonical CI reference. Relevant commands:

```bash
python -m compileall -q code external_validation figures
(cd code && pytest -q test_stats_core.py test_baselines.py -k "not scores_shape_and_range")
```

If a test requires unavailable external datasets or models, state what was skipped and why. Do not claim a complete reproduction based solely on a smoke test.

## Scientific integrity and release policy

Protect the frozen JBI-specific evidence boundary. Do not add licensed biomedical datasets, personal health data, proprietary models, or claim clinical validity beyond the versioned evidence. Preserve separation from historical fdr-kg work.

- State the **base commit**, exact command, Python/tool versions, input provenance, and random seed when reporting an experiment.
- Separate observed measurements from interpretation or hypotheses. No fabricated metrics, stars, acceptance claims, DOI links, or environment guarantees.
- Explain any change to assumptions, preprocessing, population, evaluation protocol, or multiple-testing procedure.
- Do not commit credentials, data covered by third-party access terms, private identifiers, oversized derived data, or unlicensed assets.
- If the repository includes a SHA-256 release manifest, regenerate it deterministically when changing tracked files, and explain that the new manifest covers **the working tree**, not the historical release tag.

## Pull request checklist

Keep changes narrowly scoped, link relevant issues, and describe tests run and limitations. Use the repository's pull-request template. Version tags and archival evidence are not edited in place.
