# Figure 5 pipeline provenance

This note records the synchronization of the manuscript-facing pipeline figure for the `v1.0.1` reproducibility patch.

## Canonical manuscript asset

The final JBI submission package uses `fig5_pipeline.pdf` with SHA-256:

`7b4ffcc1e922261cf920c6842e0834d7e454cd8a214bf0909714f388c9a28a01`

The corresponding final PNG generated during submission hardening has SHA-256:

`2b18880f216c32aac4cb42f8de9dc346fb81a766783b6f4a29ea9e7c55a361c8`

The v5 repair changed layout geometry only: it removed label collisions/overflow and rerouted the audit-feedback arrow while preserving the frozen quantitative callouts and interpretation.

## Public-release source

`code/generate_pipeline_fig.py` has been synchronized to the v5 collision-free layout and reads all quantitative callouts from the frozen JSON exports in `results/`.

A clean local run of the synchronized generator produced a PDF that, when rasterized at 150 dpi, was pixel-for-pixel identical to the canonical manuscript PDF. The PDF file hashes are not expected to be identical because Matplotlib embeds generation metadata such as creation time.

The stale pre-hardening `analysis/master_pipeline.pdf` and `analysis/master_pipeline.png` carried by `v1.0.0` are therefore intentionally absent from the `v1.0.1` candidate tree. They remain preserved in the immutable historical `v1.0.0` tag.

The other manuscript-facing result figures were also cross-checked against the audited v4 release. They are either exact binary matches or rasterize pixel-for-pixel identically; those differences are therefore non-scientific rendering metadata and do not require replacement in the patch.

To regenerate the synchronized figure locally:

```bash
python -m pip install -r requirements-core.txt
python code/generate_pipeline_fig.py
```

This writes:

- `analysis/master_pipeline.pdf`
- `analysis/master_pipeline.png`
- `figures/fig5_pipeline.pdf`
- `figures/fig5_pipeline.png`
- `figures/fig5_pipeline.svg`

## Scientific boundary

This synchronization does not change any frozen result JSON, threshold, candidate set, FDR estimate, cost estimate, or external-validation result. It only aligns the public figure-generation source with the final manuscript layout.
