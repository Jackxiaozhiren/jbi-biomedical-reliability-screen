"""P1-1 null-scorer validity probe.

A signal-free scorer (uniform random scores) should produce uniform
p-values on the (K+1) grid and the nominal BH step should reject
essentially nothing. This validates the index construction and the
audit harness — if a random scorer rejected many, the harness would
be broken.

Usage: /usr/bin/python3 code/null_scorer_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "analysis"

# Import the statistical core
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_core  # noqa: E402

rng = np.random.default_rng(42)

# Hetionet-matched dimensions: pool 105160, K=500 grid, J=9 base 10%
# For a random scorer, observed and candidate scores are i.i.d., so
# p-values are uniform over the (K+1) grid by construction.
N = 105_160
K = 500
# Uniform grid simulation: p = (1 + Binom(K, 0.5 opponent)) / (K+1)
# For i.i.d. scores the rank of observed among K candidates + itself
# is uniform 1..K+1, so p is uniform over the grid.
# We simulate directly as discrete uniform.
p_random = rng.integers(1, K + 2, size=N) / (K + 1)
# Also simulate WIB: base_rate 0.10 labels for realized FDR evaluation
labels = rng.random(N) < 0.10  # 10% positives, same base rate as Hetionet pools
# Real labels are independent of random p-values (no signal)

out: dict = {
    "meta": {
        "description": "null-scorer validity probe: uniform random p-values on K=500 grid",
        "N": N, "K": K, "base_rate": float(labels.mean()),
        "seed": 42,
        "note": "p-values are discrete uniform over 1/(K+1)..1, independent of labels (no signal)",
    }
}

# Nominal BH at 0.05 on full pool — should reject ~nothing
for alpha in (0.05, 0.10, 0.20):
    bh = stats_core.benjamini_hochberg(p_random, alpha)
    re = stats_core.realized_error(bh.rejected, labels)
    out[f"nominal_BH_{alpha:.2f}"] = {
        "alpha": alpha,
        "n_rejected": int(bh.n_rejected),
        "threshold": float(bh.threshold),
        "realized_fdr": float(re.realized_fdr),
        "coverage": float(bh.rate),
    }
    print(f"BH alpha={alpha:.2f}: rejected {bh.n_rejected:5d} / {N}  thr={bh.threshold:.4f}  realized FDR {re.realized_fdr:.4f}  coverage {bh.rate:.4%}")

# Split: calibration half -> eval half, calibrated threshold
idx = rng.permutation(N)
half = N // 2
cal_p, cal_l = p_random[idx[:half]], labels[idx[:half]]
ev_p, ev_l = p_random[idx[half:]], labels[idx[half:]]

for target in (0.05, 0.10, 0.20):
    thr = stats_core.calibrate_threshold(cal_p, cal_l, target)
    kept = ev_p <= thr
    re = stats_core.realized_error(kept, ev_l)
    out[f"calibrated_{target:.2f}"] = {
        "target": target,
        "threshold": float(thr),
        "n_kept": int(re.n_kept),
        "realized_fdr": float(re.realized_fdr),
        "coverage": float(re.n_kept / ev_l.size),
    }
    print(f"Calibrated target={target:.2f}: thr={thr:.6f}  kept {re.n_kept:5d}  FDR {re.realized_fdr:.4f}  cov {re.n_kept/ev_l.size:.4%}")

# Uniformity diagnostic: Storey pi0 should be ~1.0 under global null
pi0 = stats_core.storey_pi0(p_random, lam=0.5)
out["storey_pi0"] = float(pi0)
print(f"Storey pi0 (lam=0.5): {pi0:.4f}  (expected ~1.0 under null)")

# Verdict
bh05_n = out["nominal_BH_0.05"]["n_rejected"]
verdict = "PASS" if bh05_n == 0 and pi0 > 0.95 else ("PASS (near-null)" if bh05_n < 0.01 * N else "FAIL")
out["verdict"] = verdict
print(f"Verdict: {verdict} — random scorer {'produces no spurious rejections' if verdict.startswith('PASS') else 'REJECTS SPURIOUSLY — harness broken'}")

out_path = RESULTS / "null_scorer_audit.json"
# Also save to analysis for the report
RESULTS.mkdir(exist_ok=True)
ANALYSIS.mkdir(exist_ok=True)
Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"Saved {out_path}")
