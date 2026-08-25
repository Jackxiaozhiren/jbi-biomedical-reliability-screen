"""Baseline: score calibration (KGE-Calibrator style) screening.

Fits a calibrated probability P(true | score) on a calibration pool of
(score, label) pairs -- Platt scaling (logistic) or isotonic regression --
then screens evaluation candidates by a calibrated-probability threshold.
This is the natural competitor to the p-value screen: it uses the SAME labels,
via score calibration instead of rank-based significance.

API mirrors the other baselines so the experiment harness can call it with
(calibration scores+labels, eval scores+labels, target) and get MethodRow-like
metrics.
"""
from __future__ import annotations

import numpy as np

from stats_core import realized_error_metrics, screening_decisions


def _to_range01(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-12:
        return np.full(scores.shape, 0.5)
    return (scores - lo) / (hi - lo)


def fit_calibrator(cal_scores, cal_labels, method="platt"):
    """Return a callable score -> calibrated probability P(true|score)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    x = _to_range01(np.asarray(cal_scores, dtype=np.float64)).reshape(-1, 1)
    y = np.asarray(cal_labels, dtype=int)
    if method == "platt":
        lr = LogisticRegression(max_iter=2000)
        lr.fit(x, y)
        return lambda s: lr.predict_proba(_to_range01(s).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(np.asarray(cal_scores, dtype=np.float64), y)
        return lambda s: iso.predict(np.asarray(s, dtype=np.float64))
    raise ValueError(method)


def calibrator_curve(cal_scores, cal_labels, eval_scores, eval_labels,
                     method="platt", targets=(0.70, 0.80, 0.90, 0.95)):
    """Calibrated-probability screening at several probability thresholds.

    Returns MethodRow-like dicts with realized FDR / precision / recall on the
    eval pool at each threshold.
    """
    cal = fit_calibrator(cal_scores, cal_labels, method)
    probs = cal(np.asarray(eval_scores, dtype=np.float64))
    rows = []
    for tau in targets:
        dec = probs >= tau
        m = realized_error_metrics(dec, np.asarray(eval_labels, dtype=bool))
        rows.append({
            "method": f"calibrator-{method}", "threshold": tau, "coverage": m["coverage"],
            "n_screened_in": m["n_screened_in"], "realized_fdr": m["realized_fdr"],
            "precision": m["precision"], "recall": m["recall"],
        })
    return rows
