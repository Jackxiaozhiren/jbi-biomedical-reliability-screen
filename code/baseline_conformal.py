"""Baseline: split-conformal per-query screening (CondKGCP-style, simplified).

A conformal set over candidate tails per query: nonconformity score
s(h,r,t) = -score(h,r,t) (higher = less conforming). On a calibration pool the
(1-alpha) conformal quantile of the TRUE tails' nonconformity scores is
computed; on the evaluation pool a candidate tail is kept iff its
nonconformity is <= that quantile (i.e. its score is extreme enough).

Under split-conformal exchangeability this yields marginal coverage of the
true tail at level (1-alpha). The realized error among KEPT candidates on the
labeled eval pool is the audited quantity (does the coverage-guarantee screen
also control realized FDR?).
"""
from __future__ import annotations

import numpy as np

from stats_core import realized_error_metrics


def conformal_quantile(cal_true_scores: np.ndarray, alpha: float) -> float:
    """(1-alpha) split-conformal quantile on calibration true-tail scores.

    nonconformity = -score; the quantile is the k-th smallest value with
    k = ceil((n+1)*(1-alpha)) (standard split-conformal, gives >= 1-alpha
    marginal coverage under exchangeability).
    """
    n = int(cal_true_scores.size)
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    ncs = np.sort(-np.asarray(cal_true_scores, dtype=np.float64))  # ascending
    return float(ncs[k - 1])  # this is -score_threshold


def conformal_curve(cal_scores, cal_labels, eval_scores, eval_labels,
                    alphas=(0.05, 0.10, 0.20)):
    """Split-conformal screening at several coverage levels on the eval pool.

    Returns MethodRow-like dicts: kept = score >= -q_hat, with realized FDR /
    precision / recall and the empirical coverage of true tails on eval.
    """
    cal_true = np.asarray(cal_scores, dtype=np.float64)[
        np.asarray(cal_labels, dtype=bool)]
    ev_scores = np.asarray(eval_scores, dtype=np.float64)
    ev_labels = np.asarray(eval_labels, dtype=bool)
    rows = []
    for alpha in alphas:
        q_hat = conformal_quantile(cal_true, alpha)
        dec = ev_scores >= -q_hat
        m = realized_error_metrics(dec, ev_labels)
        true_covered = float(np.mean(dec[ev_labels])) if ev_labels.any() else 0.0
        rows.append({
            "method": "conformal-split", "coverage_level": 1.0 - alpha,
            "score_threshold": -q_hat, "coverage": m["coverage"],
            "n_screened_in": m["n_screened_in"], "realized_fdr": m["realized_fdr"],
            "precision": m["precision"], "recall": m["recall"],
            "true_coverage": true_covered,
        })
    return rows
