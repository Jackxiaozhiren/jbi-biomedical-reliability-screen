"""Screening-vs-baselines evaluation harness (dataset-agnostic).

Compares candidate selection methods at matched operating points:

* raw-score top-k (select the q fraction of highest-scored predictions)
* score-threshold abstention (select predictions above a score quantile)
* nominal BH screening (screen in p <= BH cutoff at alpha)
* calibrated screening (cutoff tuned on a calibration split to target a
  realized FDR, applied to an evaluation split)

Every method is scored at the same set of coverage levels q (fraction of
candidate predictions acted on), and we report realized precision, realized
FDR, recall, and (for the error-control methods) the gap between claimed and
realized error. Matching on coverage makes the comparison fair: it answers
"for the same action budget, which screen finds more true links / fewer false
ones".

All functions are pure numpy and label-aware: labels[i] = 1 means the
candidate link is a verified true link, 0 means a verified-absent (false) link.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stats_core import realized_error_metrics


def _select_top_q(scores: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask selecting the top-q fraction by descending score."""
    n = scores.shape[0]
    k = int(np.floor(q * n))
    if k <= 0:
        return np.zeros(n, dtype=bool)
    k = min(k, n)
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


@dataclass(frozen=True)
class MethodRow:
    method: str
    coverage: float
    n_screened_in: int
    realized_fdr: float
    precision: float
    recall: float
    extra: dict | None = None


def top_k_curve(scores: np.ndarray, labels: np.ndarray, coverages) -> list[MethodRow]:
    rows = []
    for q in coverages:
        dec = _select_top_q(scores, q)
        m = realized_error_metrics(dec, labels)
        rows.append(
            MethodRow("top-k", q, m["n_screened_in"], m["realized_fdr"],
                      m["precision"], m["recall"])
        )
    return rows


def threshold_curve(scores: np.ndarray, labels: np.ndarray, coverages) -> list[MethodRow]:
    """Score-quantile abstention: screen in the top q by score (same as top-k),
    kept separate so the paper can label the two distinctly."""
    return top_k_curve(scores, labels, coverages)


def nominal_bh_curve(
    p_values: np.ndarray,
    labels: np.ndarray,
    alphas=(0.01, 0.05, 0.10, 0.20),
) -> list[MethodRow]:
    from stats_core import benjamini_hochberg

    rows = []
    for alpha in alphas:
        rej, thr = benjamini_hochberg(p_values, alpha=alpha)
        m = realized_error_metrics(rej, labels)
        rows.append(
            MethodRow("BH-nominal", m["coverage"], m["n_screened_in"],
                      m["realized_fdr"], m["precision"], m["recall"],
                      {"alpha": alpha, "cutoff": thr})
        )
    return rows


def calibrated_curve(
    cal_p_values: np.ndarray,
    cal_labels: np.ndarray,
    eval_p_values: np.ndarray,
    eval_labels: np.ndarray,
    targets=(0.01, 0.05, 0.10, 0.20),
) -> list[MethodRow]:
    """Calibrate a cutoff on a calibration split, evaluate on the eval split.

    The claimed error is the target realized FDR; the realized error on the
    eval split is what the paper audits (claimed vs realized gap).
    """
    from stats_core import calibrate_screening_threshold, screening_decisions

    rows = []
    for target in targets:
        th = calibrate_screening_threshold(cal_p_values, cal_labels, target)
        dec = screening_decisions(eval_p_values, th.cutoff)
        m = realized_error_metrics(dec, eval_labels)
        rows.append(
            MethodRow("calibrated", m["coverage"], m["n_screened_in"],
                      m["realized_fdr"], m["precision"], m["recall"],
                      {"claimed_fdr": target, "calibrated_cutoff": th.cutoff,
                       "realized_fdr": m["realized_fdr"]})
        )
    return rows


def summarize_comparison(rows: list[MethodRow]) -> dict:
    return {
        r.method: {
            "coverage": r.coverage,
            "realized_fdr": r.realized_fdr,
            "precision": r.precision,
            "recall": r.recall,
            **(r.extra or {}),
        }
        for r in rows
    }
