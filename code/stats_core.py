"""Statistical core for the reliability-screening layer (P1).

Pure numpy -- no torch/pykeen dependency, so it stays testable in isolation
and can be reused for any exported score file.

Protocol summary (P1):
  * Per-triple empirical p-value (permutation-style, +1 correction):
        p_i = (1 + #{candidate scores >= observed_i}) / (K + 1)
    Callers must provide candidate scores already filtered for the protocol:
    tail-only, without replacement, excluding ALL known positives (train/val/
    test) and the observed tail. This module does not see the exclusions.
  * Nominal diagnostics: largest-k Benjamini-Hochberg and Storey's fixed-lambda
    pi_0. Reported as diagnostics only -- their FDR interpretation is audited
    empirically (claimed vs realized), never assumed.
  * Screening rule: a decision on each prediction (keep / withhold) from a
    p-value threshold. The threshold may come from nominal BH at alpha, or be
    *calibrated* on a labeled set to hit a target realized FDR / precision.
  * Realized-error evaluator: given ground-truth labels on the evaluated pool,
    compute realized FDR, precision, recall, and the claimed-vs-realized gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class BHResult:
    alpha: float
    rejected: np.ndarray  # bool mask, length m
    threshold: float  # largest p-value accepted (0.0 if none)
    n_rejected: int
    rate: float


@dataclass(frozen=True)
class RealizedError:
    n_pool: int
    n_kept: int
    n_kept_positive: int
    n_kept_negative: int
    realized_fdr: float  # FP/(FP+TP) among kept; 0.0 if nothing kept
    precision: float  # TP/kept
    recall: float  # TP/total positives in pool
    sensitivity: float
    specificity: float


def empirical_p_values(observed: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Permutation-style empirical p-values.

    Args:
        observed: (m,) scores of the true/observed tails.
        candidates: (m, K) scores of the sampled negative tails per triple.

    Returns:
        (m,) p-values in (0, 1] on a (K+1)-spaced grid.
    """
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    candidates = np.asarray(candidates, dtype=np.float64)
    if observed.shape[0] != candidates.shape[0]:
        raise ValueError("observed and candidates must have the same leading dim")
    K = candidates.shape[1]
    n_ge = np.sum(candidates >= observed[:, None], axis=1)
    return (n_ge + 1) / (K + 1)


def benjamini_hochberg(p: np.ndarray, alpha: float, m0: Optional[int] = None) -> BHResult:
    """Standard largest-k BH at level ``alpha`` (optionally adaptive with m0)."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    m = p.size
    eff_m = m0 if m0 is not None else m
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    # k* = max{k : p_(k) <= k*alpha/m}
    admissible = sorted_p <= (np.arange(1, m + 1) * alpha / eff_m)
    if not np.any(admissible):
        rejected = np.zeros(m, dtype=bool)
        threshold = 0.0
    else:
        k_star = int(np.max(np.nonzero(admissible)[0])) + 1
        threshold = float(sorted_p[k_star - 1])
        rejected = p <= threshold
    return BHResult(
        alpha=alpha, rejected=rejected, threshold=threshold,
        n_rejected=int(np.sum(rejected)), rate=float(np.mean(rejected)),
    )


def storey_pi0(p: np.ndarray, lam: float = 0.5) -> float:
    """Fixed-lambda Storey null-proportion estimate."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be in (0, 1)")
    return min(1.0, float(np.mean(p >= lam)) / (1.0 - lam))


def screen_by_pvalue(p: np.ndarray, threshold: float) -> np.ndarray:
    """Keep predictions whose p-value is at most ``threshold`` (abstain on rest)."""
    return np.asarray(p, dtype=np.float64) <= threshold


def realized_error(kept: np.ndarray, labels: np.ndarray) -> RealizedError:
    """Empirical error/precision on the labeled pool, restricted to kept items.

    ``labels``: 1 = true/positive, 0 = false/negative on the evaluated pool.
    ``kept``:   bool mask over the same pool (screened-in predictions).
    """
    kept = np.asarray(kept, dtype=bool).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if kept.size != labels.size:
        raise ValueError("kept and labels must be same length")
    tp = int(np.sum(kept & labels))
    fp = int(np.sum(kept & ~labels))
    n_pos = int(np.sum(labels))
    n_kept = int(np.sum(kept))
    realized_fdr = fp / n_kept if n_kept else 0.0
    precision = tp / n_kept if n_kept else 0.0
    recall = tp / n_pos if n_pos else 0.0
    n_neg = int(np.sum(~labels))
    specificity = (n_neg - fp) / n_neg if n_neg else 1.0
    return RealizedError(
        n_pool=labels.size, n_kept=n_kept,
        n_kept_positive=tp, n_kept_negative=fp,
        realized_fdr=realized_fdr, precision=precision, recall=recall,
        sensitivity=recall, specificity=specificity,
    )


def calibrate_threshold(
    p_cal: np.ndarray, labels_cal: np.ndarray, target: float,
    grid: Optional[np.ndarray] = None, metric: str = "fdr",
) -> float:
    """Pick a p-value threshold meeting a target realized metric on calibration data.

    Scans candidate thresholds (the sorted p-values) and returns the LOOSEST
    threshold (largest p-value cutoff) whose realized metric is at most ``target``
    (fdr) or at least ``target`` (precision), i.e. max{tau : metric(tau) ok},
    maximizing coverage subject to the constraint. If no positive-coverage
    threshold meets the target, returns 0.0 (keep nothing).
    """
    p_cal = np.asarray(p_cal, dtype=np.float64).reshape(-1)
    labels_cal = np.asarray(labels_cal, dtype=bool).reshape(-1)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0], np.sort(p_cal), [1.0]]))
    best = 0.0  # default: keep nothing (no operating point meets the target)
    found = False
    for th in grid:
        kept = screen_by_pvalue(p_cal, th)
        if not np.any(kept):
            continue  # degenerate: nothing kept has vacuous FDR 0; must not qualify
        r = realized_error(kept, labels_cal)
        met = r.realized_fdr if metric == "fdr" else r.precision
        if (metric == "fdr" and met <= target) or (metric == "precision" and met >= target):
            best = th  # keep the LOOSEST threshold that still meets the target
            found = True
    return best


def expected_decision_cost(
    kept: np.ndarray, labels: np.ndarray, c_fp: float = 1.0, c_fn: float = 1.0,
) -> float:
    """Expected decision cost of a screening rule on the labeled pool.

    c_fp: cost of acting on a false prediction (kept, negative).
    c_fn: cost of withholding a true prediction (withheld, positive).
    """
    kept = np.asarray(kept, dtype=bool).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    fp = int(np.sum(kept & ~labels))
    fn = int(np.sum(~kept & labels))
    n = labels.size
    return (c_fp * fp + c_fn * fn) / n


def calibrate_threshold_by_cost(
    p_cal: np.ndarray, labels_cal: np.ndarray, c_fp: float = 1.0,
    c_fn: float = 1.0,
) -> float:
    """Cost-aware p-value threshold minimizing expected decision cost on calibration."""
    p_cal = np.asarray(p_cal, dtype=np.float64).reshape(-1)
    labels_cal = np.asarray(labels_cal, dtype=bool).reshape(-1)
    grid = np.unique(np.concatenate([[0.0], np.sort(p_cal), [1.0]]))
    best_th, best_cost = 0.0, np.inf
    for th in grid:
        kept = screen_by_pvalue(p_cal, th)
        cost = expected_decision_cost(kept, labels_cal, c_fp, c_fn)
        if cost < best_cost:
            best_cost, best_th = cost, th
    return best_th


def claimed_vs_realized(
    p: np.ndarray, labels: np.ndarray, alpha: float = 0.05,
) -> Dict[str, float]:
    """Audit: BH's claimed FDR (alpha) against the realized FDR on labeled data."""
    bh = benjamini_hochberg(p, alpha)
    r = realized_error(bh.rejected, labels)
    return {
        "alpha": alpha,
        "n_rejected": bh.n_rejected,
        "realized_fdr": r.realized_fdr,
        "precision": r.precision,
        "gap": r.realized_fdr - alpha,
        "n_kept_positive": r.n_kept_positive,
        "n_kept_negative": r.n_kept_negative,
    }


# ---- compatibility wrappers (prior workspace API; used by evaluation.py /
#      experiment_realized_fdr.py / spike_k500.py). Thin, tested core above. ----


def screening_decisions(p: np.ndarray, cutoff: float) -> np.ndarray:
    """Boolean screening decision (keep) for p-value cutoff. Alias of screen_by_pvalue."""
    return screen_by_pvalue(p, cutoff)


def realized_error_metrics(mask: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Dict-style realized-error metrics (prior API)."""
    r = realized_error(mask, labels)
    return {
        "n_screened_in": r.n_kept,
        "coverage": r.n_kept / r.n_pool if r.n_pool else 0.0,
        "realized_fdr": r.realized_fdr,
        "precision": r.precision,
        "recall": r.recall,
        "n_kept_positive": r.n_kept_positive,
        "n_kept_negative": r.n_kept_negative,
    }


@dataclass(frozen=True)
class CalibratedCutoff:
    cutoff: float
    realized_fdr_on_cal: float
    precision_on_cal: float
    n_cal: int


def calibrate_screening_threshold(
    p_cal: np.ndarray, labels_cal: np.ndarray, target: float,
) -> CalibratedCutoff:
    """Calibrated p-value cutoff to a target realized FDR on calibration data.

    Returns a CalibratedCutoff exposing ``.cutoff`` (prior API).
    """
    th = calibrate_threshold(p_cal, labels_cal, target, metric="fdr")
    kept = screen_by_pvalue(p_cal, th)
    r = realized_error(kept, labels_cal)
    return CalibratedCutoff(
        cutoff=th, realized_fdr_on_cal=r.realized_fdr,
        precision_on_cal=r.precision, n_cal=int(np.sum(labels_cal)),
    )


def relation_stratified(
    p: np.ndarray, labels: np.ndarray, rel_ids: np.ndarray, alpha: float = 0.05,
) -> Dict[int, Dict[str, float]]:
    """Per-relation claimed-vs-realized audit (descriptive, not joint control)."""
    out: Dict[int, Dict[str, float]] = {}
    for rel in np.unique(rel_ids):
        sel = rel_ids == rel
        out[int(rel)] = claimed_vs_realized(p[sel], labels[sel], alpha)
    return out
