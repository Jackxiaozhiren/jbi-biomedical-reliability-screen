"""Baseline: ReliK reliability scoring (Egger et al., WWW 2024), sampled.

Faithful adaptation of Eq. (1) of ReliK for a candidate prediction (h, r, t):

    rank_H = |{x in N-(h) : s(x) > s(h,r,t)}| + 1   (same-head negatives)
    rank_T = |{x in N-(t) : s(x) > s(h,r,t)}| + 1   (same-tail negatives)
    ReliK  = 0.5 * (1/rank_H + 1/rank_T)

Because computing the full negative neighborhood is O(|E||R|), we use the
paper's sampling estimator (ReliKSmp): draw M same-head and M same-tail
negative triples, all verified absent from the KG, and estimate the ranks.
Higher ReliK = more reliable. The screen keeps candidates above a calibrated
ReliK threshold (fit on validation to a target realized FDR).

This is a genuinely different reliability notion from our rank-based p-value:
it pools over ALL relations sharing the head (and ALL heads sharing the tail),
not over the per-query (h, r) candidate set.
"""
from __future__ import annotations

import numpy as np
import torch

from stats_core import realized_error_metrics


def build_sets(training, validation, testing):
    """Known triples as sets for absent-negative checks."""
    hr_tails = {}  # (h, r) -> set(tails)
    rt_heads = {}  # (r, t) -> set(heads)
    for tf in (training, validation, testing):
        for h, r, t in tf.mapped_triples.tolist():
            h, r, t = int(h), int(r), int(t)
            hr_tails.setdefault((h, r), set()).add(t)
            rt_heads.setdefault((r, t), set()).add(h)
    return hr_tails, rt_heads


def _sample_absent(hr_tails, rt_heads, num_entities, num_relations, m,
                   h, r, t, rng, tries=8):
    """Sample m same-head (h,r',t') and m same-tail (h',r',t) absent triples."""
    heads_h = np.zeros((m, 3), dtype=np.int64)
    tails_t = np.zeros((m, 3), dtype=np.int64)
    for i in range(m):
        for _ in range(tries):
            rp = int(rng.integers(num_relations))
            tp = int(rng.integers(num_entities))
            if tp not in hr_tails.get((h, rp), set()):
                heads_h[i] = (h, rp, tp); break
        else:
            heads_h[i] = (h, int(rng.integers(num_relations)),
                          int(rng.integers(num_entities)))
        for _ in range(tries):
            hp = int(rng.integers(num_entities))
            rp = int(rng.integers(num_relations))
            if hp not in rt_heads.get((rp, t), set()):
                tails_t[i] = (hp, rp, t); break
        else:
            tails_t[i] = (int(rng.integers(num_entities)),
                          int(rng.integers(num_relations)), t)
    return heads_h, tails_t


def relik_scores(model, h, r, t, hr_tails, rt_heads, num_entities,
                 num_relations, m=30, rng=None):
    """Vectorized ReliK scores (Eq. 1) for arrays of candidate predictions."""
    h = np.asarray(h, dtype=np.int64)
    r = np.asarray(r, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    n = h.size
    if rng is None:
        rng = np.random.default_rng(0)
    rank_h = np.ones(n, dtype=np.float64)
    rank_t = np.ones(n, dtype=np.float64)
    BATCH = 512
    with torch.no_grad():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            hh, tt = [], []
            for i in range(s, e):
                hh_i, tt_i = _sample_absent(hr_tails, rt_heads, num_entities,
                                            num_relations, m, int(h[i]),
                                            int(r[i]), int(t[i]), rng)
                hh.append(hh_i); tt.append(tt_i)
            hh = np.concatenate(hh)  # (slice*m, 3)
            tt = np.concatenate(tt)
            item_s = model.score_hrt(torch.tensor(
                np.stack([h[s:e], r[s:e], t[s:e]], axis=1))).cpu().numpy().ravel()
            head_s = model.score_hrt(torch.tensor(hh)).cpu().numpy().reshape(e - s, m)
            tail_s = model.score_hrt(torch.tensor(tt)).cpu().numpy().reshape(e - s, m)
            rank_h[s:e] = np.sum(head_s > item_s[:, None], axis=1) + 1.0
            rank_t[s:e] = np.sum(tail_s > item_s[:, None], axis=1) + 1.0
    return 0.5 * (1.0 / rank_h + 1.0 / rank_t)


def relik_curve(cal_h, cal_r, cal_t, cal_labels, ev_h, ev_r, ev_t, ev_labels,
                model, hr_tails, rt_heads, num_entities, num_relations,
                m=30, targets=(0.20, 0.40, 0.60, 0.80), seed=0):
    """ReliK screening at several reliability thresholds.

    Calibration scores are used to choose thresholds that hit a target keep
    fraction; the audit reports realized error on the eval pool at each
    threshold (threshold = quantile of calibration ReliK scores).
    """
    rng = np.random.default_rng(seed)
    cal_s = relik_scores(model, cal_h, cal_r, cal_t, hr_tails, rt_heads,
                         num_entities, num_relations, m, rng)
    ev_s = relik_scores(model, ev_h, ev_r, ev_t, hr_tails, rt_heads,
                        num_entities, num_relations, m, rng)
    qs = np.quantile(cal_s, [1.0 - x for x in targets])
    rows = []
    for q in qs:
        dec = ev_s >= q
        mm = realized_error_metrics(dec, np.asarray(ev_labels, dtype=bool))
        rows.append({
            "method": "relik", "threshold": float(q),
            "coverage": mm["coverage"], "n_screened_in": mm["n_screened_in"],
            "realized_fdr": mm["realized_fdr"], "precision": mm["precision"],
            "recall": mm["recall"],
        })
    return rows
