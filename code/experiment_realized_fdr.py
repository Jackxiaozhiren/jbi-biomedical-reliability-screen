"""Realized-vs-claimed FDR experiment (the paper's core audit).

Given a trained KG embedding model and a test split, this constructs a labeled
candidate pool and measures whether the screening layer's *claimed* error rate
(BH alpha, or a calibrated target) matches the *realized* error on held-out
ground truth.

Pool construction (per test query (h, r)):
  * one true candidate: the held-out tail t_true, label = 1;
  * J false candidates: tails sampled uniformly from entities that are NOT
    known positives for (h, r) in train+valid+test and not the true tail,
    label = 0. With sparse KGs these sampled tails are absent from the full
    graph with overwhelming probability, so they are verified-absent links.

p-value construction (shared reference set):
  * sample K reference tails R per query, excluding all known positives and
    every candidate tail;
  * p(c) = (1 + #{r in R : s(h,r,r) >= s(h,r,c)}) / (K+1), the rank of the
    candidate's score among the K reference draws. The same R is used for all
    candidates of a query, so p-values are comparable within a query.

Screen outputs (on the whole pool, labels known for auditing):
  * nominal BH at alpha -> claimed FDR alpha, realized FDR measured;
  * calibrated cutoff (tuned on a calibration split of the pool to a target
    realized FDR) -> claimed target, realized on the eval split;
  * raw-score top-k at matched coverage.

Usage: python code/experiment_realized_fdr.py [--dataset WN18RR] [--model-cache PATH]
                                            [--j 4] [--k 200] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import env_patch  # noqa: E402,F401  (torch.load weights_only shim, import first)

from stats_core import (
    empirical_p_values, benjamini_hochberg, storey_pi0,
    screening_decisions, realized_error_metrics, calibrate_screening_threshold,
)
from evaluation import top_k_curve, nominal_bh_curve, calibrated_curve


def build_known_positives(dataset):
    known = {}
    for tf in (dataset.training, dataset.validation, dataset.testing):
        for h, r, t in tf.mapped_triples.tolist():
            known.setdefault((h, r), set()).add(t)
    return known


def build_pool_and_pvalues(model, dataset, testing, known, j, k, rng):
    """Return dict with candidates (head,rel,tail,label) and per-candidate p-values."""
    records = []
    for h, r, t_true in testing.tolist():
        excl = known.get((h, r), set()) | {t_true}
        # candidate set
        cand_tails = [t_true]
        cand_labels = [1]
        if j > 0:
            pool = [e for e in range(dataset.num_entities) if e not in excl]
            if len(pool) < j:
                raise RuntimeError(f"({h},{r}): only {len(pool)} non-positive entities")
            negs = rng.choice(pool, size=j, replace=False).tolist()
            cand_tails.extend(negs)
            cand_labels.extend([0] * j)
        # reference tails (exclude known positives AND candidates)
        ref_excl = excl | set(cand_tails)
        ref_pool = [e for e in range(dataset.num_entities) if e not in ref_excl]
        if len(ref_pool) < k:
            ref = np.array(ref_pool)
        else:
            ref = rng.choice(ref_pool, size=k, replace=False)
        # scores
        hrt = np.stack([
            np.full(len(cand_tails) + len(ref), h),
            np.full(len(cand_tails) + len(ref), r),
            np.concatenate([np.array(cand_tails), np.array(ref)]),
        ], axis=1)
        with torch.no_grad():
            scores = model.score_hrt(torch.tensor(hrt)).numpy().reshape(-1)
        cand_scores = scores[: len(cand_tails)]
        ref_scores = scores[len(cand_tails):]
        for tail, label, sc in zip(cand_tails, cand_labels, cand_scores):
            p = (1.0 + np.sum(ref_scores >= sc)) / (len(ref) + 1.0)
            records.append({
                "head": int(h), "relation": int(r), "tail": int(tail),
                "label": int(label), "score": float(sc), "p": float(p),
            })
    return records


def run_experiment(dataset_name, model_cache, j, k, limit):
    from pykeen.datasets import WN18RR, PrimeKG, FB15k237
    from pykeen.triples import TriplesFactory

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    t0 = time.time()
    model = torch.load(model_cache, map_location="cpu", weights_only=False)
    model.eval()
    if dataset_name == "WN18RR":
        d = WN18RR()
    elif dataset_name == "FB15k237":
        d = FB15k237()
    else:
        d = PrimeKG()
    testing = d.testing.mapped_triples
    if limit:
        testing = testing[:limit]
    print(f"[{dataset_name}] model={type(model).__name__} "
          f"entities={d.num_entities} test={testing.shape[0]} "
          f"(loaded {time.time()-t0:.1f}s)")

    known = build_known_positives(d)
    t1 = time.time()
    records = build_pool_and_pvalues(model, d, testing, known, j, k, rng)
    print(f"  pool built: {len(records)} candidates (J={j}, K={k}) in "
          f"{time.time()-t1:.1f}s")

    scores = np.array([r["score"] for r in records])
    p = np.array([r["p"] for r in records])
    labels = np.array([r["label"] for r in records])

    # ---- nominal BH ----
    bh = benjamini_hochberg(p, alpha=0.05)
    rej, thr = bh.rejected, bh.threshold
    m_nom = realized_error_metrics(rej, labels)
    print("\n=== NOMINAL BH (claimed FDR = 0.05) ===")
    print(f"  screened in {m_nom['n_screened_in']}/{len(labels)} ({m_nom['coverage']:.1%})")
    print(f"  REALIZED FDR = {m_nom['realized_fdr']:.4f}  (claimed 0.05)  "
          f"precision={m_nom['precision']:.4f} recall={m_nom['recall']:.4f}")

    # ---- calibrated (split pool into calibration / eval) ----
    n = len(p)
    half = n // 2
    idx = np.arange(n)
    rng.shuffle(idx)
    cal_p, cal_l = p[idx[:half]], labels[idx[:half]]
    ev_p, ev_l = p[idx[half:]], labels[idx[half:]]
    cal_rows = calibrated_curve(cal_p, cal_l, ev_p, ev_l, targets=[0.01, 0.05, 0.10])
    print("\n=== CALIBRATED (claimed = target realized FDR) ===")
    for r in cal_rows:
        extra = r.extra or {}
        print(f"  target {extra['claimed_fdr']:.2f}: realized on eval "
              f"FDR={r.realized_fdr:.4f} prec={r.precision:.4f} cov={r.coverage:.1%} "
              f"(cutoff p<={extra['calibrated_cutoff']:.4f})")

    # ---- top-k at matched coverage ----
    tk = top_k_curve(scores, labels, coverages=[0.05, 0.1, 0.2])
    print("\n=== RAW-SCORE TOP-K (reference baseline) ===")
    for r in tk:
        print(f"  coverage {r.coverage:.0%}: realized FDR={r.realized_fdr:.4f} "
              f"precision={r.precision:.4f} recall={r.recall:.4f}")

    pi0 = storey_pi0(p)
    print(f"\n  pi0(0.5)={pi0:.4f}, pool base rate={labels.mean():.3f}")
    out = {
        "dataset": dataset_name, "j": j, "k": k,
        "pool_size": int(n), "base_rate": float(labels.mean()), "pi0": pi0,
        "nominal_bh_0.05": {
            "claimed_fdr": 0.05, "n_screened": int(m_nom["n_screened_in"]),
            "coverage": m_nom["coverage"], "realized_fdr": m_nom["realized_fdr"],
            "precision": m_nom["precision"], "recall": m_nom["recall"],
            "cutoff": thr,
        },
        "calibrated": [
            {"claimed_fdr": r.extra["claimed_fdr"], "coverage": r.coverage,
             "realized_fdr": r.realized_fdr, "precision": r.precision,
             "recall": r.recall, "cutoff": r.extra["calibrated_cutoff"]}
            for r in cal_rows
        ],
        "topk": [
            {"coverage": r.coverage, "realized_fdr": r.realized_fdr,
             "precision": r.precision, "recall": r.recall}
            for r in tk
        ],
    }
    out_path = ROOT / "results" / f"realized_fdr_{dataset_name}_J{j}_K{k}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WN18RR")
    parser.add_argument("--model-cache", default=None,
                        help="path to trained model; defaults to models/<dataset>_<model>.pt")
    parser.add_argument("--j", type=int, default=9)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.model_cache is None:
        args.model_cache = ROOT / "models" / f"{args.dataset}_TransE.pt"
    torch.set_num_threads(min(2, torch.get_num_threads()))  # gentle on concurrent training
    run_experiment(args.dataset, args.model_cache, args.j, args.k, args.limit)
