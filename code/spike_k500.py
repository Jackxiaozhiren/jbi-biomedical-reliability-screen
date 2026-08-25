"""P2 feasibility spike: WN18RR at K=500 with the corrected protocol.

Verifies the pipeline mechanics that the new paper's experiments depend on:
1. filtered tail-only candidate sampling with FULL known-positive exclusion
   (train + validation + test) and observed-tail exclusion;
2. batched scoring throughput at K=500 on CPU;
3. BH no longer collapsing to a Hits@{1-3} count at the finer grid;
4. per-relation p-value / pi0 diagnostics remain computable.

By default uses the converged WN18RR TransE model (falling back to the old
cached 200-epoch TransE), overridable via --model-path. The JSON output records
the resolved model identity so results are attributable.

Usage: python code/spike_k500.py [--k 500] [--limit 400]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OLD_MODEL = ROOT.parent / "experiments/models/WN18RR_TransE/trained_model.pkl"
NEW_MODEL = ROOT / "models" / "WN18RR_TransE.pt"
MODEL_OVERRIDE = None  # set by --model-path; takes precedence over everything

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import env_patch  # noqa: E402,F401  (torch.load weights_only shim, import first)

from stats_core import empirical_p_values, benjamini_hochberg, storey_pi0


def _resolve_model():
    if MODEL_OVERRIDE is not None:
        return MODEL_OVERRIDE
    return NEW_MODEL if NEW_MODEL.exists() else OLD_MODEL


def build_known_positives(training, validation, testing):
    """Full known-positive exclusion map: train + validation + test tails per (h, r)."""
    known = {}
    for tf in (training, validation, testing):
        for h, r, t in tf.mapped_triples.tolist():
            known.setdefault((h, r), set()).add(t)
    return known


def run_spike(k: int, limit: int | None):
    from pykeen.datasets import WN18RR
    from pykeen.triples import TriplesFactory

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    t0 = time.time()
    model_path = _resolve_model()
    print("Loading model and dataset...")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    d = WN18RR()
    print(f"  model={type(model).__name__} entities={d.num_entities} "
          f"relations={d.num_relations} in {time.time()-t0:.1f}s")

    testing = d.testing.mapped_triples
    total = testing.shape[0]
    if limit is not None:
        testing = testing[:limit]

    known = build_known_positives(d.training, d.validation, d.testing)
    print(f"  known-positive keys: {len(known)}; testing triples to process: {testing.shape[0]}")

    # ---- batched candidate sampling + scoring ----
    BATCH = 100
    all_p = []
    all_obs = []
    t1 = time.time()
    n_scored = 0
    for start in range(0, testing.shape[0], BATCH):
        batch = testing[start : start + BATCH]
        obs_scores = []
        cand_scores = []
        for h, r, t_true in batch.tolist():
            excl = known.get((h, r), set())
            excl = excl | {t_true}  # observed tail also excluded
            pool = [e for e in range(d.num_entities) if e not in excl]
            if len(pool) < k:
                raise RuntimeError(f"triple ({h},{r}) has only {len(pool)} valid candidates")
            neg = rng.choice(pool, size=k, replace=False)
            hrt = np.stack([np.full(k, h), np.full(k, r), neg], axis=1)
            with torch.no_grad():
                s_true = model.score_hrt(torch.tensor([[h, r, t_true]])).item()
                s_neg = model.score_hrt(torch.tensor(hrt)).numpy().reshape(-1)
            obs_scores.append(s_true)
            cand_scores.append(s_neg)
            n_scored += k + 1
        cand_mat = np.array(cand_scores)
        p = empirical_p_values(np.array(obs_scores), cand_mat)
        all_p.append(p)
        all_obs.extend(obs_scores)
        if (start // BATCH + 1) % 5 == 0:
            elapsed = time.time() - t1
            print(f"  processed {start+BATCH}/{testing.shape[0]} "
                  f"({n_scored/elapsed:.0f} scores/s)")

    p_all = np.concatenate(all_p)
    m = p_all.shape[0]
    floor = 1.0 / (k + 1)
    print(f"\nScoring done: {n_scored:,} scores in {time.time()-t1:.1f}s "
          f"({n_scored/(time.time()-t1):.0f}/s)")

    # ---- diagnostics ----
    bh = benjamini_hochberg(p_all, alpha=0.05)
    rej, thr = bh.rejected, bh.threshold
    pi0 = storey_pi0(p_all)
    n_floor = int(np.sum(p_all <= floor * 1.0001))
    print("\n=== WN18RR spike, K=%d (model: %s) ===" % (k, model_path))
    print(f"  m={m}, floor p={floor:.5f}, {n_floor} at floor ({n_floor/m:.1%})")
    print(f"  pi0(0.5)={pi0:.4f}")
    print(f"  BH alpha=0.05: threshold={thr:.5f}, rejected={rej.sum()} "
          f"({rej.mean():.1%})")
    # rank composition of rejected set
    grid = (np.ceil(p_all * (k + 1)).astype(int)).clip(1, k + 1)
    rej_ranks = grid[rej]
    if rej_ranks.size:
        from collections import Counter
        top = Counter(rej_ranks.tolist()).most_common(5)
        print(f"  rejected-set rank composition (top): {top}")
    print(f"  mean p={p_all.mean():.4f}, median p={np.median(p_all):.4f}")

    out = {
        "k": k, "m": m, "floor_p": floor, "n_at_floor": int(n_floor),
        "pi0": pi0, "bh_threshold": thr, "n_rejected": int(rej.sum()),
        "rejected_rate": float(rej.mean()), "mean_p": float(p_all.mean()),
        "scores_per_sec": n_scored / (time.time() - t1),
        "model": type(model).__name__, "model_path": str(model_path),
    }
    out_path = ROOT / "results" / f"spike_k{k}_transE.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-path", default=None,
                        help="path to trained model; defaults to the converged WN18RR TransE, else the old cached one")
    args = parser.parse_args()
    if args.model_path is not None:
        MODEL_OVERRIDE = Path(args.model_path)  # explicit override beats defaults
    torch.set_num_threads(min(2, torch.get_num_threads()))  # gentle on concurrent training
    run_spike(args.k, args.limit)
