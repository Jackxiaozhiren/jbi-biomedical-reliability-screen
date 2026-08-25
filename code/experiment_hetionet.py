"""Realized-vs-claimed FDR audit on the Hetionet drug-centric core subgraph.

The application experiment (P3): a practitioner screens predicted drug-target
and drug-disease links; the reliability layer must decide which to act on at a
controllable error rate. We build a LABELED candidate pool from the held-out
decision test edges:

  * per query (h, r) the pool holds 1 true candidate (held-out tail, label 1)
    and J verified-absent candidates (tails with no known edge of (h,r) in the
    whole subgraph, label 0);
  * shared-reference p-values: sample K reference tails per query (excluding
    all known positives and the candidate tails) and set
    p(c) = (1 + #{k : score(r_k) >= score(c)}) / (K+1).

Then we audit four screening rules on the labeled pool:
  * nominal BH at alpha (claim: FDR <= alpha) -> realized FDR;
  * global calibrated cutoff (fit on a calibration half of the pool to a
    target realized FDR) -> realized on the eval half;
  * relation/"task"-conditional calibrated cutoffs (one per decision relation,
    then pooled) -> realized;
  * raw-score top-k at matched coverage (baseline).

Usage: python code/experiment_hetionet.py [--j 9] [--k 500] [--models TransE]
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
import env_patch  # noqa: E402,F401

from evaluation import top_k_curve
from stats_core import (
    benjamini_hochberg, calibrate_screening_threshold, realized_error_metrics,
    screening_decisions, storey_pi0,
)

DECISION_RELATIONS = ["CbG", "CuG", "CdG", "CtD", "CpD"]


def load_core_dataset(seed: int = 42):
    from train_hetionet_core import build_core_dataset
    return build_core_dataset(seed)


def build_known_positives(dataset):
    known = {}
    for tf in (dataset.training, dataset.validation, dataset.testing):
        for h, r, t in tf.mapped_triples.tolist():
            known.setdefault((int(h), int(r)), set()).add(int(t))
    return known


def build_pool_and_pvalues(model, dataset, split_tf, known, j, k, rng):
    """Labeled pool + shared-reference p-values for a triples factory split."""
    num_entities = dataset.training.num_entities
    records = []
    t_q = time.time()
    rows = split_tf.mapped_triples.tolist()
    for qi, (h, r, t_true) in enumerate(rows):
        h, r, t_true = int(h), int(r), int(t_true)
        excl = known.get((h, r), set()) | {t_true}
        excl_arr = np.asarray([e for e in excl if e < num_entities], dtype=np.int64)
        universe = np.ones(num_entities, dtype=bool)
        universe[excl_arr] = False
        cand_idx = np.flatnonzero(universe)
        if cand_idx.size < j:
            raise RuntimeError(f"({h},{r}): only {cand_idx.size} valid candidates")
        negs = rng.choice(cand_idx, size=j, replace=False)
        cand_tails = np.concatenate([np.asarray([t_true]), negs])
        cand_labels = np.concatenate([np.ones(1, bool), np.zeros(j, bool)])

        ref_excl = np.ones(num_entities, dtype=bool)
        ref_excl[np.asarray([e for e in excl if e < num_entities], dtype=np.int64)] = False
        ref_excl[cand_tails] = False
        ref_pool = np.flatnonzero(ref_excl)
        if ref_pool.size < k:
            ref = ref_pool
        else:
            ref = rng.choice(ref_pool, size=k, replace=False)

        hrt = np.stack([
            np.full(len(cand_tails) + len(ref), h),
            np.full(len(cand_tails) + len(ref), r),
            np.concatenate([cand_tails, ref]),
        ], axis=1)
        with torch.no_grad():
            scores = model.score_hrt(torch.tensor(hrt)).cpu().numpy().reshape(-1)
        cand_scores = scores[: len(cand_tails)]
        ref_scores = scores[len(cand_tails):]
        denom = len(ref) + 1
        ps = (1.0 + np.sum(ref_scores[None, :] >= cand_scores[:, None], axis=1)) / denom
        for tail, label, sc, p in zip(cand_tails, cand_labels, cand_scores, ps):
            records.append({
                "head": h, "relation": r, "tail": int(tail),
                "label": bool(label), "score": float(sc), "p": float(p),
            })
        if (qi + 1) % 2000 == 0:
            print(f"  {qi+1}/{len(rows)} queries in {time.time()-t_q:.0f}s", flush=True)
    return records


def run_audit(model, dataset, j, k, rng, seed, sample_seed=None,
              model_name="RotatE"):
    tf_test = dataset.testing
    print(f"\n[Hetionet core] building labeled pool on {tf_test.num_triples:,} test decision edges "
          f"(J={j}, K={k})...", flush=True)
    records = build_pool_and_pvalues(model, dataset, tf_test, dataset_known(dataset), j, k, rng)

    p = np.array([r["p"] for r in records])
    scores = np.array([r["score"] for r in records])
    labels = np.array([r["label"] for r in records])
    # relation id -> name for per-relation splits (id -> label, NOT inverted)
    rel_names = np.array([dataset.training.relation_id_to_label[r_["relation"]]
                          for r_ in records])

    n = p.size
    idx = np.arange(n)
    rng.shuffle(idx)
    half = n // 2
    cal_p, cal_l = p[idx[:half]], labels[idx[:half]]
    ev_p, ev_l = p[idx[half:]], labels[idx[half:]]
    ev_rel = rel_names[idx[half:]]
    cal_scores, cal_labels = scores[idx[:half]], labels[idx[:half]]
    ev_scores, ev_labels = scores[idx[half:]], labels[idx[half:]]

    out = {"pool_size": n, "base_rate": float(labels.mean()),
           "j": j, "k": k, "pi0": storey_pi0(p)}

    # ---- 1. nominal BH ----
    bh = benjamini_hochberg(p, alpha=0.05)
    m = realized_error_metrics(bh.rejected, labels)
    out["nominal_bh_0.05"] = {
        "claimed_fdr": 0.05, **m, "cutoff": bh.threshold,
    }

    # ---- 2. global calibrated ----
    out["calibrated"] = []
    for target in (0.05, 0.10, 0.20):
        th = calibrate_screening_threshold(cal_p, cal_l, target)
        dec = screening_decisions(ev_p, th.cutoff)
        mm = realized_error_metrics(dec, ev_l)
        out["calibrated"].append({
            "claimed_fdr": target, "calibrated_cutoff": th.cutoff, **mm,
        })

    # ---- 3. relation-conditional calibrated (fit + eval per relation) ----
    out["calibrated_per_relation"] = {}
    dec_per_rel = np.zeros(n, dtype=bool)
    for rn in DECISION_RELATIONS:
        sel_cal = rel_names[idx[:half]] == rn
        sel_ev = ev_rel == rn
        if sel_cal.sum() < 50 or sel_ev.sum() == 0:
            continue
        th = calibrate_screening_threshold(cal_p[sel_cal], cal_l[sel_cal], 0.10)
        local = screening_decisions(ev_p[sel_ev], th.cutoff)
        dec_per_rel[idx[half:][sel_ev]] = local
        mm = realized_error_metrics(local, ev_l[sel_ev])
        out["calibrated_per_relation"][rn] = {
            "claimed_fdr": 0.10, "n_rel_pool": int(sel_ev.sum()),
            "cutoff": th.cutoff, **mm,
        }
    pooled = realized_error_metrics(dec_per_rel[idx[half:]], ev_l)
    out["calibrated_per_relation"]["_pooled_all_"] = pooled

    # ---- 4. top-k baseline at matched coverage ----
    coverages = [0.05, 0.1, 0.2]
    out["topk"] = [r.__dict__ for r in top_k_curve(scores, labels, coverages)]

    # ---- 5. calibrator baseline (platt + isotonic) ----
    from baseline_calibrator import calibrator_curve
    out["calibrator"] = {
        "platt": calibrator_curve(cal_scores, cal_labels, ev_scores, ev_labels, "platt"),
        "isotonic": calibrator_curve(cal_scores, cal_labels, ev_scores, ev_labels, "isotonic"),
    }

    # ---- 6. split-conformal baseline ----
    from baseline_conformal import conformal_curve
    out["conformal"] = conformal_curve(cal_scores, cal_labels, ev_scores, ev_labels)

    # ---- 7. ReliK baseline (sampled, WWW 2024) ----
    from baseline_relik import build_sets, relik_curve
    hr_tails, rt_heads = build_sets(dataset.training, dataset.validation,
                                    dataset.testing)
    h_arr = np.array([r_["head"] for r_ in records])
    r_arr = np.array([r_["relation"] for r_ in records])
    t_arr = np.array([r_["tail"] for r_ in records])
    out["relik"] = relik_curve(
        h_arr[idx[:half]], r_arr[idx[:half]], t_arr[idx[:half]], cal_labels,
        h_arr[idx[half:]], r_arr[idx[half:]], t_arr[idx[half:]], ev_labels,
        model, hr_tails, rt_heads, dataset.training.num_entities,
        dataset.training.num_relations, m=30, seed=42,
    )

    # ---- 8. cost-aware screening + decision-cost comparison ----
    from stats_core import calibrate_threshold_by_cost, expected_decision_cost
    ratios = [(1, 1), (5, 1), (1, 5)]  # (c_fp, c_fn)
    out["cost_aware"] = []
    for cfp, cfn in ratios:
        th = calibrate_threshold_by_cost(cal_p, cal_l, cfp, cfn)
        dec = screening_decisions(ev_p, th)
        mm = realized_error_metrics(dec, ev_l)
        out["cost_aware"].append({
            "c_fp": cfp, "c_fn": cfn, "threshold": th,
            "n_screened_in": mm["n_screened_in"], "coverage": mm["coverage"],
            "realized_fdr": mm["realized_fdr"], "precision": mm["precision"],
            "recall": mm["recall"],
            "expected_cost": expected_decision_cost(dec, ev_l, cfp, cfn),
        })

    def _cost_block(kept, label):
        return [{"method": label, "c_fp": cfp, "c_fn": cfn,
                 "expected_cost": expected_decision_cost(kept, ev_l, cfp, cfn)}
                for cfp, cfn in ratios]

    out["decision_cost"] = {}
    bh_ev = benjamini_hochberg(ev_p, 0.05)
    out["decision_cost"]["nominal_bh_0.05"] = _cost_block(bh_ev.rejected,
                                                          "nominal_bh_0.05")
    th10 = calibrate_screening_threshold(cal_p, cal_l, 0.10)
    cal10_mask = screening_decisions(ev_p, th10.cutoff)
    out["decision_cost"]["calibrated_0.10"] = _cost_block(cal10_mask,
                                                          "calibrated_0.10")
    nkeep = int(np.sum(cal10_mask))
    order = np.argsort(-ev_scores, kind="stable")
    tk_mask = np.zeros(ev_p.size, dtype=bool)
    tk_mask[order[:nkeep]] = True
    out["decision_cost"]["topk_matched"] = _cost_block(tk_mask, "topk_matched")

    out_path = ROOT / "results" / (
        f"hetionet_audit_J{j}_K{k}"
        f"{'_sample'+str(sample_seed) if sample_seed is not None else ''}"
        f"{'_'+model_name.lower() if model_name != 'RotatE' else ''}.json"
    )
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    return out


def dataset_known(dataset):
    return build_known_positives(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j", type=int, default=9)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--model", default="TransE")
    ap.add_argument("--seed", type=int, default=42,
                    help="split seed (dataset 70/10/20 permutation; keep fixed across sampling-sensitivity runs)")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="sampling seed for negatives/refs/cal-eval shuffle; defaults to --seed")
    args = ap.parse_args()

    torch.set_num_threads(min(2, torch.get_num_threads()))  # gentle on concurrent training
    ds = load_core_dataset(args.seed)
    model = torch.load(ROOT / "models" / f"hetionet_core_{args.model}.pt",
                       map_location="cpu", weights_only=False)
    model.eval()
    rng = np.random.default_rng(args.sample_seed if args.sample_seed is not None else args.seed)
    run_audit(model, ds, args.j, args.k, rng, args.seed, args.sample_seed,
              model_name=args.model)


if __name__ == "__main__":
    main()