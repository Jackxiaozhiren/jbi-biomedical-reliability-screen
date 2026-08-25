"""Run P1 heavy experiments (Scheme B): write all into main text.

Variants (all on frozen models, no retraining; each is 5-10s CPU per Hetionet audit):
  P1-2: type-constrained reference sampling (tail type must match relation)
  P1-3: J/K grid (J=4,9,19; K=100,500; spike already covers K=1000)
  P1-4: per-query grouped calibration/evaluation split (instead of record-level)

Writes:
  results/hetionet_audit_type_constrained.json
  results/jk_grid/hetionet_J{N}_K{K}.json (and spike_J/K matrix)
  results/hetionet_audit_grouped_split.json

Usage: /usr/bin/python3 code/run_p1_heavy.py [--variants 2,3,4]
  Default: run all three (2,3,4). Each variant is independent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: F401

from train_hetionet_core import build_core_dataset

# Tail type allowed per decision relation
ALLOWED_TAIL_PREFIX = {
    "CbG": "Gene::",
    "CuG": "Gene::",
    "CdG": "Gene::",
    "CtD": "Disease::",
    "CpD": "Disease::",
}


def entity_type_index_map(dataset):
    """Return dict prefix->set(entity_id) and per-eid prefix."""
    m = {}
    for eid, label in dataset.training.entity_id_to_label.items():
        prefix = label.split("::")[0] + "::" if "::" in label else ""
        m.setdefault(prefix, set()).add(eid)
    # also return per-eid prefix for quick lookup
    eid_to_prefix = {eid: (label.split("::")[0] + "::" if "::" in label else "") for eid, label in dataset.training.entity_id_to_label.items()}
    return m, eid_to_prefix


def allowed_tails_for_relation(rel_name: str, type_map: dict) -> set[int]:
    prefix = ALLOWED_TAIL_PREFIX.get(rel_name)
    if prefix is None:
        return set()
    return type_map.get(prefix, set())


def build_pool_and_pvalues_constrained(
    model, dataset, split_tf, known, j, k, rng, type_map: dict,
    rel_id_to_name: dict,
):
    """Type-constrained pool: candidate + reference tails restricted to legal tail type."""
    import time
    num_entities = dataset.training.num_entities
    records = []
    rows = split_tf.mapped_triples.tolist()
    t_q = time.time()
    n_over_total = 0
    n_over_cand = 0
    n_over_ref = 0
    for qi, (h, r, t_true) in enumerate(rows):
        h, r, t_true = int(h), int(r), int(t_true)
        rn = rel_id_to_name[r]
        allowed = allowed_tails_for_relation(rn, type_map)
        if not allowed:
            raise RuntimeError(f"no allowed tails for {rn}")
        excl = known.get((h, r), set()) | {t_true}
        # Candidate pool: allowed tails minus known positives and true tail
        universe = allowed - excl
        if len(universe) < j:
            # Fall back to mixing allowed + other if not enough allowed (should not happen for Gene)
            # For Disease (134 total, many known), may be tight
            n_over_cand += 1
        cand_idx = np.array(list(universe), dtype=np.int64)
        if cand_idx.size < j:
            raise RuntimeError(f"({h},{rn}): only {cand_idx.size} type-constrained candidates (need {j})")
        negs = rng.choice(cand_idx, size=j, replace=False)
        cand_tails = np.concatenate([np.asarray([t_true]), negs])
        cand_labels = np.concatenate([np.ones(1, bool), np.zeros(j, bool)])

        # Reference pool: allowed tails minus known + minus candidates
        ref_excl = set(excl) | set(cand_tails.tolist())
        ref_pool = np.array(list(allowed - ref_excl), dtype=np.int64)
        if ref_pool.size < k:
            # Not enough type-constrained refs — note but still sample what we have
            n_over_ref += 1
            ref = ref_pool
        else:
            ref = rng.choice(ref_pool, size=k, replace=False)

        denom = len(ref) + 1
        hrt = np.stack([
            np.full(len(cand_tails) + len(ref), h),
            np.full(len(cand_tails) + len(ref), r),
            np.concatenate([cand_tails, ref]),
        ], axis=1)
        with torch.no_grad():
            scores = model.score_hrt(torch.tensor(hrt)).cpu().numpy().reshape(-1)
        cand_scores = scores[:len(cand_tails)]
        ref_scores = scores[len(cand_tails):]
        # If ref empty (edge case, Disease with large known set), set p=1.0 for all
        if ref.size == 0:
            ps = np.ones(len(cand_tails))
        else:
            ps = (1.0 + np.sum(ref_scores[None, :] >= cand_scores[:, None], axis=1)) / denom
        for tail, label, sc, p in zip(cand_tails, cand_labels, cand_scores, ps):
            records.append({"head": h, "relation": r, "tail": int(tail), "label": bool(label), "score": float(sc), "p": float(p)})
        if (qi + 1) % 2000 == 0:
            print(f"  {qi+1}/{len(rows)} queries in {time.time()-t_q:.0f}s  (type-constrained)", flush=True)
    if n_over_cand or n_over_ref:
        print(f"  [constrained] queries hitting fallback: cand_short={n_over_cand} ref_short={n_over_ref}", flush=True)
    # Record floor stats
    floor_p = 1.0 / (500 + 1)
    n_at_floor = sum(1 for r in records if abs(r["p"] - floor_p) < 1e-9)
    print(f"  floor {floor_p:.6f}: {n_at_floor}/{len(records)} ({n_at_floor/len(records):.1%})", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="2,3,4", help="comma-separated among 2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wanted = {int(x.strip()) for x in args.variants.split(",") if x.strip()}
    print(f"Variants: {sorted(wanted)}  seed={args.seed}")

    # Load frozen model once (Hetionet RotatE)
    model_path = ROOT / "models/hetionet_core_RotatE.pt"
    dataset = build_core_dataset(seed=args.seed)
    # Load stats helpers
    from stats_core import benjamini_hochberg, calibrate_screening_threshold, realized_error_metrics, screening_decisions, storey_pi0
    from evaluation import top_k_curve

    # Shared helpers for audit (same as experiment_hetionet.py but with pluggable pool builder)
    def run_audit_from_records(records, pool_name: str, j: int, k: int):
        """Run the full audit (nominal BH / calibrated / per-relation / top-k / cost) from prebuilt records."""
        from baseline_calibrator import calibrator_curve
        from baseline_conformal import conformal_curve
        from baseline_relik import build_sets, relik_curve
        from stats_core import calibrate_threshold_by_cost, expected_decision_cost
        import time as _t

        p = np.array([r["p"] for r in records])
        scores = np.array([r["score"] for r in records])
        labels = np.array([r["label"] for r in records])
        rel_names = np.array([dataset.training.relation_id_to_label[r["relation"]] for r in records])
        n = p.size
        rng2 = np.random.default_rng(args.seed)
        idx = np.arange(n); rng2.shuffle(idx)
        half = n // 2
        cal_p, cal_l = p[idx[:half]], labels[idx[:half]]
        ev_p, ev_l = p[idx[half:]], labels[idx[half:]]
        ev_rel = rel_names[idx[half:]]
        cal_scores, cal_labels = scores[idx[:half]], labels[idx[:half]]
        ev_scores, ev_labels = scores[idx[half:]], labels[idx[half:]]

        out = {"pool_size": n, "base_rate": float(labels.mean()), "j": j, "k": k, "pi0": float(storey_pi0(p)), "pool_name": pool_name}
        # 1. nominal BH
        bh = benjamini_hochberg(p, 0.05)
        out["nominal_bh_0.05"] = {"claimed_fdr": 0.05, **realized_error_metrics(bh.rejected, labels), "cutoff": bh.threshold}
        # 2. global calibrated
        out["calibrated"] = []
        for target in (0.05, 0.10, 0.20):
            th = calibrate_screening_threshold(cal_p, cal_l, target)
            dec = screening_decisions(ev_p, th.cutoff)
            out["calibrated"].append({"claimed_fdr": target, "calibrated_cutoff": th.cutoff, **realized_error_metrics(dec, ev_l)})
        # 3. per-relation
        out["calibrated_per_relation"] = {}
        dec_per_rel = np.zeros(n, dtype=bool)
        for rn in ["CbG", "CuG", "CdG", "CtD", "CpD"]:
            sel_cal = rel_names[idx[:half]] == rn
            sel_ev = ev_rel == rn
            if sel_cal.sum() < 50 or sel_ev.sum() == 0:
                continue
            th = calibrate_screening_threshold(cal_p[sel_cal], cal_l[sel_cal], 0.10)
            local = screening_decisions(ev_p[sel_ev], th.cutoff)
            dec_per_rel[idx[half:][sel_ev]] = local
            out["calibrated_per_relation"][rn] = {"claimed_fdr": 0.10, "n_rel_pool": int(sel_ev.sum()), "cutoff": th.cutoff, **realized_error_metrics(local, ev_l[sel_ev])}
        out["calibrated_per_relation"]["_pooled_all_"] = realized_error_metrics(dec_per_rel[idx[half:]], ev_l)
        # 4. top-k
        out["topk"] = [r.__dict__ for r in top_k_curve(scores, labels, [0.05, 0.1, 0.2])]
        # 5. calibrator / conformal / relik (best-effort, may be slow)
        try:
            out["calibrator"] = {"platt": calibrator_curve(cal_scores, cal_labels, ev_scores, ev_labels, "platt"), "isotonic": calibrator_curve(cal_scores, cal_labels, ev_scores, ev_labels, "isotonic")}
        except Exception as e:
            print(f"  calibrator skipped: {e}")
        try:
            out["conformal"] = conformal_curve(cal_scores, cal_labels, ev_scores, ev_labels)
        except Exception as e:
            print(f"  conformal skipped: {e}")
        # 6. relik (optional, slow)
        # Skip by default for heavy variants; caller can re-enable.
        # 7. cost-aware
        out["cost_aware"] = []
        for cfp, cfn in [(1, 1), (5, 1), (1, 5)]:
            th = calibrate_threshold_by_cost(cal_p, cal_l, cfp, cfn)
            dec = screening_decisions(ev_p, th)
            out["cost_aware"].append({"c_fp": cfp, "c_fn": cfn, "threshold": th, "n_screened_in": int(realized_error_metrics(dec, ev_l)["n_screened_in"]), "coverage": float(realized_error_metrics(dec, ev_l)["coverage"]), "realized_fdr": float(realized_error_metrics(dec, ev_l)["realized_fdr"]), "precision": float(realized_error_metrics(dec, ev_l)["precision"]), "recall": float(realized_error_metrics(dec, ev_l)["recall"]), "expected_cost": float(expected_decision_cost(dec, ev_l, cfp, cfn))})
        def _cost_block(kept):
            return [{"method": "block", "c_fp": cfp, "c_fn": cfn, "expected_cost": float(expected_decision_cost(kept, ev_l, cfp, cfn))} for cfp, cfn in [(1, 1), (5, 1), (1, 5)]]
        bh_ev = benjamini_hochberg(ev_p, 0.05)
        out["decision_cost"] = {}
        out["decision_cost"]["nominal_bh_0.05"] = _cost_block(bh_ev.rejected)
        th10 = calibrate_screening_threshold(cal_p, cal_l, 0.10)
        cal10 = screening_decisions(ev_p, th10.cutoff)
        out["decision_cost"]["calibrated_0.10"] = _cost_block(cal10)
        nkeep = int(np.sum(cal10))
        order = np.argsort(-ev_scores, kind="stable")
        tk = np.zeros(ev_p.size, dtype=bool); tk[order[: max(1, nkeep)]] = True
        out["decision_cost"]["topk_matched"] = _cost_block(tk)
        return out

    # Load model once
    import torch as _torch
    model = _torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.eval()
    # Known positives map (same as experiment_hetionet.py)
    known = {}
    for tf in (dataset.training, dataset.validation, dataset.testing):
        for h, r, t in tf.mapped_triples.tolist():
            known.setdefault((int(h), int(r)), set()).add(int(t))
    type_map, eid_to_prefix = entity_type_index_map(dataset)
    rel_id_to_name = dataset.training.relation_id_to_label

    if 2 in wanted:
        print("\n=== P1-2: type-constrained reference sampling (J=9, K=500) ===", flush=True)
        rng = np.random.default_rng(args.seed)
        records = build_pool_and_pvalues_constrained(model, dataset, dataset.testing, known, j=9, k=500, rng=rng, type_map=type_map, rel_id_to_name=rel_id_to_name)
        out = run_audit_from_records(records, "type_constrained_J9_K500", j=9, k=500)
        # also record comparison vs unconstrained baseline
        base = json.loads((ROOT / "results/hetionet_audit_J9_K500.json").read_text())
        out["_baseline_calibrated_0.10_FDR"] = base["calibrated"][1]["realized_fdr"]
        out["_baseline_calibrated_0.10_coverage"] = base["calibrated"][1]["coverage"]
        Path(ROOT / "results/hetionet_audit_type_constrained.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"Saved results/hetionet_audit_type_constrained.json  cal0.10 FDR {out['calibrated'][1]['realized_fdr']:.4f} vs baseline {out['_baseline_calibrated_0.10_FDR']:.4f}", flush=True)

    if 3 in wanted:
        print("\n=== P1-3: J/K grid ===", flush=True)
        # Spike already covers K=100/500/1000 at J=9; now vary J
        from experiment_hetionet import build_pool_and_pvalues as build_plain
        # Reuse plain builder for J grid (type-unconstrained, to isolate J effect)
        for j in (4, 19):
            for k in (100, 500):
                rng = np.random.default_rng(args.seed)
                print(f"\n-- J={j} K={k} --", flush=True)
                records = build_plain(model, dataset, dataset.testing, known, j=j, k=k, rng=rng)
                out = run_audit_from_records(records, f"jk_J{j}_K{k}", j=j, k=k)
                Path(ROOT / f"results/jk_grid/hetionet_J{j}_K{k}.json").parent.mkdir(parents=True, exist_ok=True)
                Path(ROOT / f"results/jk_grid/hetionet_J{j}_K{k}.json").write_text(json.dumps(out, indent=2, default=str))
                print(f"Saved jk_grid/hetionet_J{j}_K{k}.json  pi0={out['pi0']:.4f}  cal0.10 FDR {out['calibrated'][1]['realized_fdr']:.4f} cov {out['calibrated'][1]['coverage']:.2%}", flush=True)
        print("P1-3 J/K grid done")

    if 4 in wanted:
        print("\n=== P1-4: per-query grouped calibration/evaluation split ===", flush=True)
        from experiment_hetionet import build_pool_and_pvalues as build_plain
        rng = np.random.default_rng(args.seed)
        records = build_plain(model, dataset, dataset.testing, known, j=9, k=500, rng=rng)
        # Grouped split: all candidates of a query go to same half
        # records are in query order: 1+J per query, so group by query index
        J = 9
        n_queries = len(records) // (J + 1)
        # Shuffle queries, not records
        q_idx = np.arange(n_queries)
        rng2 = np.random.default_rng(args.seed)
        rng2.shuffle(q_idx)
        half_q = n_queries // 2
        cal_q = set(q_idx[:half_q].tolist())
        # Build cal/ev masks by query
        is_cal = np.array([ (i // (J+1)) in cal_q for i in range(len(records)) ])
        is_ev  = ~is_cal
        p = np.array([r["p"] for r in records])
        scores = np.array([r["score"] for r in records])
        labels = np.array([r["label"] for r in records])
        rel_names = np.array([rel_id_to_name[r["relation"]] for r in records])
        cal_p, cal_l = p[is_cal], labels[is_cal]
        ev_p, ev_l = p[is_ev], labels[is_ev]
        ev_rel = rel_names[is_ev]
        print(f"Grouped split: cal {is_cal.sum()} items ({cal_l.mean():.3%} pos), ev {is_ev.sum()} items ({ev_l.mean():.3%} pos)", flush=True)
        # Run audit with grouped halves (reuse run_audit_from_records path with pre-split indices)
        # For simplicity, rerun via helper by monkey-patching the shuffle
        from stats_core import benjamini_hochberg as _bh, calibrate_screening_threshold as _cal, realized_error_metrics as _re, screening_decisions as _sd, storey_pi0 as _pi0
        from evaluation import top_k_curve as _tk
        from baseline_calibrator import calibrator_curve as _cc
        from baseline_conformal import conformal_curve as _conf
        from stats_core import calibrate_threshold_by_cost as _ctc, expected_decision_cost as _edc
        cal_scores, cal_labels = scores[is_cal], labels[is_cal]
        ev_scores, ev_labels = scores[is_ev], labels[is_ev]
        out = {"pool_size": len(records), "base_rate": float(labels.mean()), "j": 9, "k": 500, "pi0": float(_pi0(p)), "pool_name": "grouped_split", "split_mode": "per-query grouped"}
        bh = _bh(p, 0.05)
        out["nominal_bh_0.05"] = {"claimed_fdr": 0.05, **_re(bh.rejected, labels), "cutoff": bh.threshold}
        out["calibrated"] = []
        for target in (0.05, 0.10, 0.20):
            th = _cal(cal_p, cal_l, target)
            out["calibrated"].append({"claimed_fdr": target, "calibrated_cutoff": th.cutoff, **_re(_sd(ev_p, th.cutoff), ev_l)})
        out["calibrated_per_relation"] = {}
        dec_all = np.zeros(ev_p.size, dtype=bool)
        # per-relation pooled needs global index; approximate via ev-only
        for rn in ["CbG", "CuG", "CdG", "CtD", "CpD"]:
            sel_cal = rel_names[is_cal] == rn
            sel_ev = ev_rel == rn
            if sel_cal.sum() < 50 or sel_ev.sum() == 0:
                continue
            th = _cal(cal_p[sel_cal], cal_l[sel_cal], 0.10)
            local = _sd(ev_p[sel_ev], th.cutoff)
            out["calibrated_per_relation"][rn] = {"claimed_fdr": 0.10, "n_rel_pool": int(sel_ev.sum()), "cutoff": th.cutoff, **_re(local, ev_l[sel_ev])}
        out["topk"] = [r.__dict__ for r in _tk(scores, labels, [0.05, 0.1, 0.2])]
        try:
            out["calibrator"] = {"platt": _cc(cal_scores, cal_labels, ev_scores, ev_labels, "platt"), "isotonic": _cc(cal_scores, cal_labels, ev_scores, ev_labels, "isotonic")}
        except Exception as e:
            print(f"  calibrator skip: {e}")
        try:
            out["conformal"] = _conf(cal_scores, cal_labels, ev_scores, ev_labels)
        except Exception as e:
            print(f"  conformal skip: {e}")
        out["cost_aware"] = []
        for cfp, cfn in [(1,1),(5,1),(1,5)]:
            th = _ctc(cal_p, cal_l, cfp, cfn)
            dec = _sd(ev_p, th)
            r = _re(dec, ev_l)
            out["cost_aware"].append({"c_fp": cfp, "c_fn": cfn, "threshold": th, **r, "expected_cost": float(_edc(dec, ev_l, cfp, cfn))})
        base = json.loads((ROOT / "results/hetionet_audit_J9_K500.json").read_text())
        out["_baseline_calibrated_0.10_FDR"] = base["calibrated"][1]["realized_fdr"]
        out["_baseline_grouped_delta"] = "compare grouped vs record-level at same J/K"
        Path(ROOT / "results/hetionet_audit_grouped_split.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"Saved results/hetionet_audit_grouped_split.json  cal0.10 FDR {out['calibrated'][1]['realized_fdr']:.4f} vs baseline {out['_baseline_calibrated_0.10_FDR']:.4f} (record-level)", flush=True)
        print("P1-4 grouped split done")

    print("\nAll requested P1 variants complete.")
