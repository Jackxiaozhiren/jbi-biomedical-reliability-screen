"""E1 exporter: apply FROZEN thresholds to the full absent candidate space.

Implements external_validation/protocol_frozen.md (2026-08-22):
  * candidate space: all (Compound, CtD, Disease) and (Compound, CbG, Gene)
    triples absent from the ENTIRE core subgraph (train+valid+test union);
  * index: shared-reference rank-extremeness p, K=500 refs per query, refs
    sampled from entities excluding all known positives of the query and all
    candidate tails (type-unconstrained, matching the main-audit construction
    under which the calibrated cutoffs were frozen);
  * ref stream: default_rng(42), independent of the main-audit sampling stream;
  * KEEP/WITHHOLD flags at the frozen thresholds read from
    results/hetionet_audit_J9_K500.json (never hard-coded here).

Model loading is pickle-free: the workspace keeps state-dict artifacts
(models/hetionet_core_<M>.sd.pth + .meta.json) produced from the training
outputs; they are loaded with torch.load(weights_only=True) only, and the
model object is rebuilt from the same triples factory used in training.
Parity of the rebuild against the original artifact was verified once
(max |delta score| = 0.0 on probe triples; see qa/02 experiment log).

Outputs (external_validation/):
  candidates_ctd.tsv / candidates_cbg.tsv
  candidates_summary.json

Usage: python external_validation/export_candidates.py \
           [--model RotatE] [--k 500] [--seed 42]
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
sys.path.insert(0, str(ROOT / "code"))
import env_patch  # noqa: E402,F401

from train_hetionet_core import build_core_dataset  # noqa: E402

# CLI-supplied names are validated against allowlists before any filesystem
# path is built from them (prevents path traversal via arguments).
RELATIONS = {
    "CtD": "Disease",
    "CbG": "Gene",
}
ALLOWED_MODELS = {"RotatE", "ComplEx"}
OUT_DIR_NAME = "external_validation"


def resolve_under(base: Path, candidate: Path) -> Path:
    """Resolve candidate and require it to stay inside base."""
    base = base.resolve()
    resolved = candidate.resolve()
    if base != resolved and base not in resolved.parents:
        raise ValueError(f"path escapes {base}: {resolved}")
    return resolved


def load_frozen_thresholds(thresholds_json: str = "hetionet_audit_J9_K500.json"
                           ) -> dict:
    """Read the frozen operating points from the audited results JSON."""
    path = resolve_under(ROOT, ROOT / "results" / thresholds_json)
    d = json.loads(path.read_text())
    cal = {c["claimed_fdr"]: c["calibrated_cutoff"] for c in d["calibrated"]}
    cost11 = [c["threshold"] for c in d["cost_aware"]
              if c["c_fp"] == 1 and c["c_fn"] == 1][0]
    per_rel = {rn: c["cutoff"] for rn, c in d["calibrated_per_relation"].items()
               if rn != "_pooled_all_"}
    return {
        "main": cal[0.10],
        "t20": cal[0.20],
        "cost11": cost11,
        "relcond": per_rel,
    }


def load_model_safely(model_name: str, training_factory):
    """Rebuild the model from a tensor-only state-dict artifact.

    No pickle deserialization: the .sd.pth holds plain tensors and is loaded
    with weights_only=True; the architecture is rebuilt from the recorded
    class name (validated against pykeen's model registry) on the same
    triples factory used at training time.
    """
    import pykeen.models as pkm
    meta_path = resolve_under(ROOT, ROOT / "models" / f"hetionet_core_{model_name}.meta.json")
    sd_path = resolve_under(ROOT, ROOT / "models" / f"hetionet_core_{model_name}.sd.pth")
    meta = json.loads(meta_path.read_text())
    cls_name = meta["class"]
    if cls_name not in pkm.__dict__:
        raise ValueError(f"unknown model class in meta: {cls_name}")
    model = getattr(pkm, cls_name)(triples_factory=training_factory,
                                   embedding_dim=256, random_seed=42)
    state = torch.load(sd_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def entity_kind_index(dataset):
    """Map entity label prefix -> array of entity ids for the training factory.

    entity_id_to_label is a Labeling object whose .items() iteration is
    unreliable in this pykeen version; index access is the supported path
    (same access pattern as train_hetionet_core.py).
    """
    lab = dataset.training.entity_id_to_label
    n = dataset.training.num_entities
    kinds = {}
    for eid in range(n):
        label = lab[eid]
        kind = label.split("::", 1)[0]
        kinds.setdefault(kind, []).append(eid)
    return {k: np.asarray(v, dtype=np.int64) for k, v in kinds.items()}


def known_tails(dataset):
    known = {}
    for tf in (dataset.training, dataset.validation, dataset.testing):
        for h, r, t in tf.mapped_triples.tolist():
            known.setdefault((int(h), int(r)), set()).add(int(t))
    return known


def relation_id(dataset, name):
    rl = dataset.training.relation_id_to_label
    for rid in range(dataset.training.num_relations):
        if rl[rid] == name:
            return rid
    raise KeyError(name)


def export_relation(model, dataset, rel_name, tail_kind, k, seed, thr, out_path):
    out_path = resolve_under(ROOT, out_path)
    rid = relation_id(dataset, rel_name)
    lab = dataset.training.entity_id_to_label
    known = known_tails(dataset)
    kinds = entity_kind_index(dataset)
    tails_all = kinds[tail_kind]
    n_ent = dataset.training.num_entities
    rng = np.random.default_rng(seed)
    t0 = time.time()

    rows = []
    heads = [h for (h, r) in known if r == rid]
    heads += [h for h in range(n_ent)
              if lab[h].startswith("Compound::") and (h, rid) not in known]
    heads = sorted(set(heads))
    print(f"[{rel_name}] {len(heads)} queries x {len(tails_all)} {tail_kind} tails",
          flush=True)

    for qi, h in enumerate(heads):
        known_h = np.asarray(sorted(known.get((h, rid), set())), dtype=np.int64)
        cand = np.setdiff1d(tails_all, known_h, assume_unique=False)
        if cand.size == 0:
            continue
        # refs: exclude known positives AND all candidates (protocol §4)
        excl = np.union1d(known_h, cand)
        mask = np.ones(n_ent, dtype=bool)
        mask[excl] = False
        ref_pool = np.flatnonzero(mask)
        ref = ref_pool if ref_pool.size < k else rng.choice(ref_pool, size=k,
                                                            replace=False)
        cands_and_refs = np.concatenate([cand, ref])
        hrt = np.stack([
            np.full(cands_and_refs.size, h),
            np.full(cands_and_refs.size, rid),
            cands_and_refs,
        ], axis=1)
        with torch.no_grad():
            s = model.score_hrt(torch.tensor(hrt)).cpu().numpy().reshape(-1)
        cand_scores, ref_scores = s[: cand.size], s[cand.size:]
        ps = (1.0 + np.sum(ref_scores[None, :] >= cand_scores[:, None],
                           axis=1)) / (ref.size + 1)
        for t, sc, p in zip(cand, cand_scores, ps):
            rows.append((lab[h], rel_name, lab[int(t)], float(p), float(sc)))
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{len(heads)} queries, {len(rows):,} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)

    main, t20, cost11 = thr["main"], thr["t20"], thr["cost11"]
    relc = thr["relcond"].get(rel_name, thr["main"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("head_label\trelation\ttail_label\tp\tscore\t"
                "keep_main\tkeep_t20\tkeep_cost11\tkeep_relcond\n")
        for hl, rn, tl, p, sc in rows:
            f.write(f"{hl}\t{rn}\t{tl}\t{p:.8g}\t{sc:.8g}\t"
                    f"{int(p <= main)}\t{int(p <= t20)}\t{int(p <= cost11)}\t"
                    f"{int(p <= relc)}\n")
    summary = {
        "relation": rel_name, "model_rows": len(rows),
        "n_keep_main": sum(1 for r in rows if r[3] <= main),
        "n_keep_t20": sum(1 for r in rows if r[3] <= t20),
        "n_keep_cost11": sum(1 for r in rows if r[3] <= cost11),
        "n_keep_relcond": sum(1 for r in rows if r[3] <= relc),
        "thresholds": {"main": main, "t20": t20, "cost11": cost11,
                       "relcond": relc},
        "k": k, "seed": seed, "seconds": round(time.time() - t0, 1),
    }
    print(f"[{rel_name}] done: {summary}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RotatE", choices=sorted(ALLOWED_MODELS))
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relations", default="CtD,CbG")
    ap.add_argument("--thresholds-json", default="hetionet_audit_J9_K500.json",
                    help="audit JSON holding the frozen cutoffs "
                         "(use hetionet_audit_J9_K500_complex.json for the "
                         "ComplEx sensitivity)")
    ap.add_argument("--suffix", default="",
                    help="output filename suffix, e.g. _complex")
    args = ap.parse_args()

    rels = [r.strip() for r in args.relations.split(",") if r.strip()]
    for r in rels:
        if r not in RELATIONS:
            raise SystemExit(f"unknown relation {r!r}; allowed: {sorted(RELATIONS)}")

    torch.set_num_threads(max(1, torch.get_num_threads() - 2))
    thr = load_frozen_thresholds(args.thresholds_json)
    print(f"Frozen thresholds from {args.thresholds_json}: {thr}", flush=True)
    ds = build_core_dataset(args.seed)
    model = load_model_safely(args.model, ds.training)

    outdir = ROOT / OUT_DIR_NAME
    summaries = {"model": args.model, "k": args.k, "seed": args.seed,
                 "thresholds": thr}
    for rel in rels:
        tail_kind = RELATIONS[rel]
        s = export_relation(model, ds, rel, tail_kind, args.k, args.seed, thr,
                            outdir / f"candidates_{rel.lower()}{args.suffix}.tsv")
        summaries[rel] = s
    summary_path = resolve_under(ROOT,
                                 outdir / f"candidates_summary{args.suffix}.json")
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
