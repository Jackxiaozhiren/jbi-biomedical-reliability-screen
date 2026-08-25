"""Build the drug-centric Hetionet subgraph for the biomedical application.

Design (locked in P3):
- Relations kept: drug-disease decisions (CtD, CpD), drug-target/DTI
  (CbG, CuG, CdG), bridges (DaG), drug-side-effect (CcSE), drug-drug
  (CrC, CcGA if present), and a small gene-gene structure subset
  (GcG, GiG). Excludes the anatomy-heavy and huge gene-function bulk
  (AeG, AdG, AuG, GpBP, Gr>G, GpMF, GpPW, GpCC, DuG, DdG, DlA, DpS, DrD, PCiC).
- Decision-task holdout: the DTI edges (CbG/CuG/CdG) and drug-disease edges
  (CtD/CpD) are each split train/valid/test (70/10/20, seeded), and the test
  slices become the "true positive" pool for the realized-FDR experiment.
- Output: a saved subgraph package (JSON edges + splits) cached under
  results/ so downstream experiments do not rebuild it.

Usage: python code/build_hetionet_subgraph.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent

KEEP_RELATIONS = {
    "CtD", "CpD", "CbG", "CuG", "CdG", "DaG", "CcSE", "CrC", "CcGA",
    "GcG", "GiG",
}
DECISION_DTI = {"CbG", "CuG", "CdG"}
DECISION_DD = {"CtD", "CpD"}


def main(seed: int = 42):
    from pykeen.datasets import Hetionet

    d = Hetionet()
    id_to_rel = {v: k for k, v in d.relation_to_id.items()}

    edges = []  # (h, r, t, relname)
    for tf in (d.training, d.validation, d.testing):
        for h, r, t in tf.mapped_triples.tolist():
            rn = id_to_rel[r]
            if rn in KEEP_RELATIONS:
                edges.append((int(h), int(r), int(t), rn))

    rel_names = {rn for _, _, _, rn in edges}
    counts = {}
    for rn in sorted(rel_names):
        counts[rn] = sum(1 for e in edges if e[3] == rn)
    total = len(edges)
    n_nodes = len({e[0] for e in edges} | {e[2] for e in edges})
    print(f"Subgraph: {total:,} edges, {n_nodes} entities, "
          f"{len(rel_names)} relations")
    for rn in sorted(counts):
        print(f"  {rn}: {counts[rn]:,}")

    # split decision edges train/valid/test
    rng = np.random.default_rng(seed)

    def split_decision(names: set[str]) -> tuple[list, list, list]:
        idx = [i for i, e in enumerate(edges) if e[3] in names]
        perm = rng.permutation(len(idx))
        n_tr = int(0.70 * len(idx))
        n_va = int(0.10 * len(idx))
        tr = [idx[perm[i]] for i in range(n_tr)]
        va = [idx[perm[i]] for i in range(n_tr, n_tr + n_va)]
        te = [idx[perm[i]] for i in range(n_tr + n_va, len(idx))]
        return tr, va, te

    dti_tr, dti_va, dti_te = split_decision(DECISION_DTI)
    dd_tr, dd_va, dd_te = split_decision(DECISION_DD)

    train_idx = set(dti_tr) | set(dd_tr)
    valid_idx = set(dti_va) | set(dd_va)
    test_idx = dti_te + dd_te

    # all non-decision edges go to train
    for i, e in enumerate(edges):
        if e[3] not in DECISION_DTI | DECISION_DD:
            train_idx.add(i)

    def to_records(idx):
        return [
            {"head": edges[i][0], "relation_id": edges[i][1],
             "relation": edges[i][3], "tail": edges[i][2]}
            for i in idx
        ]

    out = {
        "seed": seed,
        "relations_kept": sorted(rel_names),
        "train": to_records(sorted(train_idx)),
        "valid": to_records(sorted(valid_idx)),
        "test_true_positives": to_records(test_idx),
        "n_entities": n_nodes,
        "counts": counts,
        "test_dti": len(dti_te),
        "test_dd": len(dd_te),
        "meta": {
            "decision_dti": sorted(DECISION_DTI),
            "decision_dd": sorted(DECISION_DD),
            "holdout": "70/10/20 seeded",
        },
    }
    out_path = ROOT / "results" / "hetionet_subgraph_v1.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nSaved: {out_path}")
    print(f"  train={len(out['train']):,} valid={len(out['valid']):,} "
          f"test_true={len(out['test_true_positives'])} "
          f"(dti={len(dti_te)}, dd={len(dd_te)})")


if __name__ == "__main__":
    main()
