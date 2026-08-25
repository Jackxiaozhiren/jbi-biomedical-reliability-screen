"""Explore Hetionet drug-disease decision space for the biomedical subgraph.

Computes per-relation train/valid/test counts, the CtD/CpD decision-space size
(drugs x diseases, true edges, positive density), and the size of a
drug-disease-centric subgraph (relations involving compounds/diseases/genes/
side-effects, excluding the pure gene-gene and anatomy bulk).

Usage: python code/explore_hetionet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: E402,F401

from collections import Counter


def main():
    from pykeen.datasets import Hetionet

    d = Hetionet()
    print("Hetionet loaded:", d.num_entities, "entities,", d.num_relations,
          "relations")
    rel_to_id = {k: v for k, v in d.relation_to_id.items()}
    id_to_rel = {v: k for k, v in rel_to_id.items()}

    all_counts = Counter()
    for tf in (d.training, d.validation, d.testing):
        for h, r, t in tf.mapped_triples.tolist():
            all_counts[id_to_rel[r]] += 1

    print("\nAll relations (name: total edges):")
    for name, cnt in all_counts.most_common():
        print(f"  {name}: {cnt:,}")

    # drug-disease decision relations
    decision = {"CtD", "CpD"}
    dec_count = sum(all_counts.get(r, 0) for r in decision)
    print(f"\nDecision relations {sorted(decision)}: {dec_count:,} edges total")

    # compounds and diseases (infer metanode from node name prefix in pykeen ids)
    tf_all = d.training
    print("\nSample entity ids (first 15):", tf_all.entity_id_to_label)
    # build drug->set(disease) for CtD/CpD
    drugs = set()
    diseases = set()
    drug_disease = Counter()
    for tf in (d.training, d.validation, d.testing):
        for h, r, t in tf.mapped_triples.tolist():
            rn = id_to_rel[r]
            if rn in decision:
                drugs.add(h)
                diseases.add(t)
                drug_disease[(h, t)] += 1
    print(f"\nDrug-disease decision space:")
    print(f"  unique compounds (h): {len(drugs)}")
    print(f"  unique diseases (t): {len(diseases)}")
    print(f"  unique true (drug,disease) pairs: {len(drug_disease)}")
    print(f"  candidate space = {len(drugs)*len(diseases):,}")
    print(f"  positive density = {len(drug_disease)/(len(drugs)*len(diseases)):.4f}")

    # drug-disease-centric subgraph: relations involving compound/disease/gene/side-effect
    keep = {
        "CtD", "CpD", "CbG", "CuG", "CdG", "DaG", "CcSE",
        "CrC", "CcGA", "DlA", "GpG", "GdG",
    }
    keep = keep & set(all_counts.keys())
    sub_total = sum(all_counts.get(r, 0) for r in keep)
    print(f"\nDrug-disease-centric subgraph ({len(keep)} relations):")
    for r in sorted(keep):
        print(f"  {r}: {all_counts.get(r,0):,}")
    print(f"  TOTAL: {sub_total:,} edges")


if __name__ == "__main__":
    main()
