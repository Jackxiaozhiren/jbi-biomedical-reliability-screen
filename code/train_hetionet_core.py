"""Build the drug-centric Hetionet CORE subgraph and train models on it.

Core relations (P3, application): drug-disease CtD/CpD, DTI CbG/CuG/CdG,
disease-gene bridge DaG, drug-drug CrC. The big gene-gene (GcG, GiG) and
side-effect (CcSE) bulks are dropped so CPU-converged training is feasible
(~0.8x WN18RR edge count). The drug-repurposing pathway (drug -> gene,
disease -> gene, drug -> disease) is fully preserved.

Decision edges (CbG, CuG, CdG, CtD, CpD) are split 70/10/20 (seeded); context
edges (DaG, CrC) go entirely to training. The test slice is the "true
positive" pool for the realized-vs-claimed FDR experiment.

Training is checkpoint-resumable (pykeen's minute-based checkpoint_frequency):
kill the process any time, rerun the identical command to continue.

Usage:
    python code/train_hetionet_core.py [--models TransE] [--max-epochs 1500] \
        [--dim 256] [--ckpt-every 5]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: E402,F401

import numpy as np
import torch  # noqa: E402

CORE_RELATIONS = {"CtD", "CpD", "CbG", "CuG", "CdG", "DaG", "CrC"}
DECISION = {"CtD", "CpD", "CbG", "CuG", "CdG"}


def build_core_dataset(seed: int):
    from pykeen.datasets import Hetionet
    from pykeen.triples import TriplesFactory
    from pykeen.datasets.base import EagerDataset

    d = Hetionet()
    id_to_rel = {v: k for k, v in d.relation_to_id.items()}
    head_label = d.training.entity_id_to_label
    tail_label = head_label

    # collect core edges as (head_label, rel_name, tail_label)
    edges = []
    for tf in (d.training, d.validation, d.testing):
        for h, r, t in tf.mapped_triples.tolist():
            rn = id_to_rel[int(r)]
            if rn in CORE_RELATIONS:
                edges.append((head_label[int(h)], rn, tail_label[int(t)]))

    rng = np.random.default_rng(seed)
    dec_idx = [i for i, e in enumerate(edges) if e[1] in DECISION]
    ctx_idx = [i for i, e in enumerate(edges) if e[1] not in DECISION]
    perm = rng.permutation(len(dec_idx))
    n_tr = int(0.70 * len(dec_idx)); n_va = int(0.10 * len(dec_idx))
    tr = [dec_idx[perm[i]] for i in range(n_tr)]
    va = [dec_idx[perm[i]] for i in range(n_tr, n_tr + n_va)]
    te = [dec_idx[perm[i]] for i in range(n_tr + n_va, len(dec_idx))]
    train_edges = [edges[i] for i in tr + ctx_idx]
    valid_edges = [edges[i] for i in va]
    test_edges = [edges[i] for i in te]

    # Build ONE TriplesFactory over ALL edges so entity/relation IDs are
    # IDENTICAL across splits (separate from_labeled_triples calls would
    # re-assign relation IDs and silently corrupt evaluation). map_triples
    # converts each split's label triples to the shared ID space, preserving
    # split membership.
    all_edges = train_edges + valid_edges + test_edges
    full_tf = TriplesFactory.from_labeled_triples(np.asarray(all_edges, dtype=object))

    def make_tf(rows):
        return TriplesFactory(
            mapped_triples=full_tf.map_triples(np.asarray(rows, dtype=object)),
            entity_to_id=full_tf.entity_to_id,
            relation_to_id=full_tf.relation_to_id,
            create_inverse_triples=False,
        )

    tr_tf = make_tf(train_edges)
    va_tf = make_tf(valid_edges)
    te_tf = make_tf(test_edges)
    print(f"Hetionet core subgraph: train={len(train_edges):,} "
          f"valid={len(valid_edges):,} test(true)={len(test_edges):,} "
          f"entities={tr_tf.num_entities} relations={tr_tf.num_relations}")
    assert (tr_tf.relation_id_to_label == va_tf.relation_id_to_label
            == te_tf.relation_id_to_label), "relation ID maps must agree across splits"
    return EagerDataset(training=tr_tf, validation=va_tf, testing=te_tf,
                        metadata={"name": "HetionetCore-v3", "seed": str(seed)})


def train_one(dataset, model_name, max_epochs, dim, seed, ckpt_dir, ckpt_freq):
    from pykeen.pipeline import pipeline

    checkpoint_path = ckpt_dir / f"hetionet_core_{model_name}_d{dim}.ckpt"
    resumed_from = checkpoint_path.exists()
    print(f"\n=== HetionetCore/{model_name}: max {max_epochs} epochs, d={dim}, "
          f"ckpt={checkpoint_path.name} (resume={'YES' if resumed_from else 'no'}) ===",
          flush=True)
    t0 = time.time()
    result = pipeline(
        dataset=dataset,
        model=model_name,
        model_kwargs=dict(embedding_dim=dim),
        epochs=max_epochs,
        training_kwargs=dict(
            batch_size=256,
            use_tqdm_batch=False,
            continue_training=checkpoint_path.exists(),
            checkpoint_directory=ckpt_dir,
            checkpoint_name=f"hetionet_core_{model_name}_d{dim}.ckpt",
            checkpoint_frequency=ckpt_freq,
            checkpoint_on_failure=True,
        ),
        optimizer_kwargs=dict(lr=0.001),
        evaluation_kwargs=dict(batch_size=256),
        stopper="early",
        stopper_kwargs=dict(
            frequency=15, patience=45,
            metric="both.realistic.inverse_harmonic_mean_rank",
            relative_delta=0.001, larger_is_better=True,
        ),
        random_seed=seed,
        device="cpu",
        use_tqdm=False,
    )
    dt = time.time() - t0
    metrics = result.metric_results.to_flat_dict()
    fwd_mrr = metrics.get("both.realistic.inverse_harmonic_mean_rank")
    emit = {"model": model_name, "dataset": "hetionet_core", "dim": dim,
            "resumed": resumed_from, "hours": round(dt / 3600, 2),
            "filtered_mrr": float(fwd_mrr) if fwd_mrr is not None else None}
    save_path = ROOT / "models" / f"hetionet_core_{model_name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model, save_path)
    manifest = ROOT / "models" / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    existing = [e for e in existing if not (e.get("dataset") == "hetionet_core"
                                            and e.get("model") == model_name)]
    existing.append(emit)
    manifest.write_text(json.dumps(existing, indent=1))
    print(f"  {emit} -> {save_path}", flush=True)
    checkpoint_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="TransE")
    ap.add_argument("--max-epochs", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    args = ap.parse_args()

    torch.set_num_threads(min(10, os.cpu_count() or 1))
    ds = build_core_dataset(args.seed)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        train_one(ds, m, args.max_epochs, args.dim, args.seed, ckpt_dir, args.ckpt_every)
    print("\nHetionet core models complete.")


if __name__ == "__main__":
    main()