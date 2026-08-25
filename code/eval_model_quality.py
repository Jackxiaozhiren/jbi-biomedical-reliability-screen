"""Evaluate filtered MRR / Hits@10 for the trained models (reproducibility + paper model table).

Loads the frozen .pt checkpoints and evaluates on the standard test split with
train+validation filtering, matching how the paper's "converged models" claim is
stated. For the Hetionet core the custom seeded split is rebuilt via
build_core_dataset(seed 42).

Usage: python code/eval_model_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: E402,F401


def eval_standard(dataset_cls, model_path: str, name: str):
    from pykeen.evaluation import RankBasedEvaluator
    ds = dataset_cls()
    model = torch.load(ROOT / "models" / model_path, map_location="cpu", weights_only=False)
    model.eval()
    evaluator = RankBasedEvaluator()
    filt = [ds.training.mapped_triples, ds.validation.mapped_triples]
    metrics = evaluator.evaluate(model, ds.testing.mapped_triples, batch_size=256,
                                 additional_filter_triples=filt).to_flat_dict()
    mrr = metrics.get("both.realistic.inverse_harmonic_mean_rank")
    hits10 = metrics.get("both.realistic.hits_at_10")
    hits1 = metrics.get("both.realistic.hits_at_1")
    print(f"{name}: filtered MRR={mrr:.4f} Hits@10={hits10:.4f} Hits@1={hits1:.4f}")
    return {"model": name, "filtered_mrr": float(mrr), "filtered_hits10": float(hits10)}


def main():
    from pykeen.datasets import FB15k237, WN18RR
    results = []
    results.append(eval_standard(WN18RR, "WN18RR_RotatE.pt", "WN18RR RotatE"))
    results.append(eval_standard(WN18RR, "WN18RR_TransE.pt", "WN18RR TransE"))
    if (ROOT / "models" / "WN18RR_ComplEx.pt").exists():
        results.append(eval_standard(WN18RR, "WN18RR_ComplEx.pt", "WN18RR ComplEx"))
    results.append(eval_standard(FB15k237, "FB15k237_RotatE.pt", "FB15k-237 RotatE"))
    # Hetionet core: rebuild the seeded split
    try:
        from train_hetionet_core import build_core_dataset
        from pykeen.evaluation import RankBasedEvaluator
        ds = build_core_dataset(42)
        model = torch.load(ROOT / "models" / "hetionet_core_RotatE.pt",
                           map_location="cpu", weights_only=False)
        model.eval()
        ev = RankBasedEvaluator()
        filt = [ds.training.mapped_triples]
        m = ev.evaluate(model, ds.validation.mapped_triples, batch_size=256,
                        additional_filter_triples=filt).to_flat_dict()
        print(f"Hetionet core RotatE (validation): filtered MRR={m.get('both.realistic.inverse_harmonic_mean_rank'):.4f}")
    except Exception as e:  # noqa: BLE001
        print("Hetionet core eval skipped:", type(e).__name__, e)
    print("RESULT_JSON=", end="")
    import json
    print(json.dumps(results))


if __name__ == "__main__":
    main()
