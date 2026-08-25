"""Train converged WN18RR models for the reliability-screening paper.

Trains TransE, RotatE, ConvE, ComplEx on WN18RR with a uniform recipe
(d=256, batch 256, lr 1e-3, Adam) and validation-based early stopping,
then saves each model with torch.save and records the final filtered MRR.

Unlike the old manuscript's fixed 200-epoch budget (under-trained), these
models are trained to early-stopped convergence so the screening diagnostics
describe trained-model behavior, not training-budget artifacts.

ConvE uses pykeen's default ConvE architecture (embedding_dim derived from
its convolution channels), a documented architecture difference.

Usage: python code/train_wn18rr_models.py [--max-epochs 1500] [--dim 256]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: E402,F401  (torch.load weights_only shim, import first)

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "wn18rr"

MODELS = ["TransE", "RotatE", "ComplEx", "ConvE"]


def train_one(model_name: str, max_epochs: int, dim: int, seed: int) -> dict:
    from pykeen.pipeline import pipeline

    print(f"\n=== Training {model_name} (max {max_epochs} epochs, d={dim}) ===",
          flush=True)
    t0 = time.time()
    model_kwargs = dict(embedding_dim=dim)
    if model_name == "ConvE":
        # ConvE derives embedding_dim from its channels; keep pykeen defaults.
        model_kwargs = dict(
            input_channels=1, output_channels=32,
            embedding_height=10, embedding_width=20,
        )
    result = pipeline(
        dataset="WN18RR",
        model=model_name,
        model_kwargs=model_kwargs,
        epochs=max_epochs,
        training_kwargs=dict(batch_size=256, use_tqdm_batch=False),
        optimizer_kwargs=dict(lr=0.001),
        evaluation_kwargs=dict(batch_size=256),
        stopper="early",
        stopper_kwargs=dict(
            frequency=15,
            patience=45,
            metric="both.realistic.inverse_harmonic_mean_rank",
            relative_delta=0.001,
            larger_is_better=True,
        ),
        random_seed=seed,
        device="cpu",
        use_tqdm=False,
    )
    dt = time.time() - t0
    metrics = result.metric_results.to_flat_dict()
    fwd_mrr = metrics.get("both.realistic.inverse_harmonic_mean_rank")
    filt_hits10 = metrics.get("both.realistic.hits_at_10")
    stopper = result.stopper
    elapsed_epochs = stopper.best_epoch if (stopper and stopper.best_epoch is not None) else max_epochs
    print(f"  {model_name}: stopped at epoch {elapsed_epochs} "
          f"after {dt/3600:.1f}h; filtered MRR={fwd_mrr:.4f} "
          f"Hits@10={filt_hits10:.4f}", flush=True)

    model = result.model
    save_path = OUT_DIR / f"{model_name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, save_path)
    return {
        "model": model_name,
        "dim": dim,
        "elapsed_epochs": int(elapsed_epochs),
        "hours": round(dt / 3600, 2),
        "filtered_mrr": float(fwd_mrr) if fwd_mrr is not None else None,
        "filtered_hits10": float(filt_hits10) if filt_hits10 is not None else None,
        "save_path": str(save_path),
    }


def main(max_epochs: int, dim: int, seed: int, only: str | None):
    manifest = []
    manifest_path = OUT_DIR / "manifest.json"
    for m in MODELS:
        if only and m != only:
            continue
        entry = train_one(m, max_epochs, dim, seed)
        manifest.append(entry)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  manifest updated: {manifest_path}", flush=True)
    print(f"\nAll done. Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="train a single model name")
    args = parser.parse_args()
    main(args.max_epochs, args.dim, args.seed, args.only)
