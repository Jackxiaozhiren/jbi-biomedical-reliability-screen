"""Resumable converged training of WN18RR models (checkpoint + continue).

Uses pykeen's native training-loop checkpoints: every ``--ckpt-every``
MINUTES (pykeen's checkpoint_frequency is wall-clock minutes, not epochs) the
model/optimizer/stopper state is written to ``--ckpt-dir/<Model>_d<dim>.ckpt``,
and with ``--resume`` (default on) an existing checkpoint is loaded and
training continues from the saved epoch toward ``--max-epochs``. You can kill
the process at any time (Ctrl-C / kill) and restart the exact same command
later; it picks up where it left off. A completed model's checkpoint is
deleted and the model is saved as ``models/wn18rr/<Model>.pt``.

Parallel notes: ``torch.set_num_threads`` is set to the CPU count, and the
default model set is TransE,RotatE (add more via ``--models``).

Usage (start or resume — the command is identical):
    python code/train_wn18rr_models_v2.py [--models TransE,RotatE] \
        [--max-epochs 1500] [--dim 256] [--ckpt-every 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_patch  # noqa: E402,F401  (torch.load weights_only shim, import first)

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models"
DEFAULT_CKPT_DIR = ROOT / "checkpoints"


def _report(manifest_path, entry):
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = []
    manifest = [e for e in manifest if e["model"] != entry["model"]]
    manifest.append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=1))


def train_one(model_name, dataset_name, max_epochs, dim, seed, ckpt_dir, ckpt_freq,
              stopper_frequency=15, patience=45):
    from pykeen.pipeline import pipeline

    checkpoint_path = ckpt_dir / f"{dataset_name}_{model_name}_d{dim}.ckpt"
    resumed_from = checkpoint_path.exists()
    print(f"\n=== {dataset_name}/{model_name}: max {max_epochs} epochs, d={dim}, "
          f"ckpt={ckpt_dir.name}/{checkpoint_path.name} "
          f"(resume={'YES' if resumed_from else 'no'}) ===", flush=True)
    t0 = time.time()

    model_kwargs = dict(embedding_dim=dim)
    if model_name == "ConvE":
        model_kwargs = dict(
            input_channels=1, output_channels=32,
            embedding_height=10, embedding_width=20,
        )

    result = pipeline(
        dataset=dataset_name,
        model=model_name,
        model_kwargs=model_kwargs,
        epochs=max_epochs,
        training_kwargs=dict(
            batch_size=256,
            use_tqdm_batch=False,
            continue_training=checkpoint_path.exists(),
            checkpoint_directory=ckpt_dir,
            checkpoint_name=f"{dataset_name}_{model_name}_d{dim}.ckpt",
            checkpoint_frequency=ckpt_freq,
            checkpoint_on_failure=True,
        ),
        optimizer_kwargs=dict(lr=0.001),
        evaluation_kwargs=dict(batch_size=256),
        stopper="early",
        stopper_kwargs=dict(
            frequency=stopper_frequency,
            patience=patience,
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
    elapsed = stopper.best_epoch if (stopper and stopper.best_epoch is not None) else max_epochs
    print(f"  {model_name}: best epoch {elapsed}, filtered MRR={fwd_mrr:.4f} "
          f"Hits@10={filt_hits10:.4f} in {dt/3600:.2f}h", flush=True)

    save_path = OUT_DIR / f"{dataset_name}_{model_name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model, save_path)
    _report(OUT_DIR / "manifest.json", {
        "model": model_name, "dataset": dataset_name, "dim": dim,
        "resumed": resumed_from,
        "elapsed_epochs": int(elapsed), "hours": round(dt / 3600, 2),
        "filtered_mrr": float(fwd_mrr) if fwd_mrr is not None else None,
        "filtered_hits10": float(filt_hits10) if filt_hits10 is not None else None,
        "save_path": str(save_path),
    })
    # a completed model's checkpoint is no longer needed for resume
    checkpoint_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WN18RR",
                        help="pykeen dataset name, e.g. WN18RR, FB15k237, PharmKG8k")
    parser.add_argument("--models", default="TransE,RotatE")
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-every", type=int, default=5,
                        help="checkpoint interval in MINUTES (pykeen uses wall-clock minutes, not epochs)")
    parser.add_argument("--stopper-frequency", type=int, default=15,
                        help="evaluate the early stopper every N epochs (raise for large datasets)")
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    args = parser.parse_args()

    n_threads = min(10, os.cpu_count() or 1)
    torch.set_num_threads(n_threads)
    print(f"torch threads: {n_threads}", flush=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        train_one(m, args.dataset, args.max_epochs, args.dim, args.seed, ckpt_dir,
                  args.ckpt_every, args.stopper_frequency, args.patience)
    print("\nAll requested models complete.")


if __name__ == "__main__":
    main()
