"""Resumable orchestrator: runs every training job and experiment in sequence.

Safe to run, kill, and re-run the identical command at any time: each model
resumes from its own checkpoint (pykeen minute-based checkpoints), and each
experiment runs only when its trained model exists.

  python code/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

JOBS = [
    {
        "cmd": [sys.executable, "code/train_hetionet_core.py", "--models", "RotatE"],
        "name": "HetionetCore RotatE v3 (headline application, fixed eval)",
        "done_file": "models/hetionet_core_RotatE.pt",
    },
    {
        "cmd": [sys.executable, "code/train_wn18rr_models_v2.py", "--models", "RotatE"],
        "name": "WN18RR RotatE (resume, strong signal)",
        "done_file": "models/WN18RR_RotatE.pt",
    },
    {
        "cmd": [sys.executable, "code/train_wn18rr_models_v2.py", "--dataset", "FB15k237", "--models", "RotatE", "--stopper-frequency", "50", "--patience", "20"],
        "name": "FB15k-237 RotatE (second dataset)",
        "done_file": "models/FB15k237_RotatE.pt",
    },
    {
        "cmd": [sys.executable, "code/train_wn18rr_models_v2.py", "--models", "TransE"],
        "name": "WN18RR TransE (resume, weak-signal control)",
        "done_file": "models/WN18RR_TransE.pt",
    },
]

# experiments run only if the trained model exists
EXPERIMENTS = [
    ("models/WN18RR_RotatE.pt", [sys.executable, "code/spike_k500.py", "--k", "500", "--model-path", "models/WN18RR_RotatE.pt"], "WN18RR spike K=500 (RotatE)"),
    ("models/WN18RR_RotatE.pt", [sys.executable, "code/experiment_realized_fdr.py", "--dataset", "WN18RR", "--j", "9", "--k", "500", "--model-cache", "models/WN18RR_RotatE.pt"], "WN18RR realized-FDR (RotatE)"),
    ("models/FB15k237_RotatE.pt", [sys.executable, "code/experiment_realized_fdr.py", "--dataset", "FB15k237", "--j", "9", "--k", "500", "--model-cache", "models/FB15k237_RotatE.pt"], "FB15k-237 realized-FDR (RotatE)"),
    ("models/hetionet_core_RotatE.pt", [sys.executable, "code/experiment_hetionet.py", "--model", "RotatE", "--j", "9", "--k", "500"], "HetionetCore realized-FDR audit (cautionary)"),
]


def run_ready_experiments():
    """Run any experiment whose trained model now exists (idempotent: reruns ok, overwrites json)."""
    ran_any = False
    for model_file, cmd, name in EXPERIMENTS:
        if (ROOT / model_file).exists():
            print(f"\n=== EXPERIMENT: {name} ===", flush=True)
            subprocess.run(cmd, cwd=CODE.parent)
            ran_any = True
        else:
            print(f"  (experiment waiting on {model_file}: {name})", flush=True)
    return ran_any


def main():
    for job in JOBS:
        done = job.get("done_file")
        if done and (ROOT / done).exists():
            print(f"\n=== SKIP (already trained): {job['name']} ===", flush=True)
            run_ready_experiments()
            continue
        print(f"\n=== JOB: {job['name']} ===", flush=True)
        subprocess.run(job["cmd"], cwd=CODE.parent)
        run_ready_experiments()  # run experiments as soon as their model is ready
    print("\n=== REPORT: consolidate all results ===", flush=True)
    subprocess.run([sys.executable, "code/summarize_all.py"], cwd=CODE.parent)
    print("\nAll jobs complete.")

if __name__ == "__main__":
    main()