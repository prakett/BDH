"""
BDH-GPU training launcher.

This file automatically uses 2 GPUs when they are available.
Run simply:

    python train.py

It launches train_ddp.py with PyTorch DistributedDataParallel.
For the current Kaggle machine this means:
    GPU 0: Tesla T4
    GPU 1: Tesla T4
    batch per GPU: 8
    effective batch: 16
"""

import os
import subprocess
import sys

import torch


NUM_GPUS_TO_USE = 2
DDP_TRAIN_FILE = "train_ddp.py"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    available = torch.cuda.device_count()
    world_size = min(NUM_GPUS_TO_USE, available)

    print("=" * 72)
    print("BDH-GPU TRAINING LAUNCHER")
    print("=" * 72)
    print(f"CUDA GPUs detected: {available}")

    for i in range(available):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"GPUs selected:      {world_size}")
    print("=" * 72)

    if not os.path.exists(DDP_TRAIN_FILE):
        raise FileNotFoundError(
            f"Could not find {DDP_TRAIN_FILE}. "
            "Keep train.py and train_ddp.py in the same folder."
        )

    # Single GPU fallback.
    if world_size == 1:
        result = subprocess.run(
            [sys.executable, DDP_TRAIN_FILE],
            check=False,
        )
        raise SystemExit(result.returncode)

    # Two-GPU DDP.
    # --tee 3 makes stdout/stderr from both ranks visible in Kaggle,
    # which is especially useful if one rank crashes.
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--tee",
        "3",
        DDP_TRAIN_FILE,
    ]

    print()
    print("Launching 2-GPU DistributedDataParallel...")
    print("Command:", " ".join(command))
    print()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(world_size)
    )

    result = subprocess.run(
        command,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        print()
        print("=" * 72)
        print("DDP TRAINING FAILED")
        print("=" * 72)
        print(
            "The launcher is working; the output above contains the "
            "actual rank-level error."
        )
        print("=" * 72)

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
