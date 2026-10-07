
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

from model import BDHModel, count_parameters


# ============================================================
# CONFIGURATION
# ============================================================

N = 16384
D = 256
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.10
VOCAB_SIZE = 256

SEQ_LEN = 1024
BATCH_SIZE = 2             # Per GPU: global batch = 4

TOTAL_STEPS = 10000
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

PRINT_EVERY = 10
SEED = 1337

DATASET_NAME = "roneneldan/TinyStories"
SAVE_PATH = "bdh_tinystories_ddp.pt"


# ============================================================
# DISTRIBUTED SETUP
# ============================================================

def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def get_learning_rate(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / max(1, WARMUP_STEPS)

    decay_steps = max(1, TOTAL_STEPS - WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / decay_steps
    progress = max(0.0, min(1.0, progress))

    return LEARNING_RATE + progress * (
        MIN_LEARNING_RATE - LEARNING_RATE
    )


# ============================================================
# DATASET
# ============================================================

def load_byte_stream():
    dataset = load_dataset(DATASET_NAME, split="train")

    # Represent the text as raw UTF-8 bytes.
    stream = bytearray()

    for row in dataset:
        text = row.get("text", "")
        stream.extend(text.encode("utf-8", errors="replace"))
        stream.append(10)

    if len(stream) <= SEQ_LEN + 1:
        raise RuntimeError("Dataset is too short.")

    # Keep token bytes in CPU memory; transfer only batches to GPU.
    return torch.frombuffer(stream, dtype=torch.uint8)


def get_batch(data, device, generator):
    # Random fixed-length windows. The two ranks use different seeds.
    starts = torch.randint(
        0,
        data.numel() - SEQ_LEN - 1,
        (BATCH_SIZE,),
        generator=generator,
    )

    offsets = torch.arange(SEQ_LEN + 1)
    positions = starts[:, None] + offsets[None, :]
    tokens = data[positions].long()

    x = tokens[:, :-1].to(device)
    targets = tokens[:, 1:].to(device)

    return x, targets


# ============================================================
# TRAINING
# ============================================================

def main():
    rank, world_size, local_rank, device = setup_distributed()

    try:
        seed = SEED + rank
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if is_main(rank):
            print("=" * 65)
            print("BDH-GPU DISTRIBUTED TRAINING")
            print("=" * 65)
            print(f"Number of GPUs:       {world_size}")
            print(f"Sequence length:      {SEQ_LEN}")
            print(f"Batch per GPU:        {BATCH_SIZE}")
            print(f"Global batch:         {BATCH_SIZE * world_size}")
            print(f"Training steps:       {TOTAL_STEPS}")
            print("Mode: fixed-context DDP")
            print("=" * 65, flush=True)

        # Each rank creates the same model structure.
        model = BDHModel(
            n=N,
            d=D,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            vocab_size=VOCAB_SIZE,
        ).to(device)

        parameter_count = count_parameters(model)

        # Synchronize model gradients across GPUs.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        if is_main(rank):
            print(f"Trainable parameters: {parameter_count:,}")
            print("Loading TinyStories...", flush=True)

        data = load_byte_stream()

        # Different random context windows on each GPU.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        if is_main(rank):
            print(f"Dataset bytes: {data.numel():,}")
            print("Starting training...", flush=True)

        model.train()
        start_time = time.time()

        for step in range(1, TOTAL_STEPS + 1):
            lr = get_learning_rate(step)

            for group in optimizer.param_groups:
                group["lr"] = lr

            x, targets = get_batch(data, device, generator)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}, rank {rank}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
                error_if_nonfinite=True,
            )

            optimizer.step()

            if step % PRINT_EVERY == 0:
                # Report the mean loss across both GPUs.
                mean_loss = loss.detach().clone()
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= world_size

                if is_main(rank):
                    loss_value = mean_loss.item()
                    perplexity = math.exp(min(loss_value, 20.0))
                    elapsed = time.time() - start_time

                    allocated = (
                        torch.cuda.memory_allocated(device) / 1024**3
                    )
                    total = (
                        torch.cuda.get_device_properties(device).total_memory
                        / 1024**3
                    )

                    print(
                        f"Step {step:5d}/{TOTAL_STEPS} | "
                        f"LR {lr:.7f} | "
                        f"Loss {loss_value:.4f} | "
                        f"PPL {perplexity:.2f} | "
                        f"GradNorm {float(grad_norm):.4f} | "
                        f"Speed {step / max(elapsed, 1e-6):.2f} steps/s | "
                        f"GPU0 VRAM {allocated:.2f}/{total:.2f} GB",
                        flush=True,
                    )

        # Wait until both GPUs finish before saving.
        dist.barrier()

        if is_main(rank):
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "n": N,
                "d": D,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "seq_len": SEQ_LEN,
                "batch_size_per_gpu": BATCH_SIZE,
                "world_size": world_size,
                "global_batch_size": BATCH_SIZE * world_size,
                "vocab_size": VOCAB_SIZE,
                "learning_rate": LEARNING_RATE,
                "min_learning_rate": MIN_LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "weight_decay": WEIGHT_DECAY,
                "step": TOTAL_STEPS,
                "dataset": DATASET_NAME,
                "training_mode": "fixed_context_ddp",
                "persistent_state": False,
            }

            torch.save(checkpoint, SAVE_PATH)

            print("=" * 65)
            print("TRAINING COMPLETE")
            print(f"Checkpoint: {SAVE_PATH}")
            print(f"Parameters: {parameter_count:,}")
            print(f"GPUs used: {world_size}")
            print(f"Global batch: {BATCH_SIZE * world_size}")
            print("=" * 65, flush=True)

        dist.barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
