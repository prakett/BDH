import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

from model import BDHModel, count_parameters

N, D = 32768, 256
NUM_HEADS, NUM_LAYERS = 4, 4
SEQ_LEN = 128
BATCH_SIZE_PER_GPU = 8
TOTAL_STEPS = 10000
LR_START, LR_END = 1e-3, 1e-4
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
PRINT_EVERY = 10
SAVE_PATH = "bdh_model_ddp.pt"
DATASET_NAME = "roneneldan/TinyStories"

# Stable-training adaptive clipping settings used in the 10k run.
ZCLIP_ALPHA = 0.97
ZCLIP_Z_THRESHOLD = 2.5
ZCLIP_MAX_GRAD_NORM = 1.0
ZCLIP_EPS = 1e-6
ZCLIP_WARMUP_STEPS = 25


def setup():
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with: torchrun --standalone --nproc_per_node=2 train.py")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, local_rank, world


rank, local_rank, world = setup()
device = torch.device("cuda", local_rank)
is_main = rank == 0

# Performance settings; these do not change the BDH equations.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def log(*args, **kwargs):
    if is_main:
        print(*args, **kwargs)


log("=" * 72)
log("BDH-GPU - 2x GPU DistributedDataParallel")
log("=" * 72)
log(f"World size: {world}")
log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
log(f"Per-GPU batch: {BATCH_SIZE_PER_GPU}")
log(f"Effective batch: {BATCH_SIZE_PER_GPU * world}")
log(f"N={N}, D={D}, heads={NUM_HEADS}, layers={NUM_LAYERS}, seq={SEQ_LEN}")

model = BDHModel(n=N, d=D, num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
log(f"Parameters: {count_parameters(model):,}")
model = DDP(model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)


class AdaptiveZClip:
    def __init__(self):
        self.mean = None
        self.var = 0.0
        self.steps = 0

    @torch.no_grad()
    def clip(self, parameters, norm):
        self.steps += 1
        if self.mean is None:
            self.mean = norm
        else:
            delta = norm - self.mean
            self.mean = ZCLIP_ALPHA * self.mean + (1 - ZCLIP_ALPHA) * norm
            self.var = ZCLIP_ALPHA * self.var + (1 - ZCLIP_ALPHA) * delta * delta
        std = math.sqrt(max(self.var, 0.0) + ZCLIP_EPS)
        z = (norm - self.mean) / std
        do_clip = self.steps <= ZCLIP_WARMUP_STEPS or z > ZCLIP_Z_THRESHOLD
        if do_clip:
            torch.nn.utils.clip_grad_norm_(parameters, ZCLIP_MAX_GRAD_NORM)
        return z, do_clip


zclip = AdaptiveZClip()

log("Loading TinyStories...")
dataset = load_dataset(DATASET_NAME)
train_dataset = dataset["train"]
log(f"Training examples: {len(train_dataset):,}")


def story_bytes(story):
    text = story.get("text", "") if isinstance(story, dict) else str(story)
    return text.encode("utf-8", errors="replace")


def batch_generator():
    buffers = [bytearray() for _ in range(BATCH_SIZE_PER_GPU)]
    story_index = rank * BATCH_SIZE_PER_GPU
    stride = world * BATCH_SIZE_PER_GPU
    while True:
        for i in range(BATCH_SIZE_PER_GPU):
            while len(buffers[i]) < SEQ_LEN + 1:
                buffers[i].extend(story_bytes(train_dataset[story_index % len(train_dataset)]))
                buffers[i].append(10)
                story_index += stride
        rows = []
        for i in range(BATCH_SIZE_PER_GPU):
            chunk = buffers[i][:SEQ_LEN + 1]
            del buffers[i][:SEQ_LEN]
            rows.append(list(chunk))
        x = torch.tensor([r[:-1] for r in rows], dtype=torch.long, device=device)
        y = torch.tensor([r[1:] for r in rows], dtype=torch.long, device=device)
        yield x, y


def grad_norm():
    total = 0.0
    for p in model.module.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return None
        n = p.grad.detach().float().norm(2).item()
        total += n * n
    return math.sqrt(total)


def set_lr(step):
    if step <= WARMUP_STEPS:
        lr = LR_START * step / WARMUP_STEPS
    else:
        progress = min(1.0, (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS))
        lr = LR_START + (LR_END - LR_START) * progress
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def vram():
    a = torch.cuda.memory_allocated(device) / 1024**3
    r = torch.cuda.memory_reserved(device) / 1024**3
    t = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"VRAM {a:.2f}/{t:.2f} GB (reserved {r:.2f} GB)")


loader = batch_generator()
model.train()
dist.barrier()
start = last_log = time.perf_counter()
last_step = 0

try:
    for step in range(1, TOTAL_STEPS + 1):
        last_step = step
        lr = set_lr(step)
        x, targets = next(loader)
        optimizer.zero_grad(set_to_none=True)

        # FP32 intentionally retained for the first DDP benchmark.
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        gn = grad_norm()
        if gn is None:
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        z, clipped = zclip.clip(model.module.parameters(), gn)
        optimizer.step()

        if step % PRINT_EVERY == 0 and is_main:
            now = time.perf_counter()
            sps = PRINT_EVERY / max(now - last_log, 1e-9)
            last_log = now
            lv = loss.item()
            print(f"Step {step:5d} | LR {lr:.7f} | Loss {lv:.4f} | PPL {math.exp(min(lv,20)):.2f} | GradNorm {gn:.4f} | Z {z:.2f} | Clip {clipped} | {sps:.2f} step/s | ", end="")
            vram()

    dist.barrier()
    if is_main:
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "n": N, "d": D, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
            "seq_len": SEQ_LEN, "vocab_size": 256,
            "learning_rate_start": LR_START, "learning_rate_end": LR_END,
            "warmup_steps": WARMUP_STEPS, "weight_decay": WEIGHT_DECAY,
            "step": last_step, "world_size": world,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "effective_batch_size": BATCH_SIZE_PER_GPU * world,
        }, SAVE_PATH)
        print(f"Checkpoint saved: {SAVE_PATH}")
        print(f"Elapsed: {(time.perf_counter() - start)/60:.2f} min")
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
