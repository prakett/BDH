import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

from model import (
    BDHModel,
    count_parameters,
)

# ============================================================
# CONFIGURATION
# ============================================================

N = 32768
D = 256

NUM_HEADS = 4
NUM_LAYERS = 8

DROPOUT = 0.10

VOCAB_SIZE = 256

# ------------------------------------------------------------
# Paper uses 2048-token minibatches.
#
# With 25M BDH and 14-16 GB GPUs, start with batch=1.
# ------------------------------------------------------------

SEQ_LEN = 2048
BATCH_SIZE = 1

# ------------------------------------------------------------
# Number of optimization steps.
#
# Keep this configurable. The paper's large experiments use
# much larger token exposure; this is a practical first run.
# ------------------------------------------------------------

TOTAL_STEPS = 10000

# ------------------------------------------------------------
# Paper training schedule
# ------------------------------------------------------------

LR_START = 1e-3
LR_END = 1e-4

WARMUP_STEPS = 1000

WEIGHT_DECAY = 0.1

# Gradient clipping.
#
# The paper specifies adaptive gradient clipping. We keep
# a conservative finite global clip here for this first
# implementation rather than silently inventing a different
# adaptive-clipping algorithm.
# ------------------------------------------------------------

GRAD_CLIP = 1.0

PRINT_EVERY = 10

SAVE_PATH = (
    "bdh_tinystories_25m_2048.pt"
)

DATASET_NAME = (
    "roneneldan/TinyStories"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "=" * 72
)

print(
    "BDH-GPU 25M TINYSTORIES TRAINING"
)

print(
    "Paper-aligned 8-layer / 2048-token / TBPTT setup"
)

print(
    "=" * 72
)

print(
    f"Device: {DEVICE}"
)

if DEVICE.type == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# MODEL
# ============================================================

model = BDHModel(
    n=N,
    d=D,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    vocab_size=VOCAB_SIZE,
).to(
    DEVICE
)

print()

print(
    f"N:              {N}"
)

print(
    f"D:              {D}"
)

print(
    f"Heads:           {NUM_HEADS}"
)

print(
    f"Layers:          {NUM_LAYERS}"
)

print(
    f"Dropout:         {DROPOUT}"
)

print(
    f"Sequence length: {SEQ_LEN}"
)

print(
    f"Batch size:      {BATCH_SIZE}"
)

print(
    f"Parameters:      "
    f"{count_parameters(model):,}"
)

expected_parameters = 25296896

if count_parameters(model) != expected_parameters:

    print()

    print(
        "WARNING:"
    )

    print(
        f"Expected approximately "
        f"{expected_parameters:,} parameters "
        f"for this configuration."
    )

    print(
        f"Actual: "
        f"{count_parameters(model):,}"
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR_START,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# DATASET
# ============================================================

print()

print(
    "Loading TinyStories..."
)

dataset = load_dataset(
    DATASET_NAME
)

train_dataset = dataset[
    "train"
]

print(
    f"Training examples: "
    f"{len(train_dataset):,}"
)


# ============================================================
# BUILD RAW UTF-8 BYTE CORPUS
# ============================================================

print()

print(
    "Building TinyStories byte corpus..."
)

corpus = bytearray()

for example in train_dataset:

    text = example[
        "text"
    ]

    if not text:
        continue

    encoded = text.encode(
        "utf-8",
        errors="replace",
    )

    corpus.extend(
        encoded
    )

    # Separate stories.
    corpus.append(
        10
    )


corpus = bytes(
    corpus
)

print(
    f"Corpus size: "
    f"{len(corpus):,} bytes"
)


if len(corpus) <= (
    SEQ_LEN + 1
):

    raise RuntimeError(
        "Corpus is too small "
        "for the selected sequence length."
    )


# ============================================================
# SEQUENTIAL DATA GENERATOR
# ============================================================
#
# IMPORTANT:
#
# We intentionally do NOT randomly sample independent
# 2048-token chunks.
#
# The recurrent state is carried from one minibatch to the
# next, so the data must also be temporally sequential.
#
# This gives:
#
# chunk 1 -> state 1
# chunk 2 -> state 2
# chunk 3 -> state 3
#
# ============================================================

def batch_generator():

    position = 0

    corpus_length = len(
        corpus
    )

    while True:

        batch_x = []
        batch_y = []

        for _ in range(
            BATCH_SIZE
        ):

            # ------------------------------------------------
            # Wrap around corpus.
            # ------------------------------------------------

            if (
                position
                + SEQ_LEN
                + 1
                > corpus_length
            ):

                position = 0

            chunk = corpus[
                position:
                position
                + SEQ_LEN
                + 1
            ]

            x = torch.tensor(
                list(
                    chunk[:-1]
                ),
                dtype=torch.long,
            )

            y = torch.tensor(
                list(
                    chunk[1:]
                ),
                dtype=torch.long,
            )

            batch_x.append(
                x
            )

            batch_y.append(
                y
            )

            position += (
                SEQ_LEN
            )

        x = torch.stack(
            batch_x
        ).to(
            DEVICE
        )

        y = torch.stack(
            batch_y
        ).to(
            DEVICE
        )

        yield x, y


# ============================================================
# LEARNING RATE SCHEDULE
# ============================================================

def set_lr(step):

    if step <= WARMUP_STEPS:

        lr = (
            LR_START
            * step
            / WARMUP_STEPS
        )

    else:

        progress = min(
            1.0,
            (
                step
                - WARMUP_STEPS
            )
            / max(
                1,
                TOTAL_STEPS
                - WARMUP_STEPS,
            ),
        )

        lr = (
            LR_START
            + (
                LR_END
                - LR_START
            )
            * progress
        )

    for group in (
        optimizer.param_groups
    ):

        group[
            "lr"
        ] = lr

    return lr


# ============================================================
# STATE DETACH
# ============================================================
#
# This is the TBPTT boundary.
#
# The recurrent state is preserved numerically, but its
# previous computation graph is detached before the next
# 2048-token minibatch.
#
# ============================================================

def detach_states(
    states
):

    if states is None:
        return None

    return [
        state.detach()
        for state in states
    ]


# ============================================================
# VRAM
# ============================================================

def print_vram():

    if DEVICE.type != "cuda":
        return

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    total = (
        torch.cuda
        .get_device_properties(
            DEVICE
        )
        .total_memory
        / 1024**3
    )

    print(
        f"VRAM "
        f"{allocated:.2f}/"
        f"{total:.2f} GB "
        f"(reserved "
        f"{reserved:.2f} GB)"
    )


# ============================================================
# TRAINING
# ============================================================

loader = batch_generator()

model.train()

states = None

position_offset = 0

start_time = time.perf_counter()

last_log_time = (
    start_time
)

print()

print(
    "=" * 72
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 72
)

print()

print(
    "Dataset: TinyStories"
)

print(
    "Encoding: raw UTF-8 bytes"
)

print(
    "Architecture: BDH-GPU"
)

print(
    f"Layers: {NUM_LAYERS}"
)

print(
    f"Minibatch length: "
    f"{SEQ_LEN} tokens"
)

print(
    "Attention state: persistent"
)

print(
    "Training: truncated BPTT"
)

print(
    f"Total steps: "
    f"{TOTAL_STEPS}"
)

print(
    f"Warmup steps: "
    f"{WARMUP_STEPS}"
)

print(
    "-" * 72
)


for step in range(
    1,
    TOTAL_STEPS + 1,
):

    lr = set_lr(
        step
    )

    x, targets = next(
        loader
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    logits, states = model(
        x,
        states=states,
        position_offset=position_offset,
        debug=False,
    )

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1),
        ),
        targets.reshape(
            -1
        ),
    )

    if not torch.isfinite(
        loss
    ):

        raise FloatingPointError(
            f"Non-finite loss at "
            f"step {step}: "
            f"{loss.item()}"
        )

    # --------------------------------------------------------
    # BACKWARD
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # GRADIENT CLIPPING
    # --------------------------------------------------------

    grad_norm = (
        torch.nn.utils
        .clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP,
        )
    )

    if not torch.isfinite(
        grad_norm
    ):

        raise FloatingPointError(
            f"Non-finite gradient "
            f"at step {step}"
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # TBPTT
    #
    # Carry the state forward, but stop gradients from
    # propagating indefinitely through previous minibatches.
    # --------------------------------------------------------

    states = detach_states(
        states
    )

    position_offset += (
        SEQ_LEN
    )

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if step % PRINT_EVERY == 0:

        now = time.perf_counter()

        steps_per_sec = (
            PRINT_EVERY
            / max(
                now
                - last_log_time,
                1e-9,
            )
        )

        last_log_time = now

        loss_value = (
            loss.item()
        )

        perplexity = math.exp(
            min(
                loss_value,
                20.0,
            )
        )

        print(
            f"Step {step:5d} | "
            f"LR {lr:.7f} | "
            f"Loss {loss_value:.4f} | "
            f"PPL {perplexity:.2f} | "
            f"GradNorm "
            f"{grad_norm:.4f} | "
            f"{steps_per_sec:.2f} "
            f"step/s | ",
            end="",
        )

        print_vram()


# ============================================================
# SAVE CHECKPOINT
# ============================================================

elapsed_minutes = (
    time.perf_counter()
    - start_time
) / 60.0


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "n":
            N,

        "d":
            D,

        "num_heads":
            NUM_HEADS,

        "num_layers":
            NUM_LAYERS,

        "dropout":
            DROPOUT,

        "seq_len":
            SEQ_LEN,

        "vocab_size":
            VOCAB_SIZE,

        "step":
            TOTAL_STEPS,

        "dataset":
            DATASET_NAME,

        "learning_rate_start":
            LR_START,

        "learning_rate_end":
            LR_END,

        "warmup_steps":
            WARMUP_STEPS,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "tbptt":
            True,

        "persistent_attention_state":
            True,

        "training_precision":
            "FP32",
    },
    SAVE_PATH,
)


print()

print(
    "=" * 72
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 72
)

print(
    f"Checkpoint: "
    f"{SAVE_PATH}"
)

print(
    f"Parameters: "
    f"{count_parameters(model):,}"
)

print(
    f"Layers: "
    f"{NUM_LAYERS}"
)

print(
    f"Sequence length: "
    f"{SEQ_LEN}"
)

print(
    "Persistent attention state: True"
)

print(
    "TBPTT: True"
)

print(
    f"Steps: "
    f"{TOTAL_STEPS}"
)

print(
    f"Elapsed time: "
    f"{elapsed_minutes:.2f} minutes"
)

print(
    "=" * 72
)