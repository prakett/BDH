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

N = 16384
D = 256

NUM_HEADS = 4
NUM_LAYERS = 6

DROPOUT = 0.10

VOCAB_SIZE = 256

# ------------------------------------------------------------
# Paper scaling setup
# ------------------------------------------------------------

SEQ_LEN = 1024

# 1 example x 2048 tokens.
#
# This is the safe starting point for a single 16 GB T4
# or similar GPU.
#
# The recurrent state is carried across minibatches.
# ------------------------------------------------------------

BATCH_SIZE = 4

TOTAL_STEPS = 10000

LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4

WARMUP_STEPS = 1000

WEIGHT_DECAY = 0.1

GRAD_CLIP = 1.0

PRINT_EVERY = 10

SAVE_PATH = (
    "bdh_tinystories_25m_stateful.pt"
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


print("=" * 70)
print("BDH-GPU 25M TINYSTORIES TRAINING")
print("=" * 70)

print(
    f"Device:       {DEVICE}"
)

if DEVICE.type == "cuda":

    print(
        f"GPU:          "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA:         "
        f"{torch.version.cuda}"
    )

    print(
        f"VRAM total:   "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # --------------------------------------------------------
    # TF32 for large matrix multiplications.
    # --------------------------------------------------------

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
).to(DEVICE)


parameter_count = count_parameters(
    model
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
    f"Sequence:        {SEQ_LEN}"
)

print(
    f"Batch:           {BATCH_SIZE}"
)

print(
    f"Dropout:         {DROPOUT}"
)

print(
    f"Parameters:      "
    f"{parameter_count:,}"
)

if parameter_count != 25_296_896:

    print(
        "WARNING: parameter count differs "
        "from the expected 25M configuration."
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
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
    DATASET_NAME,
)

train_dataset = dataset[
    "train"
]

print(
    f"Training examples: "
    f"{len(train_dataset):,}"
)


# ============================================================
# STORY -> UTF-8 BYTES
# ============================================================

def story_to_bytes(
    story
):

    if isinstance(
        story,
        dict,
    ):

        text = story.get(
            "text",
            "",
        )

    else:

        text = str(
            story
        )

    return text.encode(
        "utf-8",
        errors="replace",
    )


# ============================================================
# CONTINUOUS BYTE STREAM
# ============================================================

def build_training_stream():

    stream = bytearray()

    print(
        "Building TinyStories UTF-8 "
        "training stream..."
    )

    for i in range(
        len(train_dataset)
    ):

        story = train_dataset[i]

        data = story_to_bytes(
            story
        )

        if len(data) == 0:
            continue

        stream.extend(
            data
        )

        # Separate stories.
        stream.append(
            10
        )

        if (
            (i + 1) % 100000
            == 0
        ):

            print(
                f"Processed "
                f"{i + 1:,} stories"
            )

    return stream


training_stream = (
    build_training_stream()
)

print(
    f"Training bytes: "
    f"{len(training_stream):,}"
)


# ============================================================
# STREAM POSITION
# ============================================================

stream_position = 0


def get_batch():

    global stream_position

    stream_length = len(
        training_stream
    )

    inputs = []
    targets = []

    for _ in range(
        BATCH_SIZE
    ):

        # ----------------------------------------------------
        # If there isn't enough data remaining, restart the
        # stream.
        #
        # The state is reset by the training loop when this
        # happens because the temporal stream has wrapped.
        # ----------------------------------------------------

        if (
            stream_position
            + SEQ_LEN
            + 1
            > stream_length
        ):

            stream_position = 0

        chunk = training_stream[
            stream_position:
            stream_position
            + SEQ_LEN
            + 1
        ]

        if len(chunk) < (
            SEQ_LEN + 1
        ):

            stream_position = 0

            chunk = training_stream[
                :SEQ_LEN + 1
            ]

        x = list(
            chunk[:-1]
        )

        y = list(
            chunk[1:]
        )

        inputs.append(
            x
        )

        targets.append(
            y
        )

        stream_position += (
            SEQ_LEN
        )

    x = torch.tensor(
        inputs,
        dtype=torch.long,
        device=DEVICE,
    )

    y = torch.tensor(
        targets,
        dtype=torch.long,
        device=DEVICE,
    )

    return x, y


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(
    step
):

    # --------------------------------------------------------
    # Linear warmup:
    #
    # 0 -> 1e-3 over 1000 steps
    # --------------------------------------------------------

    if step <= WARMUP_STEPS:

        return (
            LEARNING_RATE
            * step
            / WARMUP_STEPS
        )

    # --------------------------------------------------------
    # Linear decay:
    #
    # 1e-3 -> 1e-4
    # --------------------------------------------------------

    decay_steps = (
        TOTAL_STEPS
        - WARMUP_STEPS
    )

    progress = (
        step
        - WARMUP_STEPS
    ) / decay_steps

    progress = max(
        0.0,
        min(
            1.0,
            progress,
        ),
    )

    return (
        LEARNING_RATE
        + progress
        * (
            MIN_LEARNING_RATE
            - LEARNING_RATE
        )
    )


def set_learning_rate(
    lr
):

    for group in (
        optimizer.param_groups
    ):

        group["lr"] = lr


# ============================================================
# GRADIENT CHECK
# ============================================================

def check_gradients():

    total_norm_sq = 0.0

    for name, parameter in (
        model.named_parameters()
    ):

        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all():

            return (
                False,
                name,
                None,
            )

        norm = (
            parameter.grad
            .detach()
            .float()
            .norm(2)
            .item()
        )

        total_norm_sq += (
            norm ** 2
        )

    return (
        True,
        None,
        math.sqrt(
            total_norm_sq
        ),
    )


# ============================================================
# PARAMETER CHECK
# ============================================================

def check_parameters():

    for name, parameter in (
        model.named_parameters()
    ):

        if not torch.isfinite(
            parameter
        ).all():

            return (
                False,
                name,
            )

    return (
        True,
        None,
    )


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
        .get_device_properties(0)
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

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()
print(
    "Stateful linear attention: ENABLED"
)

print(
    "TBPTT: ENABLED"
)

print(
    "Attention complexity: "
    "chunked recurrent"
)

print(
    "Persistent state: "
    "8 attention states"
)

print()


# ------------------------------------------------------------
# One persistent state per repeated BDH layer.
# ------------------------------------------------------------

states = [
    None
    for _ in range(
        NUM_LAYERS
    )
]


# ------------------------------------------------------------
# Global position for RoPE.
#
# This advances continuously while the temporal stream is
# continuous.
# ------------------------------------------------------------

position_offset = 0

start_time = time.time()

last_position = stream_position


# ============================================================
# MAIN LOOP
# ============================================================

for step in range(
    1,
    TOTAL_STEPS + 1,
):

    # --------------------------------------------------------
    # Detect stream wrap.
    #
    # If the dataset starts again, the previous temporal state
    # is no longer meaningful.
    # --------------------------------------------------------

    if (
        stream_position
        < last_position
    ):

        states = [
            None
            for _ in range(
                NUM_LAYERS
            )
        ]

        position_offset = 0

        print(
            "\nDataset stream wrapped."
        )

        print(
            "Persistent BDH state reset."
        )

    last_position = (
        stream_position
    )

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    x, targets = get_batch()

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    lr = get_learning_rate(
        step
    )

    set_learning_rate(
        lr
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    try:

        # ====================================================
        # FORWARD
        # ====================================================

        logits, new_states = model(
            x,
            states=states,
            position_offset=position_offset,
        )

        # ====================================================
        # LOSS
        # ====================================================

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            targets.reshape(
                -1
            ),
        )

        if not torch.isfinite(
            loss
        ):

            raise FloatingPointError(
                f"Non-finite loss: "
                f"{loss.item()}"
            )

        # ====================================================
        # BACKWARD
        #
        # Gradients flow through the current 2048-token
        # minibatch.
        # ====================================================

        loss.backward()

        # ====================================================
        # GRADIENT CHECK
        # ====================================================

        (
            grad_ok,
            bad_name,
            grad_norm,
        ) = check_gradients()

        if not grad_ok:

            raise FloatingPointError(
                "Non-finite gradient in "
                f"{bad_name}"
            )

        # ====================================================
        # GRADIENT CLIPPING
        # ====================================================

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer.step()

        # ====================================================
        # PARAMETER CHECK
        # ====================================================

        (
            params_ok,
            bad_parameter,
        ) = check_parameters()

        if not params_ok:

            raise FloatingPointError(
                "Non-finite parameter: "
                f"{bad_parameter}"
            )

        # ====================================================
        # TBPTT
        #
        # IMPORTANT:
        #
        # We update the persistent state AFTER the optimizer
        # step and detach it from the previous minibatch.
        #
        # This gives:
        #
        #     minibatch 1
        #          |
        #          v
        #       state
        #          |
        #        detach
        #          |
        #          v
        #     minibatch 2
        #
        # The model retains the state itself but does not
        # backpropagate indefinitely through the entire corpus.
        # ====================================================

        states = [
            state.detach()
            for state in new_states
        ]

        # ----------------------------------------------------
        # Advance global position.
        # ----------------------------------------------------

        position_offset += (
            SEQ_LEN
        )

    except FloatingPointError as error:

        print()
        print("=" * 70)
        print(
            f"NUMERICAL ERROR AT STEP "
            f"{step}"
        )
        print("=" * 70)

        print(
            str(error)
        )

        print_vram()

        print()
        print(
            "Training stopped."
        )

        raise

    except RuntimeError as error:

        if (
            "out of memory"
            in str(error).lower()
        ):

            print()
            print("=" * 70)
            print(
                f"CUDA OUT OF MEMORY "
                f"AT STEP {step}"
            )
            print("=" * 70)

            print_vram()

            print()
            print(
                "Try reducing BATCH_SIZE."
            )

        raise

    # ========================================================
    # LOGGING
    # ========================================================

    if (
        step % PRINT_EVERY
        == 0
    ):

        loss_value = (
            loss.item()
        )

        perplexity = math.exp(
            min(
                loss_value,
                20.0,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        steps_per_second = (
            step / elapsed
        )

        print(
            f"Step {step:5d} | "
            f"LR {lr:.7f} | "
            f"Loss {loss_value:.4f} | "
            f"PPL {perplexity:.2f} | "
            f"GradNorm {grad_norm:.4f} | "
            f"{steps_per_second:.2f} step/s | ",
            end="",
        )

        print_vram()


# ============================================================
# FINAL CHECKPOINT
# ============================================================

params_ok, bad_parameter = (
    check_parameters()
)

if not params_ok:

    print()
    print(
        "Model contains non-finite "
        "parameters."
    )

    print(
        "NO CHECKPOINT SAVED."
    )

else:

    checkpoint = {

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

        "batch_size":
            BATCH_SIZE,

        "vocab_size":
            VOCAB_SIZE,

        "attention_block_size":
            model.attn.block_size,

        "learning_rate":
            LEARNING_RATE,

        "min_learning_rate":
            MIN_LEARNING_RATE,

        "warmup_steps":
            WARMUP_STEPS,

        "weight_decay":
            WEIGHT_DECAY,

        "step":
            TOTAL_STEPS,

        "dataset":
            DATASET_NAME,

        "tbptt":
            True,

        "stateful_attention":
            True,

        "position_offset":
            position_offset,
    }

    torch.save(
        checkpoint,
        SAVE_PATH,
    )

    print()
    print("=" * 70)
    print("CHECKPOINT SAVED")
    print("=" * 70)

    print(
        f"Path: "
        f"{SAVE_PATH}"
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Steps: "
        f"{TOTAL_STEPS:,}"
    )

    print(
        f"Sequence length: "
        f"{SEQ_LEN}"
    )

    print(
        "Persistent state: "
        "YES"
    )

    print(
        "TBPTT: "
        "YES"
    )