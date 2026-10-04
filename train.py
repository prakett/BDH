import math

import torch
from datasets import load_dataset
from torch import nn

from model import (
    BDHModel,
    count_parameters,
)

# ============================================================
# Configuration
# ============================================================

N = 32768
D = 256

NUM_HEADS = 4
NUM_LAYERS = 4

SEQ_LEN = 256

# 6 GB VRAM: use 8 instead of 16.
BATCH_SIZE = 8

TOTAL_STEPS = 10000

# Paper training regime:
# 1e-3 initial LR, 1000-step warmup,
# linear decay to 1e-4.
LEARNING_RATE_START = 1e-3
LEARNING_RATE_END = 1e-4
WARMUP_STEPS = 1000

WEIGHT_DECAY = 0.1

# ZClip settings from Kumar et al. (2025),
# the adaptive gradient clipping method cited
# by The Dragon Hatchling.
ZCLIP_ALPHA = 0.97
ZCLIP_Z_THRESH = 2.5
ZCLIP_MAX_GRAD_NORM = 1.0
ZCLIP_WARMUP_STEPS = 25

PRINT_EVERY = 10

SAVE_PATH = "bdh_model_paper_train_3.pt"

DATASET_NAME = "roneneldan/TinyStories"

# Keep false during normal training. Set true only when
# diagnosing a numerical problem; it prints tensor statistics.
DEBUG_NUMERICS = False


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("BDH-GPU TRAINING")
print("=" * 70)

print(f"Device:       {device}")

if device.type == "cuda":

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

    # This is a backend performance setting; it does not
    # change the BDH equations.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Model
# ============================================================

model = BDHModel(
    n=N,
    d=D,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(device)

print()
print(f"N:            {N}")
print(f"D:            {D}")
print(f"Heads:        {NUM_HEADS}")
print(f"Layers:       {NUM_LAYERS}")
print(f"Sequence:     {SEQ_LEN}")
print(f"Batch:        {BATCH_SIZE}")
print(f"LR start:     {LEARNING_RATE_START}")
print(f"LR final:     {LEARNING_RATE_END}")
print(f"Warmup:       {WARMUP_STEPS}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Parameters:   {count_parameters(model):,}")


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE_START,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Learning-rate schedule
# ============================================================

def get_learning_rate(step):
    """
    Paper schedule:
      - linear warmup to 1e-3 over 1000 steps
      - linear decay from 1e-3 to 1e-4
        over the remaining training steps
    """

    if step <= WARMUP_STEPS:
        return (
            LEARNING_RATE_START
            * step
            / WARMUP_STEPS
        )

    decay_steps = (
        TOTAL_STEPS
        - WARMUP_STEPS
    )

    progress = (
        step - WARMUP_STEPS
    ) / decay_steps

    return (
        LEARNING_RATE_START
        + progress
        * (
            LEARNING_RATE_END
            - LEARNING_RATE_START
        )
    )


def set_learning_rate(step):
    lr = get_learning_rate(step)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


# ============================================================
# ZClip
# ============================================================
#
# The Dragon Hatchling cites Kumar et al. (2025) for
# adaptive gradient clipping. The implementation below follows
# the official ZClip implementation:
#
#   https://github.com/bluorion-com/ZClip
#
# It tracks the running mean/variance of the total gradient
# norm and clips anomalous spikes adaptively.
# ============================================================

class ZClip:

    def __init__(
        self,
        alpha=0.97,
        z_thresh=2.5,
        max_grad_norm=1.0,
        eps=1e-6,
        warmup_steps=25,
        clip_factor=1.0,
    ):
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.clip_factor = clip_factor

        self.buffer = []
        self.initialized = False
        self.mean = None
        self.var = None

    def _compute_grad_norm(self, model):
        total_norm_sq = 0.0

        for param in model.parameters():

            if param.grad is None:
                continue

            grad = param.grad.detach().float()

            if not torch.isfinite(grad).all():
                return float("nan")

            norm = grad.norm(2).item()
            total_norm_sq += norm * norm

        return math.sqrt(total_norm_sq)

    def _initialize_ema(self):
        self.mean = (
            sum(self.buffer)
            / len(self.buffer)
        )

        self.var = (
            sum(
                (x - self.mean) ** 2
                for x in self.buffer
            )
            / len(self.buffer)
        )

        self.initialized = True
        self.buffer = []

    def _update_ema(self, grad_norm):
        self.mean = (
            self.alpha * self.mean
            + (1.0 - self.alpha) * grad_norm
        )

        self.var = (
            self.alpha * self.var
            + (1.0 - self.alpha)
            * (grad_norm - self.mean) ** 2
        )

    def _clip_value(self, grad_norm):
        std = math.sqrt(
            max(self.var, 0.0)
        )

        z = (
            grad_norm - self.mean
        ) / (
            std + self.eps
        )

        if z > self.z_thresh:

            eta = (
                z / self.z_thresh
            )

            threshold = (
                self.mean
                + (
                    self.z_thresh
                    * std
                ) / eta
            )

            threshold *= self.clip_factor

            return threshold

        return None

    @staticmethod
    def _scale_gradients(model, total_norm, max_norm):
        if total_norm <= max_norm:
            return

        coefficient = (
            max_norm
            / (total_norm + 1e-6)
        )

        for param in model.parameters():

            if param.grad is not None:
                param.grad.mul_(coefficient)

    def step(self, model):
        """
        Apply adaptive clipping after backward and before
        optimizer.step().

        Returns:
            total gradient norm before clipping.
        """

        total_norm = (
            self._compute_grad_norm(model)
        )

        if not math.isfinite(total_norm):
            return total_norm

        # Official ZClip warmup collects gradient statistics.
        # With max_grad_norm=1.0, the official implementation
        # also applies the maximum norm during this phase.
        if not self.initialized:

            self.buffer.append(total_norm)

            if (
                len(self.buffer)
                >= self.warmup_steps
            ):
                self._initialize_ema()

            if (
                self.max_grad_norm
                is not None
            ):
                self._scale_gradients(
                    model,
                    total_norm,
                    self.max_grad_norm,
                )

            return total_norm

        clip_value = self._clip_value(
            total_norm
        )

        effective_clip = (
            clip_value
            if clip_value is not None
            else total_norm
        )

        if (
            self.max_grad_norm
            is not None
        ):
            effective_clip = min(
                effective_clip,
                self.max_grad_norm,
            )

        self._scale_gradients(
            model,
            total_norm,
            effective_clip,
        )

        # Match the official implementation:
        # update EMA using the effective norm when a
        # spike was clipped, otherwise the original norm.
        self._update_ema(
            clip_value
            if clip_value is not None
            else total_norm
        )

        return total_norm


zclip = ZClip(
    alpha=ZCLIP_ALPHA,
    z_thresh=ZCLIP_Z_THRESH,
    max_grad_norm=ZCLIP_MAX_GRAD_NORM,
    warmup_steps=ZCLIP_WARMUP_STEPS,
)


# ============================================================
# Dataset
# ============================================================

print()
print("Loading TinyStories...")

dataset = load_dataset(
    DATASET_NAME,
)

train_dataset = dataset["train"]

print(
    f"Training examples: "
    f"{len(train_dataset):,}"
)


# ============================================================
# Byte stream
# ============================================================

def story_to_bytes(story):

    if isinstance(story, dict):

        text = story.get(
            "text",
            "",
        )

    else:

        text = str(story)

    return text.encode(
        "utf-8",
        errors="replace",
    )


# ============================================================
# Streaming batch generator
# ============================================================

def batch_generator():

    buffers = [
        bytearray()
        for _ in range(BATCH_SIZE)
    ]

    story_index = 0

    while True:

        for batch_item in range(BATCH_SIZE):

            while len(
                buffers[batch_item]
            ) < SEQ_LEN + 1:

                story = train_dataset[
                    story_index
                    % len(train_dataset)
                ]

                story_index += 1

                story_bytes = (
                    story_to_bytes(story)
                )

                buffers[
                    batch_item
                ].extend(
                    story_bytes
                )

                buffers[
                    batch_item
                ].append(
                    ord("\n")
                )

        batch = []

        for i in range(BATCH_SIZE):

            chunk = buffers[i][
                :SEQ_LEN + 1
            ]

            del buffers[i][
                :SEQ_LEN
            ]

            batch.append(
                list(chunk)
            )

        x = torch.tensor(
            [
                row[:-1]
                for row in batch
            ],
            dtype=torch.long,
            device=device,
        )

        y = torch.tensor(
            [
                row[1:]
                for row in batch
            ],
            dtype=torch.long,
            device=device,
        )

        yield x, y


# ============================================================
# Finite checking
# ============================================================

def check_model_parameters():

    for name, param in model.named_parameters():

        if not torch.isfinite(
            param
        ).all():

            return False, name

    return True, None


def check_gradients():

    total_norm_sq = 0.0

    for name, param in model.named_parameters():

        if param.grad is None:
            continue

        if not torch.isfinite(
            param.grad
        ).all():

            return False, name, None

        grad_norm = (
            param.grad.detach()
            .float()
            .norm(2)
            .item()
        )

        total_norm_sq += (
            grad_norm ** 2
        )

    total_norm = math.sqrt(
        total_norm_sq
    )

    return True, None, total_norm


# ============================================================
# VRAM
# ============================================================

def print_vram():

    if device.type != "cuda":
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
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / 1024**3
    )

    print(
        f"VRAM "
        f"{allocated:.2f}/"
        f"{total:.2f} GB "
        f"(reserved {reserved:.2f} GB)"
    )


# ============================================================
# Training
# ============================================================

loader = batch_generator()

model.train()

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

completed_steps = 0

for step in range(
    1,
    TOTAL_STEPS + 1,
):

    x, targets = next(loader)

    optimizer.zero_grad(
        set_to_none=True
    )

    # Set the exact LR for this step before optimizer.step().
    current_lr = set_learning_rate(step)

    try:

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        #
        # Deliberately no AMP here.
        #
        # The Appendix-E model code does not specify AMP, and
        # our previous FP16 run was the numerical-stability
        # experiment that produced the attention overflow.
        #
        # The attention itself remains exactly the raw
        # QK^T causal operation from the Appendix-E structure.
        # ----------------------------------------------------

        logits = model(
            x,
            debug=DEBUG_NUMERICS,
        )

        loss = nn.functional.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            targets.reshape(-1),
        )

        # ----------------------------------------------------
        # Loss check
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            print()
            print("=" * 70)
            print(
                f"NON-FINITE LOSS AT STEP {step}"
            )
            print("=" * 70)
            print(
                f"Loss: {loss.item()}"
            )
            print_vram()
            print(
                "Training stopped before "
                "the optimizer could corrupt the model."
            )
            break

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient check
        # ----------------------------------------------------

        grad_ok, bad_name, raw_grad_norm = (
            check_gradients()
        )

        if not grad_ok:

            print()
            print("=" * 70)
            print(
                f"NON-FINITE GRADIENT AT STEP {step}"
            )
            print("=" * 70)
            print(
                f"Parameter: {bad_name}"
            )
            print_vram()
            print("Training stopped.")
            break

        # ----------------------------------------------------
        # Adaptive gradient clipping
        # ----------------------------------------------------

        clipped_grad_norm = zclip.step(
            model
        )

        if not math.isfinite(
            clipped_grad_norm
        ):

            print()
            print("=" * 70)
            print(
                f"NON-FINITE ZCLIP NORM AT STEP {step}"
            )
            print("=" * 70)
            print_vram()
            print("Training stopped.")
            break

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Check parameters AFTER optimizer
        # ----------------------------------------------------

        params_ok, bad_param = (
            check_model_parameters()
        )

        if not params_ok:

            print()
            print("=" * 70)
            print(
                f"NON-FINITE PARAMETER AT STEP {step}"
            )
            print("=" * 70)
            print(
                f"Parameter: {bad_param}"
            )
            print_vram()
            print("Training stopped.")
            break

        completed_steps = step

    except FloatingPointError as e:

        print()
        print("=" * 70)
        print(
            f"NUMERICAL ERROR AT STEP {step}"
        )
        print("=" * 70)
        print(str(e))
        print_vram()
        print(
            "Training stopped before "
            "the model can be corrupted."
        )
        break

    except RuntimeError as e:

        error_text = str(e)

        if (
            "out of memory"
            in error_text.lower()
        ):

            print()
            print("=" * 70)
            print(
                f"CUDA OUT OF MEMORY AT STEP {step}"
            )
            print("=" * 70)
            print_vram()
            print(
                "Reduce BATCH_SIZE or "
                "SEQ_LEN."
            )

            raise

        raise

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if (
        step % PRINT_EVERY == 0
    ):

        loss_value = loss.item()

        ppl = math.exp(
            min(
                loss_value,
                20.0,
            )
        )

        print(
            f"Step {step:5d} | "
            f"LR {current_lr:.7f} | "
            f"Loss {loss_value:.4f} | "
            f"PPL {ppl:.2f} | "
            f"GradNorm {raw_grad_norm:.4f} | ",
            end="",
        )

        print_vram()


# ============================================================
# Final checkpoint
# ============================================================

params_ok, bad_param = (
    check_model_parameters()
)

if not params_ok:

    print()
    print(
        "Model contains non-finite parameters."
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

        "seq_len":
            SEQ_LEN,

        "vocab_size":
            256,

        "learning_rate_start":
            LEARNING_RATE_START,

        "learning_rate_end":
            LEARNING_RATE_END,

        "warmup_steps":
            WARMUP_STEPS,

        "weight_decay":
            WEIGHT_DECAY,

        "step":
            completed_steps,

        "zclip_mean":
            zclip.mean,

        "zclip_var":
            zclip.var,

        "zclip_initialized":
            zclip.initialized,
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
        f"Path: {SAVE_PATH}"
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        f"Completed steps: "
        f"{completed_steps}"
    )
