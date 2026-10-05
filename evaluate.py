import math

import torch
import torch.nn.functional as F
from datasets import load_dataset

from model import BDHModel


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "bdh_tinystories_wikitext2_phase2.pt"

N = 32768
D = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.05

SEQ_LEN = 128
VOCAB_SIZE = 256

EVAL_BATCH_SIZE = 16
EVAL_STEPS = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GPU SETUP
# ============================================================

if DEVICE.type == "cuda":

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("=" * 70)
print("BDH CONTINUAL LEARNING - PHASE 1 EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading Phase 1 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

    N = checkpoint.get("n", N)
    D = checkpoint.get("d", D)

    NUM_HEADS = checkpoint.get(
        "num_heads",
        NUM_HEADS,
    )

    NUM_LAYERS = checkpoint.get(
        "num_layers",
        NUM_LAYERS,
    )

    SEQ_LEN = checkpoint.get(
        "seq_len",
        SEQ_LEN,
    )

    checkpoint_step = checkpoint.get(
        "step",
        "unknown",
    )

    phase = checkpoint.get(
        "phase",
        "unknown",
    )

    dataset_name = checkpoint.get(
        "dataset",
        "unknown",
    )

else:

    state_dict = checkpoint

    checkpoint_step = "unknown"
    phase = "unknown"
    dataset_name = "unknown"


print(
    f"Checkpoint step: {checkpoint_step}"
)

print(
    f"Training phase: {phase}"
)

print(
    f"Training dataset: {dataset_name}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = BDHModel(
    n=N,
    d=D,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    vocab_size=VOCAB_SIZE,
).to(DEVICE)


model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n" + "-" * 70)

print(
    f"Parameters:      {parameters:,}"
)

print(
    f"N:               {N}"
)

print(
    f"D:               {D}"
)

print(
    f"Heads:            {NUM_HEADS}"
)

print(
    f"Layers:           {NUM_LAYERS}"
)

print(
    f"Sequence length:  {SEQ_LEN}"
)


# ============================================================
# LOAD TINYSTORIES VALIDATION DATA
# ============================================================

print(
    "\nLoading TinyStories validation set..."
)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation",
)

print(
    f"Validation examples: "
    f"{len(dataset):,}"
)


# ============================================================
# BUILD VALIDATION BYTE STREAM
# ============================================================

def build_validation_stream(dataset):

    buffer = []

    for example in dataset:

        text = example["text"]

        if not text:
            continue

        # Raw UTF-8 bytes.
        data = list(
            text.encode("utf-8")
        )

        buffer.extend(data)

        # Separate stories.
        buffer.append(10)

    return buffer


print(
    "Building validation byte stream..."
)

validation_stream = (
    build_validation_stream(dataset)
)

print(
    f"Validation bytes: "
    f"{len(validation_stream):,}"
)


# ============================================================
# CREATE VALIDATION BATCHES
# ============================================================

def get_validation_batch(position):

    inputs = []
    targets = []

    stream_length = len(
        validation_stream
    )

    for _ in range(
        EVAL_BATCH_SIZE
    ):

        if (
            position
            + SEQ_LEN
            + 1
            >= stream_length
        ):

            position = 0

        chunk = validation_stream[
            position:
            position + SEQ_LEN + 1
        ]

        x = chunk[:-1]
        y = chunk[1:]

        inputs.append(x)
        targets.append(y)

        position += SEQ_LEN

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

    return x, y, position


# ============================================================
# VALIDATION
# ============================================================

print(
    "\nRunning TinyStories validation..."
)

print(
    "Evaluation precision: FP32"
)

total_loss = 0.0
total_tokens = 0

position = 0

with torch.no_grad():

    for step in range(
        EVAL_STEPS
    ):

        x, targets, position = (
            get_validation_batch(
                position
            )
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            targets.reshape(-1),
        )

        if not torch.isfinite(loss):

            raise FloatingPointError(
                f"Non-finite validation loss "
                f"at evaluation step "
                f"{step + 1}"
            )

        tokens = targets.numel()

        total_loss += (
            loss.item()
            * tokens
        )

        total_tokens += tokens

        if (
            (step + 1) % 50 == 0
        ):

            current_loss = (
                total_loss
                / total_tokens
            )

            current_ppl = math.exp(
                min(
                    current_loss,
                    20.0,
                )
            )

            print(
                f"Eval step {step + 1:4d} | "
                f"Loss {current_loss:.4f} | "
                f"PPL {current_ppl:.2f}"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

average_loss = (
    total_loss
    / total_tokens
)

perplexity = math.exp(
    min(
        average_loss,
        20.0,
    )
)


print("\n" + "=" * 70)
print("PHASE 1 VALIDATION RESULTS")
print("=" * 70)

print(
    f"Checkpoint step:       "
    f"{checkpoint_step}"
)

print(
    f"Validation Loss:       "
    f"{average_loss:.4f}"
)

print(
    f"Validation Perplexity: "
    f"{perplexity:.2f}"
)


# ============================================================
# TEXT GENERATION
# ============================================================

def generate(
    prompt,
    max_new_bytes=300,
    temperature=0.8,
    top_k=50,
):

    model.eval()

    generated = list(
        prompt.encode("utf-8")
    )

    with torch.no_grad():

        for _ in range(
            max_new_bytes
        ):

            # Keep most recent context.
            context = generated[
                -SEQ_LEN:
            ]

            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(x)

            # Last-token prediction.
            next_token_logits = (
                logits[0, -1]
            )

            # Temperature.
            next_token_logits = (
                next_token_logits
                / temperature
            )

            # Top-k sampling.
            if top_k is not None:

                k = min(
                    top_k,
                    next_token_logits.size(-1),
                )

                values, indices = (
                    torch.topk(
                        next_token_logits,
                        k,
                    )
                )

                filtered = (
                    torch.full_like(
                        next_token_logits,
                        float("-inf"),
                    )
                )

                filtered.scatter_(
                    0,
                    indices,
                    values,
                )

                next_token_logits = (
                    filtered
                )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
            )

            if not torch.isfinite(
                probabilities
            ).all():

                raise FloatingPointError(
                    "Non-finite probabilities "
                    "during generation."
                )

            next_byte = (
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                )
                .item()
            )

            generated.append(
                next_byte
            )

    return bytes(
        generated
    ).decode(
        "utf-8",
        errors="replace",
    )


# ============================================================
# GENERATION TESTS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 TEXT GENERATION")
print("=" * 70)

prompts = [
    "Once upon a time",
    "The little girl",
    "One day, a dragon",
    "There was a",
]


for prompt in prompts:

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"PROMPT: {prompt}"
    )

    print(
        "-" * 70
    )

    generated_text = generate(
        prompt,
        max_new_bytes=300,
        temperature=0.8,
        top_k=50,
    )

    print(
        generated_text
    )


# ============================================================
# GPU MEMORY
# ============================================================

if DEVICE.type == "cuda":

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print(
        "\n"
        + "=" * 70
    )

    print("GPU MEMORY")

    print("=" * 70)

    print(
        f"Allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"Reserved:  "
        f"{reserved:.2f} GB"
    )