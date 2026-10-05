import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

N = 32768
D = 256
NUM_HEADS = 4
NUM_LAYERS = 8
DROPOUT = 0.10
VOCAB_SIZE = 256


# ============================================================
# RoPE
# ============================================================

def apply_rope(x, positions):
    """
    Apply RoPE along the neuron/head dimension.

    x:
        [B, H, T, Dh]

    positions:
        [T]

    Returns:
        [B, H, T, Dh]
    """

    B, H, T, Dh = x.shape

    if Dh % 2 != 0:
        raise ValueError(
            f"RoPE requires an even dimension, got {Dh}"
        )

    device = x.device
    dtype = x.dtype

    half = Dh // 2

    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(
                0,
                half,
                device=device,
                dtype=torch.float32,
            )
            / half
        )
    )

    positions = positions.to(
        device=device,
        dtype=torch.float32,
    )

    angles = torch.outer(
        positions,
        inv_freq,
    )

    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]

    return torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )


# ============================================================
# STATEFUL LINEAR ATTENTION
# ============================================================

class LinearAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q,
        K,
        V,
        state=None,
        positions=None,
    ):
        """
        Stateful causal linear attention.

        Q:
            [B, H, T, Dh]

        K:
            [B, H, T, Dh]

        V:
            [B, H, T, D]

        state:
            [B, H, Dh, D]

        Returns:
            output:
                [B, H, T, D]

            new_state:
                [B, H, Dh, D]
        """

        Q = Q.float()
        K = K.float()
        V = V.float()

        B, H, T, Dh = Q.shape
        Dv = V.size(-1)

        if positions is None:
            positions = torch.arange(
                T,
                device=Q.device,
                dtype=torch.long,
            )

        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        Qr = apply_rope(
            Q,
            positions,
        )

        Kr = apply_rope(
            K,
            positions,
        )

        # ----------------------------------------------------
        # Previous recurrent state
        #
        # state =
        # sum(previous K_r^T @ V)
        # ----------------------------------------------------

        if state is None:

            state = torch.zeros(
                B,
                H,
                Dh,
                Dv,
                device=Q.device,
                dtype=torch.float32,
            )

        else:

            state = state.float()

        # ----------------------------------------------------
        # Contribution from previous minibatches
        # ----------------------------------------------------

        previous_output = (
            Qr @ state
        )

        # ----------------------------------------------------
        # Causal attention INSIDE current minibatch
        #
        # Important:
        # diagonal=-1 excludes current token.
        # ----------------------------------------------------

        scores = (
            Qr
            @ Kr.transpose(-1, -2)
        )

        scores = torch.tril(
            scores,
            diagonal=-1,
        )

        local_output = (
            scores @ V
        )

        output = (
            previous_output
            + local_output
        )

        # ----------------------------------------------------
        # Update recurrent state AFTER producing outputs.
        #
        # Therefore the current token cannot attend to itself.
        # ----------------------------------------------------

        current_state = (
            Kr.transpose(-1, -2)
            @ V
        )

        new_state = (
            state
            + current_state
        )

        return output, new_state


# ============================================================
# BDH-GPU
# ============================================================

class BDHModel(nn.Module):

    def __init__(
        self,
        n=N,
        d=D,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        vocab_size=VOCAB_SIZE,
    ):

        super().__init__()

        if n % num_heads != 0:
            raise ValueError(
                f"N={n} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.n = n
        self.d = d
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.vocab_size = vocab_size

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        self.ln = nn.LayerNorm(
            d,
            elementwise_affine=False,
            bias=False,
        )

        # ----------------------------------------------------
        # Byte embedding
        # ----------------------------------------------------

        self.wte = nn.Embedding(
            vocab_size,
            d,
        )

        self.drop = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # BDH parameters
        #
        # Same parameters are reused through all L layers,
        # matching the Appendix-E style architecture.
        # ----------------------------------------------------

        self.encoder = nn.Parameter(
            torch.zeros(
                n,
                d,
            ).normal_(
                std=0.02
            )
        )

        self.decoder_x = nn.Parameter(
            torch.zeros(
                num_heads,
                d,
                n // num_heads,
            ).normal_(
                std=0.02
            )
        )

        self.decoder_y = nn.Parameter(
            torch.zeros(
                num_heads,
                d,
                n // num_heads,
            ).normal_(
                std=0.02
            )
        )

        # ----------------------------------------------------
        # Vocabulary readout
        # ----------------------------------------------------

        self.readout = nn.Parameter(
            torch.zeros(
                d,
                vocab_size,
            ).normal_(
                std=0.02
            )
        )

        # ----------------------------------------------------
        # Stateful attention
        # ----------------------------------------------------

        self.attn = LinearAttention()

    # ========================================================
    # FINITE CHECK
    # ========================================================

    @staticmethod
    def check_finite(
        x,
        name,
    ):

        if not torch.isfinite(
            x
        ).all():

            finite_ratio = (
                torch.isfinite(x)
                .float()
                .mean()
                .item()
            )

            finite_values = (
                x.detach()
                .float()
                .abs()
            )

            finite_values = (
                finite_values[
                    torch.isfinite(
                        finite_values
                    )
                ]
            )

            if finite_values.numel() > 0:
                max_value = (
                    finite_values.max()
                    .item()
                )
            else:
                max_value = float("nan")

            raise FloatingPointError(
                f"Non-finite tensor detected: "
                f"{name} | "
                f"finite={finite_ratio:.6f} | "
                f"max_abs={max_value:.6e}"
            )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        idx,
        states=None,
        position_offset=0,
        debug=False,
    ):

        B, T = idx.size()

        # ----------------------------------------------------
        # Position indices
        # ----------------------------------------------------

        positions = (
            torch.arange(
                position_offset,
                position_offset + T,
                device=idx.device,
                dtype=torch.long,
            )
        )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        v_ast = self.wte(
            idx
        )

        if debug:
            self.check_finite(
                v_ast,
                "embedding",
            )

        # [B, 1, T, D]

        v_ast = (
            v_ast.unsqueeze(1)
        )

        v_ast = self.ln(
            v_ast
        )

        if debug:
            self.check_finite(
                v_ast,
                "initial_layernorm",
            )

        # ----------------------------------------------------
        # Initialize states
        # ----------------------------------------------------

        if states is None:

            states = [
                None
                for _ in range(
                    self.num_layers
                )
            ]

        if len(states) != self.num_layers:

            raise ValueError(
                f"Expected "
                f"{self.num_layers} states, "
                f"got {len(states)}"
            )

        new_states = []

        # ----------------------------------------------------
        # BDH layers
        # ----------------------------------------------------

        for layer_idx in range(
            self.num_layers
        ):

            # ------------------------------------------------
            # X projection
            # ------------------------------------------------

            x = torch.matmul(
                v_ast,
                self.decoder_x,
            )

            if debug:
                self.check_finite(
                    x,
                    f"layer_{layer_idx}_decoder_x",
                )

            x = F.relu(
                x
            )

            if debug:
                self.check_finite(
                    x,
                    f"layer_{layer_idx}_relu_x",
                )

            # ------------------------------------------------
            # Stateful linear attention
            # ------------------------------------------------

            a_ast, new_state = (
                self.attn(
                    Q=x,
                    K=x,
                    V=v_ast,
                    state=states[layer_idx],
                    positions=positions,
                )
            )

            if debug:
                self.check_finite(
                    a_ast,
                    f"layer_{layer_idx}_attention",
                )

                self.check_finite(
                    new_state,
                    f"layer_{layer_idx}_state",
                )

            new_states.append(
                new_state
            )

            # ------------------------------------------------
            # Y projection
            # ------------------------------------------------

            y = torch.matmul(
                self.ln(a_ast),
                self.decoder_y,
            )

            y = F.relu(
                y
            )

            if debug:
                self.check_finite(
                    y,
                    f"layer_{layer_idx}_relu_y",
                )

            # ------------------------------------------------
            # Multiplicative interaction
            # ------------------------------------------------

            y = y * x

            if debug:
                self.check_finite(
                    y,
                    f"layer_{layer_idx}_y_times_x",
                )

            # ------------------------------------------------
            # [B,H,T,N/H]
            # ->
            # [B,1,T,N]
            # ------------------------------------------------

            y = (
                y.transpose(
                    1,
                    2,
                )
                .reshape(
                    B,
                    1,
                    T,
                    self.n,
                )
            )

            # ------------------------------------------------
            # Dropout
            # ------------------------------------------------

            y = self.drop(
                y
            )

            # ------------------------------------------------
            # Encoder
            # ------------------------------------------------

            v_ast = (
                v_ast
                + self.ln(
                    y @ self.encoder
                )
            )

            v_ast = self.ln(
                v_ast
            )

            if debug:
                self.check_finite(
                    v_ast,
                    f"layer_{layer_idx}_output",
                )

        # ----------------------------------------------------
        # Readout
        # ----------------------------------------------------

        logits = (
            v_ast.squeeze(1)
            @ self.readout
        )

        if debug:
            self.check_finite(
                logits,
                "logits",
            )

        return (
            logits,
            new_states,
        )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model,
):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = BDHModel().to(
        device
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    x = torch.randint(
        0,
        256,
        (
            1,
            128,
        ),
        device=device,
    )

    with torch.no_grad():

        logits, states = model(
            x
        )

    print(
        f"Input:  {x.shape}"
    )

    print(
        f"Logits: {logits.shape}"
    )

    print(
        f"States: {len(states)}"
    )