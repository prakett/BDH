import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BDH-GPU CONFIGURATION
# ============================================================

N = 16384
D = 256

NUM_HEADS = 4
NUM_LAYERS = 6

DROPOUT = 0.10

VOCAB_SIZE = 256

# ------------------------------------------------------------
# This is NOT an architectural parameter.
#
# It only controls how the recurrent linear-attention
# computation is evaluated efficiently on the GPU.
#
# Smaller = smaller local attention matrices.
# Larger  = fewer GPU blocks.
# ------------------------------------------------------------

ATTENTION_BLOCK_SIZE = 256


# ============================================================
# ROPE
# ============================================================

def apply_rope(
    x,
    position_offset=0,
):
    """
    Apply rotary positional encoding.

    Input:
        x: [B, H, T, Dh]

    Output:
        [B, H, T, Dh]

    position_offset allows RoPE positions to continue across
    successive TBPTT minibatches.
    """

    B, H, T, Dh = x.shape

    if Dh % 2 != 0:
        raise ValueError(
            "RoPE requires an even head dimension."
        )

    device = x.device
    dtype = x.dtype

    half_dim = Dh // 2

    # --------------------------------------------------------
    # Standard RoPE inverse frequencies
    # --------------------------------------------------------

    inv_freq = (
        1.0
        / (
            10000.0
            ** (
                torch.arange(
                    0,
                    half_dim,
                    device=device,
                    dtype=torch.float32,
                )
                / half_dim
            )
        )
    )

    # --------------------------------------------------------
    # Global positions.
    #
    # position_offset is important when the recurrent state
    # is carried across minibatches.
    # --------------------------------------------------------

    positions = torch.arange(
        position_offset,
        position_offset + T,
        device=device,
        dtype=torch.float32,
    )

    angles = torch.outer(
        positions,
        inv_freq,
    )

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # [T, Dh/2] -> [1, 1, T, Dh/2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )

    return rotated


# ============================================================
# STATEFUL LINEAR ATTENTION
# ============================================================

class LinearAttention(nn.Module):

    def __init__(
        self,
        block_size=ATTENTION_BLOCK_SIZE,
    ):
        super().__init__()

        self.block_size = block_size

    def forward(
        self,
        Q,
        K,
        V,
        state=None,
        position_offset=0,
    ):
        """
        Stateful causal linear attention.

        Q:
            [B, H, T, Dh]

        K:
            [B, H, T, Dh]

        V:
            [B, 1, T, D]

        State:
            [B, H, Dh, D]

        Mathematical recurrence:

            S_t = S_{t-1} + K_t^T V_t

            a_t = Q_t S_{t-1}

        with RoPE applied to Q and K.

        The computation is evaluated in blocks so that we
        never materialize a T x T matrix for T=2048.
        """

        B, H, T, Dh = Q.shape

        # ----------------------------------------------------
        # Attention is deliberately computed in FP32.
        #
        # This is important because BDH's raw linear attention
        # is not softmax-normalized.
        # ----------------------------------------------------

        Q = Q.float()
        K = K.float()
        V = V.float()

        # ----------------------------------------------------
        # V comes from v_ast:
        #
        # [B, 1, T, D]
        #
        # Broadcast it across attention heads.
        # ----------------------------------------------------

        if V.size(1) == 1 and H != 1:
            V = V.expand(
                B,
                H,
                T,
                V.size(-1),
            )

        elif V.size(1) != H:
            raise ValueError(
                "V must have either one head or the same "
                "number of heads as Q/K."
            )

        value_dim = V.size(-1)

        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        Qr = apply_rope(
            Q,
            position_offset=position_offset,
        )

        Kr = apply_rope(
            K,
            position_offset=position_offset,
        )

        # ----------------------------------------------------
        # Initialize persistent recurrent state.
        #
        # S = sum(K^T V)
        #
        # Shape:
        # [B, H, Dh, D]
        # ----------------------------------------------------

        if state is None:

            state = torch.zeros(
                B,
                H,
                Dh,
                value_dim,
                device=Q.device,
                dtype=torch.float32,
            )

        else:

            if state.shape != (
                B,
                H,
                Dh,
                value_dim,
            ):
                raise ValueError(
                    "Invalid attention state shape. "
                    f"Expected {(B, H, Dh, value_dim)}, "
                    f"got {tuple(state.shape)}."
                )

            state = state.float()

        # ----------------------------------------------------
        # Output blocks
        # ----------------------------------------------------

        outputs = []

        block_size = self.block_size

        for start in range(
            0,
            T,
            block_size,
        ):

            end = min(
                start + block_size,
                T,
            )

            Q_block = Qr[
                :,
                :,
                start:end,
                :,
            ]

            K_block = Kr[
                :,
                :,
                start:end,
                :,
            ]

            V_block = V[
                :,
                :,
                start:end,
                :,
            ]

            block_T = end - start

            # =================================================
            # 1. Interaction with ALL previous minibatches and
            #    previous blocks.
            #
            #     Q_block @ state
            #
            # This is the recurrent/state-space component.
            # =================================================

            previous_output = torch.matmul(
                Q_block,
                state,
            )

            # =================================================
            # 2. Causal interaction WITHIN the current block.
            #
            # This is only block_size x block_size.
            #
            # It is mathematically equivalent to:
            #
            #     tril(Q K^T, diagonal=-1) V
            #
            # but we never create a 2048 x 2048 matrix.
            # =================================================

            local_scores = torch.matmul(
                Q_block,
                K_block.transpose(
                    -1,
                    -2,
                ),
            )

            causal_mask = torch.tril(
                torch.ones(
                    block_T,
                    block_T,
                    device=Q.device,
                    dtype=torch.bool,
                ),
                diagonal=-1,
            )

            local_scores = local_scores.masked_fill(
                ~causal_mask,
                0.0,
            )

            local_output = torch.matmul(
                local_scores,
                V_block,
            )

            # =================================================
            # Total causal attention output
            # =================================================

            output_block = (
                previous_output
                + local_output
            )

            outputs.append(
                output_block
            )

            # =================================================
            # 3. Update recurrent state AFTER the block.
            #
            #     S <- S + K^T V
            #
            # This means the current token/block cannot attend
            # to itself through the persistent state.
            # =================================================

            state = state + torch.matmul(
                K_block.transpose(
                    -1,
                    -2,
                ),
                V_block,
            )

        # ----------------------------------------------------
        # Combine blocks
        # ----------------------------------------------------

        output = torch.cat(
            outputs,
            dim=2,
        )

        return output, state


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
        attention_block_size=ATTENTION_BLOCK_SIZE,
    ):
        super().__init__()

        if n % num_heads != 0:
            raise ValueError(
                f"N={n} must be divisible by "
                f"num_heads={num_heads}"
            )

        if d % num_heads != 0:
            raise ValueError(
                f"D={d} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.n = n
        self.d = d

        self.num_heads = num_heads
        self.num_layers = num_layers

        self.dropout_rate = dropout
        self.vocab_size = vocab_size

        self.head_dim = d // num_heads

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
        # BDH-GPU parameters
        #
        # These parameters are SHARED across the 8 repeated
        # BDH layers, as in the Appendix-E implementation.
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
        # Stateful linear attention
        # ----------------------------------------------------

        self.attn = LinearAttention(
            block_size=attention_block_size,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        idx,
        states=None,
        position_offset=0,
        debug=False,
    ):
        """
        idx:
            [B, T]

        states:
            list containing one recurrent attention state
            for every repeated BDH layer.

        Returns:

            logits:
                [B, T, vocab_size]

            new_states:
                list of updated recurrent states
        """

        B, T = idx.size()

        # ----------------------------------------------------
        # Initialize states for all 8 layers.
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
                f"Expected {self.num_layers} states, "
                f"got {len(states)}."
            )

        new_states = []

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        v_ast = self.wte(
            idx
        )

        # [B, T, D]
        v_ast = self.ln(
            v_ast
        )

        # [B, 1, T, D]
        v_ast = v_ast.unsqueeze(1)

        # ----------------------------------------------------
        # BDH layers
        # ----------------------------------------------------

        for layer_idx in range(
            self.num_layers
        ):

            # =================================================
            # X projection
            #
            # [B,1,T,D]
            #
            # ->
            #
            # [B,H,T,N/H]
            # =================================================

            x = torch.matmul(
                v_ast,
                self.decoder_x,
            )

            x = F.relu(
                x
            )

            # =================================================
            # Stateful causal linear attention
            # =================================================

            a_ast, layer_state = self.attn(
                Q=x,
                K=x,
                V=v_ast,
                state=states[layer_idx],
                position_offset=position_offset,
            )

            # Keep state in FP32.
            new_states.append(
                layer_state
            )

            # =================================================
            # Attention normalization
            # =================================================

            a_ast = self.ln(
                a_ast
            )

            # =================================================
            # Y projection
            # =================================================

            y = torch.matmul(
                a_ast,
                self.decoder_y,
            )

            y = F.relu(
                y
            )

            # =================================================
            # Multiplicative interaction
            # =================================================

            y = y * x

            # =================================================
            # Reshape back to N dimension
            # =================================================

            y = y.transpose(
                1,
                2,
            )

            y = y.reshape(
                B,
                1,
                T,
                self.n,
            )

            # =================================================
            # Dropout
            # =================================================

            y = self.drop(
                y
            )

            # =================================================
            # Encoder
            # =================================================

            update = torch.matmul(
                y,
                self.encoder,
            )

            # =================================================
            # Residual update
            # =================================================

            v_ast = (
                v_ast
                + self.ln(
                    update
                )
            )

            v_ast = self.ln(
                v_ast
            )

        # ----------------------------------------------------
        # Readout
        # ----------------------------------------------------

        hidden = v_ast.squeeze(
            1
        )

        logits = torch.matmul(
            hidden,
            self.readout,
        )

        return logits, new_states


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(
    model
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = BDHModel().to(
        device
    )

    print("=" * 70)
    print("BDH-GPU MODEL")
    print("=" * 70)

    print(
        f"N:                 {model.n}"
    )

    print(
        f"D:                 {model.d}"
    )

    print(
        f"Heads:              {model.num_heads}"
    )

    print(
        f"Layers:             {model.num_layers}"
    )

    print(
        f"Head dimension:     {model.head_dim}"
    )

    print(
        f"Attention block:    "
        f"{model.attn.block_size}"
    )

    print(
        f"Parameters:         "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Small forward test
    # --------------------------------------------------------

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
            x,
            states=None,
            position_offset=0,
        )

    print(
        f"Input shape:        "
        f"{tuple(x.shape)}"
    )

    print(
        f"Logits shape:       "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Number of states:   "
        f"{len(states)}"
    )

    print(
        f"State shape:        "
        f"{tuple(states[0].shape)}"
    )

    print("=" * 70)