import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Configuration
# ============================================================

N = 32768
D = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.05
VOCAB_SIZE = 256


# ============================================================
# RoPE
# ============================================================

def apply_rope(x):
    """
    x:
        [B, H, T, D_head]

    Returns:
        [B, H, T, D_head]
    """

    B, H, T, Dh = x.shape

    if Dh % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head dimension, got {Dh}"
        )

    device = x.device
    dtype = x.dtype

    half = Dh // 2

    positions = torch.arange(
        T,
        device=device,
        dtype=torch.float32,
    )

    inv_freq = 1.0 / (
        10000.0 ** (
            torch.arange(
                0,
                half,
                device=device,
                dtype=torch.float32,
            ) / half
        )
    )

    angles = torch.outer(positions, inv_freq)

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
# Linear Attention
# ============================================================

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, debug=False):
        # Appendix-E-style raw attention.
        # No 1/sqrt(d) scaling and no softmax.
        Q = Q.float()
        K = K.float()
        V = V.float()

        Qr = apply_rope(Q)
        Kr = apply_rope(K)

        scores = Qr @ Kr.transpose(-1, -2)

        # Causal attention: current token cannot attend to itself
        # or future tokens.
        scores = torch.tril(
            scores,
            diagonal=-1,
        )

        output = scores @ V

        if debug:
            def stats(name, x):
                print(
                    f"[ATTN] {name}: "
                    f"min={x.min().item():.4e}, "
                    f"max={x.max().item():.4e}, "
                    f"mean={x.mean().item():.4e}, "
                    f"std={x.std().item():.4e}, "
                    f"finite={torch.isfinite(x).all().item()}"
                )

            stats("Q", Qr)
            stats("K", Kr)
            stats("V", V)
            stats("scores", scores)
            stats("output", output)

        if not torch.isfinite(output).all():
            raise FloatingPointError(
                "Non-finite values detected in LinearAttention output"
            )

        return output


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
                f"N={n} must be divisible by num_heads={num_heads}"
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

        self.drop = nn.Dropout(dropout)

        # ----------------------------------------------------
        # BDH-GPU parameters
        #
        # encoder:
        #     [N, D]
        #
        # decoder_x:
        #     [H, D, N/H]
        #
        # decoder_y:
        #     [H, D, N/H]
        # ----------------------------------------------------

        self.encoder = nn.Parameter(
            torch.zeros(n, d).normal_(std=0.02)
        )

        self.decoder_x = nn.Parameter(
            torch.zeros(
                num_heads,
                d,
                n // num_heads,
            ).normal_(std=0.02)
        )

        self.decoder_y = nn.Parameter(
            torch.zeros(
                num_heads,
                d,
                n // num_heads,
            ).normal_(std=0.02)
        )

        # ----------------------------------------------------
        # Final vocabulary readout
        # ----------------------------------------------------

        self.readout = nn.Parameter(
            torch.zeros(
                d,
                vocab_size,
            ).normal_(std=0.02)
        )

        self.attn = LinearAttention()

    # ========================================================
    # Finite check helper
    # ========================================================

    @staticmethod
    def check_finite(x, name):
        if not torch.isfinite(x).all():
            finite_ratio = (
                torch.isfinite(x).float().mean().item()
            )

            x_abs = x.detach().float().abs()
            finite_values = x_abs[
                torch.isfinite(x_abs)
            ]

            if finite_values.numel() > 0:
                max_value = finite_values.max().item()
            else:
                max_value = float("nan")

            raise FloatingPointError(
                f"Non-finite tensor detected: {name} | "
                f"finite={finite_ratio:.6f} | "
                f"max_abs={max_value:.6e}"
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, idx, debug=False):

        B, T = idx.size()

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        v_ast = self.wte(idx)

        if debug:
            self.check_finite(
                v_ast,
                "embedding",
            )

        # [B, 1, T, D]
        v_ast = v_ast.unsqueeze(1)
        v_ast = self.ln(v_ast)

        if debug:
            self.check_finite(
                v_ast,
                "initial_layernorm",
            )

        # ----------------------------------------------------
        # BDH layers
        # ----------------------------------------------------

        for layer_idx in range(self.num_layers):

            # X projection
            x = torch.matmul(
                v_ast,
                self.decoder_x,
            )

            if debug:
                self.check_finite(
                    x,
                    f"layer_{layer_idx}_decoder_x",
                )

            x = F.relu(x)

            if debug:
                self.check_finite(
                    x,
                    f"layer_{layer_idx}_relu_x",
                )

            # Linear attention
            a_ast = self.attn(
                Q=x,
                K=x,
                V=v_ast,
                debug=debug,
            )

            if debug:
                self.check_finite(
                    a_ast,
                    f"layer_{layer_idx}_attention",
                )

            # Attention normalization
            a_ast = self.ln(a_ast)

            if debug:
                self.check_finite(
                    a_ast,
                    f"layer_{layer_idx}_attention_norm",
                )

            # Y projection
            y = torch.matmul(
                a_ast,
                self.decoder_y,
            )

            if debug:
                self.check_finite(
                    y,
                    f"layer_{layer_idx}_decoder_y",
                )

            y = F.relu(y)

            if debug:
                self.check_finite(
                    y,
                    f"layer_{layer_idx}_relu_y",
                )

            # Multiplicative interaction
            y = y * x

            if debug:
                self.check_finite(
                    y,
                    f"layer_{layer_idx}_y_times_x",
                )

            # Reshape back to neuron dimension N
            y = y.transpose(1, 2)

            y = y.reshape(
                B,
                1,
                T,
                self.n,
            )

            if debug:
                self.check_finite(
                    y,
                    f"layer_{layer_idx}_reshape",
                )

            # Dropout
            y = self.drop(y)

            # Encoder projection
            update = torch.matmul(
                y,
                self.encoder,
            )

            if debug:
                self.check_finite(
                    update,
                    f"layer_{layer_idx}_encoder_update",
                )

            # Residual update
            v_ast = v_ast + self.ln(update)

            if debug:
                self.check_finite(
                    v_ast,
                    f"layer_{layer_idx}_residual",
                )

            v_ast = self.ln(v_ast)

            if debug:
                self.check_finite(
                    v_ast,
                    f"layer_{layer_idx}_final_norm",
                )

        # ----------------------------------------------------
        # Readout
        # ----------------------------------------------------

        hidden = v_ast.squeeze(1)

        logits = torch.matmul(
            hidden,
            self.readout,
        )

        if debug:
            self.check_finite(
                logits,
                "logits",
            )

        return logits


# ============================================================
# Parameter counter
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BDHModel().to(device)

    print("=" * 60)
    print("BDH-GPU MODEL")
    print("=" * 60)

    print(f"N:          {model.n}")
    print(f"D:          {model.d}")
    print(f"Heads:      {model.num_heads}")
    print(f"Layers:     {model.num_layers}")

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    x = torch.randint(
        0,
        VOCAB_SIZE,
        (2, 128),
        device=device,
    )

    logits = model(
        x,
        debug=True,
    )

    print(
        "Output:",
        logits.shape,
        logits.dtype,
    )

    print("Forward pass successful.")
