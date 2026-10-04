import torch
import torch.nn as nn


class BDHStep(nn.Module):

    def __init__(self, n=512, d=32):
        super().__init__()

        self.n = n
        self.d = d

        # E ∈ R^(d × n)
        self.E = nn.Parameter(
            torch.randn(d, n) * 0.02
        )

        # Dx, Dy ∈ R^(n × d)
        self.Dx = nn.Parameter(
            torch.randn(n, d) * 0.02
        )

        self.Dy = nn.Parameter(
            torch.randn(n, d) * 0.02
        )

        self.attn_norm = nn.LayerNorm(d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, x, v_input, rho):

        # ==================================================
        # 1. x_t,l
        #
        # x_t,l = x_t,l-1 + Dx v*_t,l-1
        # ==================================================

        x = x + v_input @ self.Dx.T

        # ==================================================
        # 2. Attention
        #
        # IMPORTANT:
        # rho contains ONLY previous timesteps.
        # ==================================================

        a = torch.bmm(
            rho.transpose(1, 2),
            x.unsqueeze(-1)
        ).squeeze(-1)

        # ==================================================
        # 3. LayerNorm
        # ==================================================

        a_norm = self.attn_norm(a)

        # ==================================================
        # 4. Dy LN(a) ⊙ x
        # ==================================================

        y = (a_norm @ self.Dy.T) * x

        # ==================================================
        # 5. E y
        # ==================================================

        v_output = y @ self.E.T

        # ==================================================
        # 6. LayerNorm
        # ==================================================

        v_output = self.output_norm(v_output)

        # ==================================================
        # 7. IMPORTANT STATE UPDATE
        #
        # rho_t,l =
        # rho_(t-1),l +
        # x_t,l ⊗ v*_t,l-1
        #
        # NOT v_output.
        # ==================================================

        rho = rho + (
            x.unsqueeze(-1)
            *
            v_input.unsqueeze(-2)
        )

        return x, y, v_output, rho


if __name__ == "__main__":

    device = torch.device("cuda")

    B = 4
    n = 512
    d = 32

    layer = BDHStep(
        n=n,
        d=d
    ).to(device)

    x_prev = torch.randn(
        B, n,
        device=device
    )

    v_prev = torch.randn(
        B, d,
        device=device
    )

    rho = torch.zeros(
        B, n, d,
        device=device
    )

    x, y, v, rho_new = layer(
        x_prev,
        v_prev,
        rho
    )

    print("x       :", x.shape, x.device)
    print("y       :", y.shape, y.device)
    print("v       :", v.shape, v.device)
    print("rho_new :", rho_new.shape, rho_new.device)

    print("\nExpected:")
    print(f"x       : [{B}, {n}]")
    print(f"y       : [{B}, {n}]")
    print(f"v       : [{B}, {d}]")
    print(f"rho_new : [{B}, {n}, {d}]")