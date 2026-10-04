import torch
import torch.nn as nn


class BDHAttention(nn.Module):

    def __init__(self, n=512, d=32):
        super().__init__()

        self.n = n
        self.d = d

        # Paper: Dy ∈ R^(n × d)
        self.Dy = nn.Parameter(
            torch.randn(n, d) * 0.02
        )

        # Paper: E ∈ R^(d × n)
        self.E = nn.Parameter(
            torch.randn(d, n) * 0.02
        )

        self.attn_norm = nn.LayerNorm(d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, rho, x):

        # --------------------------------------------------
        # 1. Linear attention
        # --------------------------------------------------

        # rho: [B, n, d]
        # x:   [B, n]
        #
        # a:   [B, d]

        a = torch.bmm(
            rho.transpose(1, 2),
            x.unsqueeze(-1)
        ).squeeze(-1)

        # --------------------------------------------------
        # 2. LayerNorm(a)
        # --------------------------------------------------

        a_norm = self.attn_norm(a)

        # --------------------------------------------------
        # 3. Dy @ LN(a)
        # --------------------------------------------------

        # a_norm: [B, d]
        # Dy.T:   [d, n]
        #
        # result: [B, n]

        dy_a = a_norm @ self.Dy.T

        # --------------------------------------------------
        # 4. Elementwise multiplication with x
        # --------------------------------------------------

        y = dy_a * x

        # --------------------------------------------------
        # 5. E @ y
        # --------------------------------------------------

        # y:   [B, n]
        # E.T: [n, d]
        #
        # result: [B, d]

        v = y @ self.E.T

        # --------------------------------------------------
        # 6. Final LayerNorm
        # --------------------------------------------------

        v = self.output_norm(v)

        return a, y, v


if __name__ == "__main__":

    device = torch.device("cuda")

    B = 4
    n = 512
    d = 32

    rho = torch.randn(
        B, n, d,
        device=device
    )

    x = torch.randn(
        B, n,
        device=device
    )

    attention = BDHAttention(
        n=n,
        d=d
    ).to(device)

    a, y, v = attention(rho, x)

    print("rho :", rho.shape, rho.device)
    print("x   :", x.shape, x.device)
    print("a   :", a.shape, a.device)
    print("y   :", y.shape, y.device)
    print("v   :", v.shape, v.device)

    print("\nParameters:")

    for name, param in attention.named_parameters():
        print(
            f"{name:15s}",
            param.shape,
            param.device
        )