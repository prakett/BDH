import torch
import torch.nn as nn
import torch.nn.functional as F


class BDHLayer(nn.Module):
    def __init__(self, n=512, d=32):
        super().__init__()

        self.n = n
        self.d = d

        # BDH-GPU trainable matrices
        self.E = nn.Parameter(torch.randn(d, n) * 0.02)
        self.Dx = nn.Parameter(torch.randn(n, d) * 0.02)
        self.Dy = nn.Parameter(torch.randn(n, d) * 0.02)

    def forward(self, x):
        """
        x: [batch, n]

        returns:
            x_next: [batch, n]
            v:      [batch, d]
        """

        # x -> low-rank representation
        v = F.relu(x @ self.E.T)

        # Low-rank -> neuron space
        dx_v = v @ self.Dx.T

        # Residual connection
        x_next = x + dx_v

        # Low-rank representation for the next stage
        v_next = F.relu(x_next @ self.E.T)

        return x_next, v_next


if __name__ == "__main__":

    device = torch.device("cuda")

    layer = BDHLayer(
        n=512,
        d=32
    ).to(device)

    x = torch.randn(
        4,
        512,
        device=device
    )

    x_next, v = layer(x)

    print("x      :", x.shape, x.device)
    print("x_next :", x_next.shape, x_next.device)
    print("v      :", v.shape, v.device)

    print("\nParameters:")

    for name, parameter in layer.named_parameters():
        print(
            f"{name:5s}",
            parameter.shape,
            parameter.device
        )