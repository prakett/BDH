import torch
import torch.nn as nn


class BDHStep(nn.Module):

    def __init__(self, n=512, d=32):
        super().__init__()

        self.n = n
        self.d = d

        self.E = nn.Parameter(
            torch.randn(d, n) * 0.02
        )

        self.Dx = nn.Parameter(
            torch.randn(n, d) * 0.02
        )

        self.Dy = nn.Parameter(
            torch.randn(n, d) * 0.02
        )

        self.attn_norm = nn.LayerNorm(d)
        self.output_norm = nn.LayerNorm(d)

    def forward(self, x, v_input, rho):

        # ----------------------------------------------
        # 1. Current x
        #
        # x_t = x_(t,0) + Dx v_(t,0)
        # ----------------------------------------------

        x = x + v_input @ self.Dx.T

        # ----------------------------------------------
        # 2. Attention
        #
        # Uses ONLY previous rho
        # ----------------------------------------------

        a = torch.bmm(
            rho.transpose(1, 2),
            x.unsqueeze(-1)
        ).squeeze(-1)

        # ----------------------------------------------
        # 3. LayerNorm
        # ----------------------------------------------

        a = self.attn_norm(a)

        # ----------------------------------------------
        # 4. Dy LN(a) ⊙ x
        # ----------------------------------------------

        y = (a @ self.Dy.T) * x

        # ----------------------------------------------
        # 5. E y
        # ----------------------------------------------

        v = y @ self.E.T

        # ----------------------------------------------
        # 6. LayerNorm
        # ----------------------------------------------

        v = self.output_norm(v)

        # ----------------------------------------------
        # 7. Update state
        #
        # IMPORTANT:
        # current result is added AFTER attention
        # ----------------------------------------------

        rho = rho + (
            x.unsqueeze(-1)
            *
            v_input.unsqueeze(-2)
        )

        return x, y, v, rho


def run_sequence(model, v_inputs):

    B, T, d = v_inputs.shape

    device = v_inputs.device
    n = model.n

    # Initial neuron representation
    x = torch.zeros(
        B,
        n,
        device=device
    )

    # Initial recurrent state
    rho = torch.zeros(
        B,
        n,
        d,
        device=device
    )

    outputs = []
    states = []

    for t in range(T):

        # Current token's layer-0 representation
        v_input = v_inputs[:, t, :]

        x, y, v, rho = model(
            x,
            v_input,
            rho
        )

        outputs.append(v)
        states.append(rho)

    outputs = torch.stack(
        outputs,
        dim=1
    )

    return outputs, states


if __name__ == "__main__":

    device = torch.device("cuda")

    B = 2
    T = 8
    n = 512
    d = 32

    model = BDHStep(
        n=n,
        d=d
    ).to(device)

    # Simulated token encoder output.
    #
    # Later this will come from the actual
    # token embedding / encoder.
    v_inputs = torch.randn(
        B,
        T,
        d,
        device=device
    )

    outputs, states = run_sequence(
        model,
        v_inputs
    )

    print("Input shape:")
    print(v_inputs.shape)

    print("\nOutput shape:")
    print(outputs.shape)

    print("\nExpected:")
    print(f"Input  : [{B}, {T}, {d}]")
    print(f"Output : [{B}, {T}, {d}]")

    print("\nState shapes:")

    for t, rho in enumerate(states):
        print(
            f"t={t}:",
            rho.shape,
            rho.device
        )

    print("\nState norms:")

    for t, rho in enumerate(states):
        print(
            f"t={t}:",
            rho.norm().item()
        )