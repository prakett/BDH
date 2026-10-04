import torch


def update_state(rho, x, v):
    """
    Update the BDH-GPU attention state.

    rho : [B, n, d]
    x   : [B, n]
    v   : [B, d]

    returns:
        rho_new : [B, n, d]
    """

    # Outer product:
    #
    # x -> [B, n, 1]
    # v -> [B, 1, d]
    #
    # result -> [B, n, d]

    update = x.unsqueeze(-1) * v.unsqueeze(-2)

    rho_new = rho + update

    return rho_new


if __name__ == "__main__":

    device = torch.device("cuda")

    B = 4
    n = 512
    d = 32

    # Persistent BDH state
    rho = torch.zeros(
        B,
        n,
        d,
        device=device
    )

    # Current neuron representation
    x = torch.randn(
        B,
        n,
        device=device
    )

    # Current low-rank value
    v = torch.randn(
        B,
        d,
        device=device
    )

    rho = update_state(rho, x, v)

    print("x   :", x.shape, x.device)
    print("v   :", v.shape, v.device)
    print("rho :", rho.shape, rho.device)

    print("\nExpected rho:")
    print(f"[{B}, {n}, {d}]")

    print("\nState norm:", rho.norm().item())