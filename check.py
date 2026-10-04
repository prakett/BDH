import torch

device = torch.device("cuda")

n = 512
d = 32

x = torch.randn(4, n, device=device)
W = torch.randn(n, d, device=device)

y = x @ W

print("x:", x.shape, x.device)
print("W:", W.shape, W.device)
print("y:", y.shape, y.device)