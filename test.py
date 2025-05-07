import torch

# (B, L)
# (2, L)
t0 = torch.tensor([
    [0, 0],
    [0, 0]
])

# (2*3, L)
t1 = torch.tensor([
    [1, 1],
    [1, 1],
    [1, 1],
    [2, 2],
    [2, 2],
    [2, 2]
])
B = 2
G_minus_1 = 3
# (2, 3, L)
t2 = t1.view(B, G_minus_1, -1)
print(t2)
t3 = torch.concat([t0.unsqueeze(1), t2], dim=1)
print(t3)