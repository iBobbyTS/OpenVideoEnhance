import torch

W_MAX = 100
H_MAX = 200
B_MAX = 3
grid = torch.cat((
    torch.arange(0, W_MAX).repeat(H_MAX, 1).view(1, 1, H_MAX, W_MAX).repeat(B_MAX, 1, 1, 1),
    torch.arange(0, H_MAX).repeat(1, W_MAX).view(1, 1, H_MAX, W_MAX).repeat(B_MAX, 1, 1, 1)
), 1)
print(grid.shape)
