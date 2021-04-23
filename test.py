import torch
x = torch.load('/Users/ibobby/Downloads/pwc_net.pth', map_location='cpu')
print(x)
