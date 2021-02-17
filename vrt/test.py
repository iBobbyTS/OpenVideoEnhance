import torch

x = torch.randint(0, 255, (1, 5, 6, 3), dtype=torch.float)
pader = torch.nn.ReflectionPad2d([1, 2, 3, 4])
p = pader(x)
print(p)
