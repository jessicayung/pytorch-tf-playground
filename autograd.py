import torch
import numpy as np

# if you don't require grad you can't backprop
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = x * y
out = z.mean()

# backprop
out.backward()

# gradients of x wrt out
print("x grad:", x.grad)
print("y grad:", y.grad)
print("z grad:", z.grad)

# you can require grad for all model parameters using
"""
for param in model.parameters():
    param.requires_grad = True
"""
