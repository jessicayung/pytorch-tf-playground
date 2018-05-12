import torch

t = torch.Tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("Original tensor: ", t)

# reshape tensor
print("Reshaped tensors:")
print(t.view(4,3))
# -1 here means infer from other dimensions
print(t.view(-1))
print(t.view(2,-1))

