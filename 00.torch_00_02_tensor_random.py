import torch
import numpy as np

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"random tensor : {rand_tensor}")
print(f"ones tensor : {ones_tensor}")
print(f"zeros tensor : {zeros_tensor}")

'''
random tensor : tensor([[0.4998, 0.9326, 0.4113],
        [0.2128, 0.4829, 0.8919]])
ones tensor : tensor([[1., 1., 1.],
        [1., 1., 1.]])
zeros tensor : tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''