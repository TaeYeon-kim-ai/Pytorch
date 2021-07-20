import torch
import numpy as np


tensor = torch.rand(3, 4)

print(f"shape of tensor : {tensor.shape}")
print(f"datatype of tensor : {tensor.dtype}" )
print(f"device tensor is stored on : {tensor.device}")

'''
shape of tensor : torch.Size([3, 4])
datatype of tensor : torch.float32
device tensor is stored on : cpu
'''

