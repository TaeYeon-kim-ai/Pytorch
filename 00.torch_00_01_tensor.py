#tensor
import torch
import numpy as np


data = [[1,2], [3, 4]]
x_data = torch.tensor(data)

#Numpy

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#다른 텐서로 부터 생성
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor : \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"random tensor : \n {x_rand} \n")

# Ones Tensor : 
#  tensor([[1, 1],
#         [1, 1]]) 

# random tensor : 
#  tensor([[0.8457, 0.8761],
#         [0.0939, 0.7119]]) 
