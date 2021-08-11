import torch
import numpy as np

#GPU가 존재하면 텐서를 이동
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:', device)



#numpy식 표준 인덱싱, 슬라이싱

tensor = torch.ones(4, 4)
print('first row : ', tensor[0])
print('first column : ', tensor[:, 0])
print('last column : ', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

'''
True
학습을 진행하는 기기: cuda:0
first row :  tensor([1., 1., 1., 1.])
first column :  tensor([1., 1., 1., 1.])
last column :  tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
'''

