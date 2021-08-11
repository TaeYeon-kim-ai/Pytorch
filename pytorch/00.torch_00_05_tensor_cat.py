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

t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

# tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
#         [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])




#산술 연산(Arithmetic operations)
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#단일요소(single-element)텐서의 모든 값을 하나로 집계 하여 요소가 하나인 턴서의 경우 item()을 사용 하여  Python숫자 값으로 변환할 수 있습니다.
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
#12.0 <class 'float'>


#바꿔치기(in-place) 연산 연산 결과를 피연산자(operand)에 저장하는 연산을 바꿔치기 연산이라고 부르며, _ 접미사를 갖습니다. 예를 들어: x.copy_(y) 나 x.t_() 는 x 를 변경합니다.
print(tensor, "\n")
tensor.add_(4)
print(tensor)

'''
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[5., 4., 5., 5.],
        [5., 4., 5., 5.],
        [5., 4., 5., 5.],
        [5., 4., 5., 5.]])
'''






