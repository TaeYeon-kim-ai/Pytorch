#CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됩니다.
import torch
import numpy as np

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n : {n}")

'''
t: tensor([1., 1., 1., 1., 1.])
n : [1. 1. 1. 1. 1.]
'''

t.add_(1)
print(f"t : {t}")
print(f"n : {n}")
#텐서의 변경 사항이 NumPy 배열에 반영됩니다.
'''
t : tensor([2., 2., 2., 2., 2.])
n : [2. 2. 2. 2. 2.]
'''

#Numpy배열을 텐서로 변환

n = np.ones(5)
t = torch.from_numpy(n)

#Numpy 배열의 변경 사항이 텐서에 반영
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

'''
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
'''
