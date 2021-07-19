#API활용
import torch
from torch import nn
from torch.utils.data import DataLoader # 데이터를 반복 가능한 객체 iterable로 감싼다.
from torchvision import datasets # 샘플과 정답 lable 저장
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

#1. data
#train data load
training_data = datasets.FashionMNIST(
    root = "data", 
    train = True, 
    download = True, 
    transform=ToTensor(),
)

#test data load
test_data = datasets.FashionMNIST(
    root = "data", 
    train = False,
    download = True,
    transform=ToTensor(),
)


batch_size =  64

#dataloader 
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader :
    print("shape of X [N, C, H, W] : ", X.shape)
    print("shape of y : ", y.shape, y.dtype)
    break

# shape of X [N, C, H, W] :  torch.Size([64, 1, 28, 28])
# shape of y :  torch.Size([64]) torch.int64

