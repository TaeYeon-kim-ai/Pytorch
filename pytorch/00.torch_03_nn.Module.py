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


# GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usung {} device".format(device))

#2. model 
class NeuralNetwork(nn.Module) :
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), 
            nn.ReLU()
        )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

model = NeuralNetwork().to(device)
print(model)

# Usung cuda device
# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#     (5): ReLU()
#   )
# )

        
