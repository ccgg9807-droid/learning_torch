import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,5),
            nn.ReLU((True))
        )

        self.max_pool1 = nn.MaxPool2d(3,2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,5),
            nn.ReLU((True))
        )

        self.max_pool2 = nn.MaxPool2d(3,2)

        self.fc1 = nn.Sequential(
            nn.Linear(1024,384),
            nn.ReLU(True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(384,192),
            nn.ReLU(True)
        )

        self.fc3 = nn.Linear(192,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

alexnet = AlexNet()
#print(alexnet)


#检验 alexnet 的网络结构
input_demo = Variable(torch.zeros(64,3,32,32))
out_demo = alexnet(input_demo)
print(out_demo.shape)

def data_tf(x):
    x = np.array(x,dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose(2,0,1)
    x = torch.from_numpy(x)
    return x

#train_set 要用两个值来取第一个元素
#比如 im,label = train_set[0]     注意train_set 有没有经过transform 
#经过transform im.shape = torch.size([3,32,32] ) label = "0-9"的int型数据 没有shape的attribute
train_set = CIFAR10("/home/cg/learning_torch/CNN/CIFAR10",train= True, transform= data_tf,download=True)
test_set = CIFAR10("/home/cg/learning_torch/CNN/CIFAR10",train= False, transform= data_tf,download=True)
#im,label = train_set[0]
#print(im.shape)
#print(label)

#train_data 是一个迭代器 要用next（iter(train_data))来访问两个数据
#经过torch.utils.data.DataLoader后 
#im.shape = torch.Size([64, 3, 32, 32])  label.shape = torch.Size([64])
train_data = torch.utils.data.DataLoader(train_set,batch_size= 64,shuffle=True)
test_data = torch.utils.data.DataLoader(test_set,batch_size= 64,shuffle=True)
im,label = next(iter(train_data))
print(im.shape)
print(label.shape)
                
#GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlexNet().to(device)
#参数更新策略 学习率为0.1
optimizer = torch.optim.SGD(net.parameters(),lr=1e-1)

#损失函数
criterion = nn.CrossEntropyLoss()
from utils import train
print(f"检查 GPU 是否可用: {torch.cuda.is_available()}")
train(net,train_data,test_data,20,optimizer,criterion)

