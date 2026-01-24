import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch.nn import init
import torch

#torch自带的更新策略
def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data',train = True,transform = data_tf,download= True)
test_set = mnist.MNIST('./data',train = False,transform = data_tf,download= True)
train_data = DataLoader(train_set,batch_size = 64,shuffle= True)

#损失函数 交叉熵
criterion = nn.CrossEntropyLoss()

#定义三层神经网络 sequential
net = nn.Sequential(
    nn.Linear(784,200),
    nn.ReLU(),
    nn.Linear(200,10)
)
optimizer = torch.optim.SGD(net.parameters(),lr=1e-2,momentum=0.9)
#开始训练
losses = []
idx = 0
import time
start = time.time()

for e in range(5):
    train_loss = 0
    for im,label in train_data:

        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out,label)

        #清空梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        if(idx % 30 == 0):
            losses.append(loss.data)
        idx += 1
    print("epoch: {} train loss :{:.6f}".format(e,train_loss / len(train_data)))
    
end =time.time()
print("使用时间：{:5f}s".format((end-start)))

#不加动量的SGD
net = nn.Sequential(
    nn.Linear(784,200),
    nn.ReLU(),
    nn.Linear(200,10)
)
optimizer = torch.optim.SGD(net.parameters(),lr=1e-2)
#开始训练
losses1 = []
idx = 0
import time
start = time.time()

for e in range(5):
    train_loss = 0
    for im,label in train_data:

        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out,label)

        #清空梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        if(idx % 30 == 0):
            losses1.append(loss.data)
        idx += 1
    print("epoch: {} train loss :{:.6f}".format(e,train_loss / len(train_data)))
    
end =time.time()
print("使用时间：{:5f}s".format((end-start)))


x_axis = np.linspace(0,5,len(losses1),endpoint=True)
plt.figure()
plt.semilogy(x_axis,losses,label='momentum:0.9')
plt.semilogy(x_axis,losses1,label='no momentum')
plt.legend()
plt.show()
'''
#不用torch自带的更新策略

#动量法参数的更新策略 vi = r *vi-1 + lr * grad_w
#               wi = wi-1 -vi
def sgd_momentum(parameters,vs,lr,gamma):
    for param,v in zip(parameters,vs):
        v[:] = gamma * v + lr * param.grad.data
        param.data = param.data - v


#Vs初始化成和parameter大小一样
vs = []
for param in net.parameters():
    vs.append(torch.zeros_like(param.data))

#开始训练
losses1 = []
idx = 0
import time
start = time.time()

for e in range(5):
    train_loss = 0
    for im,label in train_data:

        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out,label)

        #清空梯度
        net.zero_grad()
        loss.backward()
        sgd_momentum(net.parameters(),vs,1e-2,0.9)

        train_loss += loss.data
        if(idx % 30 == 0):
            losses1.append(loss.data)
        idx += 1
    print("epoch: {} train loss :{:.6f}".format(e,train_loss / len(train_data)))
    
end =time.time()
print("使用时间：{:5f}s".format((end-start)))


x_axis = np.linspace(0,5,len(losses1),endpoint=True)
plt.semilogy(x_axis,losses1,label='batch_size = 64')
plt.legend()
plt.show()
'''