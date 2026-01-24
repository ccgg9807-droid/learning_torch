import torch
import os
print("当前工作目录是:", os.getcwd()) 
# 看看打印出来的路径，是不是和你上次运行的路径不一样？
# 你的 ./data 就会生成在这个路径下面

import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import mnist

train_set = mnist.MNIST('./data',train = True,download= True)
test_set = mnist.MNIST('./data',train = False,download= True)

#简单查看一下数据集的属性
a_data,a_label = train_set[0]
'''

plt.imshow(a_data, cmap='gray')
plt.show()
'''
a_data = np.array(a_data,dtype='float32')
#print(a_data)
#print(a_data.shape)

def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data',train = True,transform=data_tf,download= True)
test_set = mnist.MNIST('./data',train = False,transform=data_tf,download= True)

a,a_label = train_set[0]
print(a.shape)
print(a_label)

from torch.utils.data import DataLoader

train_data = DataLoader(train_set , batch_size = 64, shuffle = True)
test_data = DataLoader(test_set , batch_size = 128, shuffle = False)

a,a_label = next(iter(train_data))
print(a.shape)
print(a_label.shape)

net = nn.Sequential(
    nn.Linear(784,400),
    nn.ReLU(),
    nn.Linear(400,200),
    nn.ReLU(),
    nn.Linear(200,100),
    nn.ReLU(),
    nn.Linear(100,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=1e-1)

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im,label in train_data:
        #tensor -> Variable
        im = Variable(im)
        label = Variable(label)

        #预测值
        out = net(im)
        loss = criterion(out , label)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        _,pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct / im.shape[0]
        train_acc += acc   
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    net.eval()
    for im,label in test_data:
        #tensor -> Variable
        im = Variable(im)
        label = Variable(label)

        #预测值
        out = net(im)
        loss = criterion(out , label)

        eval_loss += loss.data

        _,pred = out.max(1)
        num_correct = (pred == label).sum().data
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print("epoch :{} Train loss:{:.6f} Train acc: {:.6f} Test loss : {:.6f} Test acc :{:.6f}".format(e+1,train_loss / len(train_data),train_acc / len(train_data),eval_loss / len(test_data),eval_acc / len(test_data)))

plt.figure()
plt.title("train loss")
plt.plot(np.arange(len(losses)),losses)
plt.figure()
plt.title("train acc")
plt.plot(np.arange(len(acces)),acces)
plt.figure()
plt.title("eval loss")
plt.plot(np.arange(len(eval_losses)),eval_losses)
plt.figure()
plt.title("train acc")
plt.plot(np.arange(len(eval_acces)),eval_acces)
plt.show()