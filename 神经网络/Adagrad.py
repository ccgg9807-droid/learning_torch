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
optimizer = torch.optim.Adagrad(net.parameters(),lr=1e-2)
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

x_axis = np.linspace(0,5,len(losses),endpoint=True)
plt.figure()
plt.semilogy(x_axis,losses,label='Adagrad')
plt.legend()
plt.show()