import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch.nn import init
import torch

def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = (x - 0.5) / 0.5
    x = x.reshape((-1,))
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data',train = True,transform = data_tf,download= True)
test_set = mnist.MNIST('./data',train = False,transform = data_tf,download= True)

#损失函数 交叉熵
criterion = nn.CrossEntropyLoss()

# w = w - lr * grad
def Sgd_update(parameters,lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data

'''
#先试一下batch_size =1
train_data = DataLoader(train_set,batch_size = 1,shuffle= True)

#等号前面必须要有两个参数
a,a_label= train_set[0]

#print(a.shape)     得到size784

#定义三层神经网络 sequential
net = nn.Sequential(
    nn.Linear(784,200),
    nn.ReLU(),
    nn.Linear(200,10)
)

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
        Sgd_update(net.parameters(),1e-2)

        train_loss += loss.data
        if(idx % 30 == 0):
            losses1.append(loss.data)
        idx += 1
    print("epoch: {} train loss :{:.6f}".format(e,train_loss / len(train_data)))
    
end =time.time()
print("使用时间：{:5f}s".format((end-start)))

#用了551.3s train_loss 函数震荡明显
x_axis = np.linspace(0,5,len(losses1),endpoint=True)
plt.semilogy(x_axis,losses1,label='batch_size = 1')
plt.legend()
plt.show()
'''
#再试一下batch_size =64
train_data = DataLoader(train_set,batch_size = 64,shuffle= True)

#等号前面必须要有两个参数
a,a_label= train_set[0]

#print(a.shape)     得到size784

#定义三层神经网络 sequential
net = nn.Sequential(
    nn.Linear(784,200),
    nn.ReLU(),
    nn.Linear(200,10)
)

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
        Sgd_update(net.parameters(),1e-2)

        train_loss += loss.data
        if(idx % 30 == 0):
            losses1.append(loss.data)
        idx += 1
    print("epoch: {} train loss :{:.6f}".format(e,train_loss / len(train_data)))
    
end =time.time()
print("使用时间：{:5f}s".format((end-start)))

#34s 震荡不明显
x_axis = np.linspace(0,5,len(losses1),endpoint=True)
plt.semilogy(x_axis,losses1,label='batch_size = 64')
plt.legend()
plt.show()



#后期可以自己调整学习率 这节内容其实没什么 就是实战看看batch_size lr对损失函数的和时间成本的影响