import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
#数据路径输入
with open('/home/cg/learning_torch/神经网络/data.txt','r') as f:
    datalist = [ i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in datalist]

#标准化 每个值不大于1
x0_max = max(i[0] for i in data)
x1_max = max(i[1] for i in data)
data = [( i[0]/x0_max , i[1]/x1_max , i[2]) for i in data]

#不同的点分类
x0 = list( filter ( lambda x : x[-1] == 0.0, data))
x1 = list( filter ( lambda x : x[-1] == 1.0, data))
plot_x0 = list( i[0] for i in x0 )
plot_y0 = list( i[1] for i in x0)
plot_x1 = list (i[0] for i in x1)
plot_y1 = list (i[1] for i in x1)

#data转化成numpy的形式
np.data = np.array(data,dtype= 'float32')
#再转化成torch
#数据x=（w1,w2）
x_data = torch.from_numpy(np.data[:,0:2])
#标签|y|=1
y_data = torch.from_numpy(np.data[:,-1]).unsqueeze(1)
''''
#转化成varible
x_data = Variable(x_data)
y_data = Variable(y_data)


w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))

def logstic_regression(x):
    return F.sigmoid(torch.mm(x,w)+b)

def binary_loss(y_,y_data):
    logits = torch.mean((y_data * y_.clamp(1e-12).log() + (1 - y_data) * (1 - y_).clamp(1e-12).log() )) #clamp函数为了保证损失值始终有一个保底的正数下限不至于出现梯度消失 的现象
    return -logits

optimizer = torch.optim.SGD([w,b],lr= 1.)

import time
start = time.time()
for e in range(1000):
    y_ = logstic_regression(x_data)
    loss = binary_loss(y_,y_data)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    mask = y_.ge(0.5).float()
    acc =(mask == y_data).sum().data/y_data.shape[0]
    if((e+1)%20 ==0):
        print("epoch : {} loss : {}".format(e+1,loss.data,acc))
during = time.time()-start

'''


#权重w和b
w = Variable(torch.randn(2,1),requires_grad= True)
b = Variable(torch.zeros(1),requires_grad= True)

#sigmoid 函数
def logstic_regression(x):
    return F.sigmoid(torch.mm(x,w)+b)

#显示初始分类面
w0 = w[0].data
w1 = w[1].data
b0 = b.data
plot_x = np.arange(0.2,1,0.01)
plot_y = (- w0 * plot_x - b0) / w1
plt.figure(1)
plt.plot(plot_x,plot_y,'g',label = 'cutting line')
plt.plot(plot_x0,plot_y0,'bo',label = 'x0')
plt.plot(plot_x1,plot_y1,'ro',label = 'x1')
plt.legend()
plt.show()

# loss = -(y*lny_ + （1-y）* ln(1-y_))  y是数据经过sigmoid函数处理过后的
def binary_loss(y_,y_data):
    logits = torch.mean((y_data * y_.clamp(1e-12).log() + (1 - y_data) * (1 - y_).clamp(1e-12).log() )) #clamp函数为了保证损失值始终有一个保底的正数下限不至于出现梯度消失 的现象
    return -logits
print(x_data.shape)
print(y_data.shape)
y_ = logstic_regression(x_data)
print(y_.shape)
loss = binary_loss(y_,y_data)
print(loss)

loss.backward()
w.data = w.data - 0.1 * w.grad.data
b.data = b.data - 0.1 * b.grad.data

y_ = logstic_regression(x_data)
loss = binary_loss(y_,y_data)
print(loss)


for i in range(1,1001):
    y_ = logstic_regression(x_data)
    w.grad.zero_()
    b.grad.zero_()
    loss = binary_loss(y_,y_data)
    
    loss.backward()
    w.data = w.data - 1 * w.grad.data
    b.data = b.data - 1 * b.grad.data

    if((i+1)%20 ==0):
        print("epoch : {} loss : {}".format(i+1,loss.data))


w0 = w[0].data
w1 = w[1].data
b0 = b.data
plot_x = np.arange(0.2,1,0.01)
plot_y = (- w0 * plot_x - b0) / w1
plt.figure()
plt.plot(plot_x,plot_y,'g',label = 'cutting line')
plt.plot(plot_x0,plot_y0,'bo',label = 'x0')
plt.plot(plot_x1,plot_y1,'ro',label = 'x1')
plt.legend()
plt.show()

