import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn

#绘制决策边界函数 model是模型 x是数据坐标点（x,y） y是对应数据的标签label 正负1
def plot_decision_boudnary(model,x,y):
    #绘制数据边界 padding=1 防止数据出现在边界 要注意的是这里的y_min和y_max 都是参数x里的数据
    x_min,x_max = x[:,0].min()- 1, x[:,0].max() + 1
    y_min,y_max = x[:,1].min()- 1, x[:,1].max() + 1
    h = 0.01

    xx , yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    #np.c_[xx.ravel(),yy.ravel()] 就把数据变成了坐标一一对应的形式
    #[[x0,y0]
    #[x1,y1]
    #[x2,y2]]
    #z = 模型后对数据点分分类情况
    z = model(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx,yy,z,cmap = plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(x[:,0],x[:,1],c=y.reshape(-1),cmap = plt.cm.Spectral)
    plt.show()

#数据
np.random.seed(1)
m = 400 
N = int(m/2)
D = 2
x = np.zeros((m,D))
y = np.zeros((m,1),dtype= 'uint8')
a = 4

for j in range(2):
    ix = range(N * j, N * (j+1))
    t = np.linspace(j * 3.12,(j+1) * 3.12,N ) + np.random.randn(N) * 0.2
    r = a * np.sin(4 * t) +np.random.randn(N) * 0.2
    x[ix] = np.c_[r * np.sin(t),r * np.cos(t)]
    y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
plt.show()
#转化成张量
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

#Moduel
class moduel_net(nn.Module):
    def __init__(self, num_input,num_hidden,num_output):
        super(moduel_net,self).__init__()
        self.layer1 = nn.Linear(num_input,num_hidden)

        self.layer2 = nn.Tanh()

        self.layer3 = nn.Linear(num_hidden,num_output)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

mo_net = moduel_net(2,4,1)

optimizer = torch.optim.SGD(mo_net.parameters(),lr=1.)

criterion = nn.BCEWithLogitsLoss()

for e in range(1000):
    y_ = mo_net(Variable(x))
    loss = criterion(y_,Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if((e+1)%1000 ==0):
        print("epoch : {} loss : {}".format(e+1,loss.data))

def plot_moduel(x):
    out = mo_net(Variable(torch.from_numpy(x).float()))
    out = F.sigmoid(out).data.numpy()
    out =(out > 0.5)*1
    return out

plot_decision_boudnary(lambda x : plot_moduel(x),x.numpy(),y.numpy())
plt.show()
'''
#sequential 直接代替w1 b1 w2 b2
seq_set  = nn.Sequential(
    nn.Linear(2,4),
    nn.Tanh(),
    nn.Linear(4,1)
)
#w = nn.Parameter(torch.randn(n,m))  让optim记住参数
param = seq_set.parameters()

optimizer = torch.optim.SGD(param,lr=1.)
criterion = nn.BCEWithLogitsLoss()
for e in range(10000):
    y_ = seq_set(Variable(x))
    loss = criterion(y_,Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if((e+1)%1000 ==0):
        print("epoch : {} loss : {}".format(e+1,loss.data))


def plot_sequential(x):
    out = F.sigmoid(seq_set(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5)*1
    return out

plot_decision_boudnary(lambda x:plot_sequential(x),x.numpy(),y.numpy())
plt.show()


'''
'''
#多层神经网络
w1 = nn.Parameter(torch.randn(2,4)*0.01)
b1 = nn.Parameter(torch.zeros(4) )

w2 = nn.Parameter(torch.randn(4,1)* 0.01)
b2 = nn.Parameter(torch.zeros(1))

def two_networks(x):
    x1 = torch.mm(x,w1) + b1
    y1 = F.tanh(x1)
    x2 = torch.mm(y1,w2)+b2
    return x2

optimizer = torch.optim.SGD([w1,w2,b1,b2],lr=1.)

criterion = nn.BCEWithLogitsLoss()

for e in range(10000):
    y_ = two_networks(Variable(x))
    loss = criterion(y_,Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if((e+1)%1000 ==0):
        print("epoch : {} loss : {}".format(e+1,loss.data))

def plot_twonetwork(x):
    x = torch.from_numpy(x).float()
    x = Variable(x)
    y_ = two_networks(x)
    out = F.sigmoid(y_)
    out =(out>0.5)*1
    return out

plot_decision_boudnary(lambda x:plot_twonetwork(x),x.numpy(),y.numpy())
plt.show()

'''
'''
#logistic 方法 
np.random.seed(1)
m = 400 
N = int(m/2)
D = 2
x = np.zeros((m,D))
y = np.zeros((m,1),dtype= 'uint8')
a = 4

for j in range(2):
    ix = range(N * j, N * (j+1))
    t = np.linspace(j * 3.12,(j+1) * 3.12,N ) + np.random.randn(N) * 0.2
    r = a * np.sin(4 * t) +np.random.randn(N) * 0.2
    x[ix] = np.c_[r * np.sin(t),r * np.cos(t)]
    y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
plt.show()



w = nn.Parameter(torch.randn(2,1))
b = nn.Parameter(torch.zeros(1))

optimizer = torch.optim.SGD([w,b],lr=1e-1)

def logistic_regression(x):
    return torch.mm(x,w)+b

criterion = nn.BCEWithLogitsLoss()

for e in range(100):
    y_ = logistic_regression(Variable(x))
    loss = criterion(y_,Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if((e+1)%20 ==0):
        print("epoch : {} loss : {}".format(e+1,loss.data))

def plot_logistic(x):
    x = torch.from_numpy(x).float()
    y_ = logistic_regression(Variable(x))
    y_ = F.sigmoid(y_)
    out = (y_ > 0.5) * 1
    return  out.data.numpy()

plot_decision_boudnary(lambda x:plot_logistic(x),x.numpy(),y.numpy())
'''