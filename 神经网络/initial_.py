import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn

#定义一个sequential模型
net1 = nn.Sequential(
    nn.Linear(30,40),
    nn.ReLU(),
    nn.Linear(40,50),
    nn.ReLU(),
    nn.Linear(50,10)
)

#访问第一层的参数
# net1[0].weight是一个Parameter 是一个Variable变量 数据要.data后变为tensor
w1 = net1[0].weight
b1 = net1[0].bias
print(w1)
#print(b1)
# np.random.uniform(3,5,size = (40,30))   生成一组正态分布 大小为size 范围在3-5之间的数据
net1[0].weight.data = torch.from_numpy(np.random.uniform(3,5,size=(40,30)))
#print (w1)

#循环修改初始参数
for layer in net1:
    if isinstance(layer,nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0,0.5,size=(param_shape)))

#print(w1)

class sim_net(nn.Module):
    def __init__(self):
        super(sim_net,self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30,40),
            nn.ReLU()
        )
        self.l1[0].weight.data = torch.randn(40,30)

        self.l2 = nn.Sequential(
            nn.Linear(40,50),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(50,10),
            nn.ReLU()
        )

        def forward(self,x):
            x = self.l1(x)
            x = self.l2(x)
            x = self.l3(x)
            return x
net2 = sim_net()

'''
#访问children
for i in net2.children():
    print(i)
'''
'''
#访问module
for i in net2.modules():
    print(i)
'''
#想看网络的大致结构（有哪些大块）：用 children()。
#想修改每一个原子层的属性（比如给所有的 Linear 层初始化权重，或者冻结参数）：用 modules()。

for layer in net2.modules():
    if isinstance(layer,nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0,0.5,size=(param_shape)))


#torch.nn.init
#xavier 的初始化使得每一层的输出方差近乎相等
from torch.nn import init

print(net1[0].weight)
init.xavier_uniform(net1[0].weight)
print(net1[0].weight)