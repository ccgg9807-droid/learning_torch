import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch.nn import init
import torch
from PIL import Image


im = Image.open('/home/cg/learning_torch/神经网络/cat/cat.png').convert('L')
#数据类型一般是uint8 不便于矩阵计算
im = np.array(im,dtype= 'float32')
print(im.shape)

#展示图片转化为uint8
plt.figure(1)
plt.imshow(im.astype('uint8'),cmap='gray')
plt.show()
'''
#卷积层
im = torch.from_numpy(im.reshape((1,1,im.shape[0],im.shape[1])))

conv1 = nn.Conv2d(1,1,3,bias=False)
sobel_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype='float32')
sobel_kernel = sobel_kernel.reshape(1,1,3,3)
conv1.weight.data = torch.from_numpy(sobel_kernel)

edge = conv1(Variable(im))
edge = edge.data.squeeze().numpy()
plt.imshow(edge,cmap='gray')
plt.show()
'''
im = torch.from_numpy(im.reshape((1,1,im.shape[0],im.shape[1])))
pool1 = nn.MaxPool2d(2,2)
print("before maxpool,image shape :{}*{}".format(im.shape[2],im.shape[3]))
small_im1 = pool1(Variable(im))
small_im1 = small_im1.data.squeeze().numpy()
print("after maxpool,image shape :{}*{}".format(small_im1.shape[0],small_im1.shape[1]))
plt.figure(2)
plt.imshow(small_im1,cmap='gray')
plt.show()