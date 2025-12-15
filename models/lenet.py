'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np

class LeNet(nn.Module):
    def __init__(self,ar1_load,config,layer_scale):
        super(LeNet, self).__init__()
        self.ar1_load = ar1_load
        self.config = config
        self.layer_scale = layer_scale
        self.conv1 = nn.Conv2d(3, 6, 5)  # 对应CIFAR-10的3通道输入
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10共有10个类别

    def forward(self, x):
        layer_counter  = 0
        # 应用第一个卷积层
        x = self.conv1(x)
        # 应用ReLU激活函数
        x = torch.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        # 应用第一个池化层
        x = nn.MaxPool2d(2, 2)(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        # 应用第二个卷积层
        x = self.conv2(x)
        # 应用ReLU激活函数
        x = torch.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        # 应用第二个池化层
        x = nn.MaxPool2d(2, 2)(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = torch.relu(self.fc2(x))
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = self.fc3(x)
       # np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        return x

def lenet_find_config(ar1_load,config,layer_scale):
    return LeNet(ar1_load,config,layer_scale)

def approx(out,level1,level2,level3,level4,scale,config):
    if(config == 3):
        out = torch.where(torch.lt(out*scale,level1),0,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level1),level1/scale,out)
        out = torch.where(torch.lt(out*scale,level3) & torch.gt(out*scale,level2),level2/scale,out)
        out = torch.where(torch.lt(out*scale,level4) & torch.gt(out*scale,level3),level3/scale,out)
    elif(config == 2):
        out = torch.where(torch.lt(out*scale,level1),0,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level1),level1/scale,out)
        out = torch.where(torch.lt(out*scale,level3) & torch.gt(out*scale,level2),level2/scale,out)
    elif(config == 1):
        out = torch.where(torch.lt(out*scale,level1),0,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level1),level1/scale,out)
    elif(config == 0):
        out = torch.where(torch.lt(out*scale,level1),0,out)
    return out