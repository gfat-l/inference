# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# class ModifiedLeNet(nn.Module):
#     def __init__(self,ar1_load,config,layer_scale):
#         super(ModifiedLeNet, self).__init__()
#         self.ar1_load = ar1_load
#         self.config = config
#         self.layer_scale = layer_scale
#         # 第一个卷积层, 使用padding=1来保持特征图大小
#         self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=1)  # CIFAR-10图像是3通道的
#         # 第二个卷积层, 同样使用padding=1
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=1)
#         # 全连接层定义
#         self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 调整尺寸以匹配经过两次池化后的特征图大小
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)  # CIFAR-10共有10个类别
#
#     def forward(self, x):
#         layer_counter = 0;
#         # 应用第一个卷积层后，使用ReLU激活函数，然后进行池化
#         x = F.relu(self.conv1(x))
#         x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         x = F.max_pool2d(x, 2)
#         x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
#
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         # 应用第二个卷积层后，同样使用ReLU激活函数，然后进行池化
#         x = F.relu(self.conv2(x))
#         x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
#
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         x = F.max_pool2d(x, 2)
#         x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
#
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         # 调整特征图的形状，以匹配全连接层的输入要求
#         x = x.view(-1, 16 * 6 * 6)
#         # 通过全连接层
#         x = F.relu(self.fc1(x))
#         x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
#
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         x = F.relu(self.fc2(x))
#         x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
#
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         x = self.fc3(x)
#         #np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter += 1
#         return x
#
#
# def lenet_find_config(ar1_load,config,layer_scale):
#     return ModifiedLeNet(ar1_load,config,layer_scale)
# def lenet_test():
#     return ModifiedLeNet(0,0,0)



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ModifiedLeNet(nn.Module):
    def __init__(self):
        super(ModifiedLeNet, self).__init__()
        # 维持卷积核尺寸为5x5
        # 第一个卷积层，增加padding到2，保证特征图尺寸适合池化
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2, stride=1)
        # 第二个卷积层，同样调整padding使得池化后尺寸合适
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2, stride=1)
        # 调整全连接层的输入尺寸
        self.fc1 = nn.Linear(16 * 8 * 8, 120) # 根据特征图尺寸调整
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10共有10个类别

    def forward(self, x):
        layer_counter = 0
        # 应用第一个卷积层后接ReLU激活函数和池化
        x = F.relu(self.conv1(x))
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = F.max_pool2d(x, 2, 2)
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        # 应用第二个卷积层后同样接ReLU激活函数和池化
        x = F.relu(self.conv2(x))
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = F.max_pool2d(x, 2, 2)
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        # 调整特征图尺寸以适配全连接层的输入
        x = x.view(-1, 16 * 8 * 8)
        # 通过全连接层
        x = F.relu(self.fc1(x))
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = F.relu(self.fc2(x))
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = self.fc3(x)
        np.savetxt(f"Lenet_data/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        return x

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