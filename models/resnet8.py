import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Initialize global layer counter
layer_counter = [0]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels,ar1_load,config,layer_scale, stride=1):
        super(ResidualBlock, self).__init__()
        self.ar1_load = ar1_load
        self.config = config
        self.layer_scale = layer_scale
        # print(self.ar1_load)
        # print(self.config)
        # print(self.layer_scale)

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        original_x = x
        out = F.relu(self.bn1(self.conv1(x)))
        global layer_counter
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1

        #self.approx_operation(out,self.ar1_load,self.config,self.layer_scale)
        #self.save_output(out)

        out = F.relu(self.bn2(self.conv2(out)))
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1

        #self.approx_operation(out,self.ar1_load,self.config,self.layer_scale)
        #self.save_output(out)

        out = self.bn3(self.conv3(out))
        # out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        # print(self.ar1_load[layer_counter[0]][3])
        layer_counter[0] += 1

        #self.approx_operation(out,self.ar1_load,self.config,self.layer_scale)
        #self.save_output(out)

        out += self.shortcut(original_x)
        out = F.relu(out)
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1
        #self.approx_operation(out,self.ar1_load,self.config,self.layer_scale)
        #self.save_output(out)

        return out

    def save_output(self, output):
        global layer_counter
        np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", output.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter[0] += 1
    def approx_operation(self,x,threshold,config,scale):
        global layer_counter
        # print(layer_counter)
        x = approx(x,threshold[layer_counter[0]][3],threshold[layer_counter[0]][2],threshold[layer_counter[0]][1],threshold[layer_counter[0]][0],scale[layer_counter[0]],config[layer_counter[0]])
        # print(threshold[layer_counter[0]][3])
        layer_counter[0] += 1
        return x

class ResNet8(nn.Module):
    def __init__(self,ar1_load,config,layer_scale):
        super(ResNet8, self).__init__()
        self.ar1_load = ar1_load
        self.config = config
        self.layer_scale = layer_scale
        # print(self.config)
        # print(self.layer_scale)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1 = ResidualBlock(32, 32, 32,ar1_load,config ,layer_scale, stride=2)
        self.block2 = ResidualBlock(32, 64, 64,ar1_load,config ,layer_scale, stride=2)
        self.block3 = ResidualBlock(64, 128, 128,ar1_load,config ,layer_scale, stride=2)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        global layer_counter
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1

        #self.approx_operation(out,self.ar1_load,self.config,self.layer_scale)
        #self.save_output(out)

        out = self.pool(out)
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1

        #self.approx_operation(out,self.ar1_load,self.config,self.layer_scale)
        #self.save_output(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        layer_counter[0] = 0
        #self.save_output(out)

        return out

    def save_output(self, output):
        global layer_counter
        np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", output.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter[0] += 1
    def approx_operation(self,x,threshold,config,scale):
        global layer_counter
        # print(layer_counter)
        x = approx(x,threshold[layer_counter[0]][3],threshold[layer_counter[0]][2],threshold[layer_counter[0]][1],threshold[layer_counter[0]][0],scale[layer_counter[0]],config[layer_counter[0]])
        # print(threshold[layer_counter[0]][3])
        layer_counter[0] += 1
        return x


def resnet8_find_config(ar1_load,config,layer_scale):
    return ResNet8(ar1_load,config,layer_scale)
def resnet8_test(ar1_load,config,layer_scale):
    return ResNet8(0,0,0)

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
# Example usage
# Define device as the first visible cuda device if available
# # 创建网络实例
# model = ResNet8(num_classes=10)
# # 打印网络结构
# print(model)
#
# # 创建一个模拟的32x32x3输入张量
# input_tensor = torch.randn(10, 3, 32, 32)
# # 获取模型输出
# output = model(input_tensor)
# # 打印输出
# print((output))


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# # global llayer_counterayer_counter[0]
# layer_counter = [0]
# layer_counter[0] = 0
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, intermediate_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0,bias = False)
#         self.bn1 = nn.BatchNorm2d(intermediate_channels)
#         self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=2, padding=1,bias = False)
#         self.bn2 = nn.BatchNorm2d(intermediate_channels)
#         self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0,bias = False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#
#         self.shortcut = nn.Sequential()
#         global layer_counter
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,bias = False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         #here add approx
#         out = F.relu(self.bn2(self.conv2(out)))
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         #here add approx
#         out = self.bn3(self.conv3(out))
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         #here add approx
#         out += self.shortcut(x)
#         out = F.relu(out)
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         #here add approx
#         return out
#
# class ResNet8(nn.Module):
#     def __init__(self, num_classes=3):
#         global layer_counter
#         super(ResNet8, self).__init__()
#         # 调整初始卷积层以适应32x32x3输入
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2,bias = False)
#         self.bn1 = nn.BatchNorm2d(32)
#
#         # 调整池化层步长以适应32x32x3输入
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # 保持残差块的步长为1以避免减少特征图尺寸
#         self.block1 = ResidualBlock(32, 32, 32, stride=2)
#         self.block2 = ResidualBlock(32, 64, 64, stride=2)
#         self.block3 = ResidualBlock(64, 128, 128, stride=2)
#         # 调整全连接层的输入特征数
#         self.fc1 = nn.Linear(128, 10)
#
#         # self.fc2 = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         print(layer_counter)
#         #here add approx
#         out = self.pool(out)
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         print(layer_counter)
#         #here add approx
#         out = self.block1(out)
#         print(layer_counter)
#         out = self.block2(out)
#         print(layer_counter)
#         out = self.block3(out)
#         print(layer_counter)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         np.savetxt(f"ResNet_data/ResNet8/layer{layer_counter[0]}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
#         layer_counter[0] += 1
#         return out
#
# # 创建网络实例
# # model = ResNet8(num_classes=10)
# # # 打印网络结构
# # print(model)
# #
# # # 创建一个模拟的32x32x3输入张量
# # input_tensor = torch.randn(10, 3, 32, 32)
# # # 获取模型输出
# # output = model(input_tensor)
# # 打印输出
# # print((output))