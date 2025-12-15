'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

layer_counter = [0]


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,ar1_load,config,layer_scale, approx,stride=1):
        self.ar1_load = ar1_load
        self.config = config
        self.layer_scale = layer_scale
        self.approx = approx
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        global layer_counter
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1
        #self.save_output(out)
        out = self.bn2(self.conv2(out))
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        out = approx_neg(out,self.ar1_load[layer_counter[0]][4],self.ar1_load[layer_counter[0]][5],self.ar1_load[layer_counter[0]][6],self.ar1_load[layer_counter[0]][7],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1
        #self.save_output(out)
        out_tmp = self.shortcut(x)
        # if(self.approx):
        #     out = approx(out_tmp,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        #     out = approx_neg(out_tmp,self.ar1_load[layer_counter[0]][4],self.ar1_load[layer_counter[0]][5],self.ar1_load[layer_counter[0]][6],self.ar1_load[layer_counter[0]][7],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1
        #self.save_output(out_tmp)
        out = F.relu(out + out_tmp)
        layer_counter[0] += 1
        #self.save_output(out)
        return out
    def save_output(self, output):
        global layer_counter
        np.savetxt(f"ResNet_data/ResNet/layer{layer_counter[0]}.txt", output.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter[0] += 1
    def approx_operation(self,x,threshold,config,scale):
        global layer_counter
        # print(layer_counter)
        x = approx(x,threshold[layer_counter[0]][3],threshold[layer_counter[0]][2],threshold[layer_counter[0]][1],threshold[layer_counter[0]][0],scale[layer_counter[0]],config[layer_counter[0]])
        # print(threshold[layer_counter[0]][3])
        layer_counter[0] += 1
        return x



class ResNet_sample(nn.Module):
    def __init__(self, block, num_blocks,ar1_load,config,layer_scale, num_classes=10):
        super(ResNet_sample, self).__init__()
        self.in_planes = 6 #64 16
        self.ar1_load = ar1_load
        self.config = config
        self.layer_scale = layer_scale
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=False) #64 16
        self.bn1 = nn.BatchNorm2d(6) #64 16
        self.layer1 = self._make_layer(block, 6, num_blocks[0],ar1_load,config,layer_scale,0, stride=1) #64 16
        self.layer2 = self._make_layer(block, 16, num_blocks[1],ar1_load,config,layer_scale,1,stride=2) #128 32
        self.layer3 = self._make_layer(block, 32, num_blocks[2],ar1_load,config,layer_scale,1,stride=2) #256 64
        self.layer4 = self._make_layer(block, 64, num_blocks[3],ar1_load,config,layer_scale,1,stride=2) #512 128
        self.linear = nn.Linear(64*block.expansion, num_classes,bias = False) #512 128

    def _make_layer(self, block, planes, num_blocks,ar1_load,config,layer_scale,approx,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,ar1_load,config,layer_scale,approx,stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        global layer_counter
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1
        #self.save_output(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = approx(out,self.ar1_load[layer_counter[0]][3],self.ar1_load[layer_counter[0]][2],self.ar1_load[layer_counter[0]][1],self.ar1_load[layer_counter[0]][0],self.layer_scale[layer_counter[0]],self.config[layer_counter[0]])
        layer_counter[0] += 1
        #self.save_output(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        layer_counter[0] = 0
        #self.save_output(out)
        return out
    def save_output(self, output):
        global layer_counter
        np.savetxt(f"ResNet_data/ResNet/layer{layer_counter[0]}.txt", output.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter[0] += 1
    def approx_operation(self,x,threshold,config,scale):
        global layer_counter
        # print(layer_counter)
        x = approx(x,threshold[layer_counter[0]][3],threshold[layer_counter[0]][2],threshold[layer_counter[0]][1],threshold[layer_counter[0]][0],scale[layer_counter[0]],config[layer_counter[0]])
        # print(threshold[layer_counter[0]][3])
        layer_counter[0] += 1
        return x
def ResNet8():
    return ResNet_sample(BasicBlock, [1, 1, 1, 1])
def resnet8_find_config(ar1_load,config,layer_scale):
    return ResNet_sample(BasicBlock, [1, 1, 1, 1],ar1_load,config,layer_scale)
def resnet8_test(ar1_load,config,layer_scale):
    return ResNet_sample(BasicBlock, [1, 1, 1, 1],ar1_load,config,layer_scale)
def approx(out,level1,level2,level3,level4,scale,config):
    if(config == 3):
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,0),0,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level1),level1/scale,out)
        out = torch.where(torch.lt(out*scale,level3) & torch.gt(out*scale,level2),level2/scale,out)
        out = torch.where(torch.lt(out*scale,level4) & torch.gt(out*scale,level3),level3/scale,out)
    elif(config == 2):
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,0),0,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level1),level1/scale,out)
        out = torch.where(torch.lt(out*scale,level3) & torch.gt(out*scale,level2),level2/scale,out)
    elif(config == 1):
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,0),0,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level1),level1/scale,out)
    elif(config == 0):
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,0),0,out)
    return out
def approx_neg(out,level1,level2,level3,level4,scale,config):
    if(config == 3):
        out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),0,out)
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,level2),level1/scale,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level3),level2/scale,out)
        out = torch.where(torch.lt(out*scale,level3) & torch.gt(out*scale,level4),level3/scale,out)
    elif(config == 2):
        out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),0,out)
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,level2),level1/scale,out)
        out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level3),level2/scale,out)
    elif(config == 1):
        out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),0,out)
        out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,level2),level1/scale,out)
    elif(config == 0):
        out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),0,out)
    return  out
# def approx_neg(out,level1,level2,level3,level4,scale,config):
#     if(config == 3):
#         out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),level1/scale,out)
#         out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,level2),level2/scale,out)
#         out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level3),level3/scale,out)
#         out = torch.where(torch.lt(out*scale,level3) & torch.gt(out*scale,level4),level4/scale,out)
#     elif(config == 2):
#         out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),level1/scale,out)
#         out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,level2),level2/scale,out)
#         out = torch.where(torch.lt(out*scale,level2) & torch.gt(out*scale,level3),level3/scale,out)
#     elif(config == 1):
#         out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),level1/scale,out)
#         out = torch.where(torch.lt(out*scale,level1) & torch.gt(out*scale,level2),level2/scale,out)
#     elif(config == 0):
#         out = torch.where(torch.lt(out*scale,0) & torch.gt(out*scale,level1),level1/scale,out)
#     return  out
# test()
