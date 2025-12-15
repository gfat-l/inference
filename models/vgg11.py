import torch
import torch.nn as nn
import numpy as np

class VGG11(nn.Module):
    def __init__(self,ar1_load,config,layer_scale):
        super(VGG11, self).__init__()
        self.ar1_load = ar1_load
        self.config = config
        self.layer_scale = layer_scale
        # 定义 VGG11 的特征提取层（卷积层和池化层）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias = False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias = False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias = False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias = False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias = False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias = False)
        self.bn8 = nn.BatchNorm2d(512)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        layer_counter = 0

        # 独立执行每个操作
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])
        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.pool(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.pool(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.pool(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.pool(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.functional.relu(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        x = self.pool(x)
        x = approx(x,self.ar1_load[layer_counter][3],self.ar1_load[layer_counter][2],self.ar1_load[layer_counter][1],self.ar1_load[layer_counter][0],self.layer_scale[layer_counter],self.config[layer_counter])

        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 将特征图扁平化以输入到全连接层
        x = x.view(x.size(0), -1)

        # 应用全连接层
        x = self.fc(x)
        #np.savetxt(f"VGG_data/vgg11/layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        return x


def vgg11_find_config(ar1_load,config,layer_scale):
    return VGG11(ar1_load,config,layer_scale)
def vgg11_test(ar1_load,config,layer_scale):
    return VGG11(0,0,0)


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