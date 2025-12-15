import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # 第一段卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # 第二段卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # 第三段卷积层
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)

        # 第四段卷积层
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)

        # 第五段卷积层
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # 应用卷积层、批归一化、ReLU 激活函数以及池化层
        # 对应于每个卷积段的操作
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)

        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = self.pool(x)

        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 实例化 VGG19 模型
vgg19 = VGG19()
