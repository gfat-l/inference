import torch
import torch.nn as nn
import numpy as np
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.relu6 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        layer_counter = 1

        # 第一层
        x = self.conv1(x)
        x = self.relu1(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = self.pool1(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 第二层
        x = self.conv2(x)
        x = self.relu2(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = self.pool2(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 第三层
        x = self.conv3(x)
        x = self.relu3(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 第四层
        x = self.conv4(x)
        x = self.relu4(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 第五层
        x = self.conv5(x)
        x = self.relu5(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1
        x = self.pool3(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # 第六层
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu6(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 第七层
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu7(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")
        layer_counter += 1

        # 输出层
        x = self.fc3(x)
        np.savetxt(f"layer{layer_counter}.txt", x.cpu().detach().numpy().ravel(), fmt="%.7f")

        return x

def Alexnet_test():
    return AlexNet()