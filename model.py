import torch
import torch.nn as nn

class PupilNet(nn.Module):
    def __init__(self):
        super(PupilNet, self).__init__()
        # 输入通道改为3，因为图像是RGB的
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        # 输出4个值：x, y, width, height
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # 224/2^3=28
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)  # 输出4个坐标值

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x