import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from model import LeNet

# 加载训练数据
train_data = torchvision.datasets.MNIST(root='./data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型
net = LeNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# 保存模型
torch.save(net.state_dict(), './LeNet.pth')