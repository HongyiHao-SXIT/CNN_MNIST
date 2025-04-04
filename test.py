import torch
import torchvision
import torch.utils.data as Data
from model import LeNet

# 加载测试数据
test_data = torchvision.datasets.MNIST(root='./data/', train=False, transform=torchvision.transforms.ToTensor(), download=False)
test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False)

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型并加载参数
net = LeNet().to(device)
net.load_state_dict(torch.load('./LeNet.pth', map_location=torch.device(device)))

# 设置为评估模式
net.eval()

# 测试模型
length = test_data.data.size(0)
acc = 0.0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        x, y = data
        y_pred = net(x.to(device, torch.float))
        pred = y_pred.argmax(dim=1)
        acc += (pred.data.cpu() == y.data).sum()
        print('Predict:', int(pred.data.cpu()), '|Ground Truth:', int(y.data))

acc = (acc / length) * 100
print('Accuracy: %.2f' % acc, '%')