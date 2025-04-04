import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载训练集
train_dataset = torchvision.datasets.ImageFolder(root='pupil_dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 加载测试集
test_dataset = torchvision.datasets.ImageFolder(root='pupil_dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印数据集信息
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 示例：遍历训练集
for images, labels in train_loader:
    print(f"图像形状: {images.shape}")
    print(f"标签: {labels}")
    break