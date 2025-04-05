import torch
from torchvision import transforms
from PIL import Image
from model import PupilNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PupilNet().to(device)
model.load_state_dict(torch.load('pupil_net_epoch_50.pth', map_location=device))
model.eval()

# 定义转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试单张图像
def test_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
    
    # 反归一化
    pred = output[0].cpu().numpy()
    x, y, w, h = pred
    x = x * width
    y = y * height
    w = w * width
    h = h * height
    
    # 绘制结果
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # 创建矩形框
    rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    plt.title('Pupil Detection')
    plt.show()

# 测试数据集中的图像
test_dir = 'pupil_dataset/test/eye'
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

for img_path in test_images[:5]:  # 测试前5张图像
    test_image(img_path)