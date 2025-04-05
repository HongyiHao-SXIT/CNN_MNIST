import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PupilDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 确保路径正确
        annotation_path = os.path.join(root_dir, '..', annotation_file)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file {annotation_path} not found")
            
        with open(annotation_path) as f:
            self.annotations = json.load(f)
        
        # 过滤掉没有标注的图像
        self.valid_annotations = []
        for ann in self.annotations:
            if 'annotations' in ann and len(ann['annotations']) > 0:
                self.valid_annotations.append(ann)

    def __len__(self):
        return len(self.valid_annotations)

    def __getitem__(self, idx):
        annotation = self.valid_annotations[idx]
        image_name = annotation['file_upload']
        image_path = os.path.join(self.root_dir, 'eye', image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f"Error loading image: {image_path}")
            return None, None

        # 提取标注信息并归一化
        result = annotation['annotations'][0]['result'][0]
        value = result['value']
        
        # 获取图像原始尺寸
        width_orig, height_orig = image.size
        
        # 归一化坐标和尺寸到[0,1]范围
        x = value['x'] * value['width'] / 100 / width_orig  # 转换为绝对坐标然后归一化
        y = value['y'] * value['height'] / 100 / height_orig
        width = value['width'] / 100  # 百分比转换为小数
        height = value['height'] / 100
        
        label = torch.tensor([x, y, width, height], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
])

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# 创建数据集实例
root_dir = 'pupil_dataset/train'
annotation_file = 'annotations.json'
train_dataset = PupilDataset(root_dir=root_dir, annotation_file=annotation_file, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)