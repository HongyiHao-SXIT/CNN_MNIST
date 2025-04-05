import torch
import torch.nn as nn
import torch.optim as optim
from model import PupilNet
from data_loader import train_loader
import time

# 初始化模型
model = PupilNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for i, (inputs, labels) in enumerate(train_loader):
        if inputs is None or labels is None:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if i % 10 == 9:  # 每10个batch打印一次
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/(i+1):.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    scheduler.step(epoch_loss)
    
    # 打印epoch信息
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s')
    
    # 保存模型
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'pupil_net_epoch_{epoch+1}.pth')

print('Finished Training')