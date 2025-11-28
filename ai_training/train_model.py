import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# --- 1. Dataset (保持不变) ---
class ChannelDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = data['X']
        self.Y = data['Y']

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        # 复数转实数
        h_pilot = self.X[idx]
        h_true = self.Y[idx]
        x_val = np.hstack([h_pilot.real, h_pilot.imag])
        y_val = np.hstack([h_true.real, h_true.imag])
        return torch.FloatTensor(x_val), torch.FloatTensor(y_val)


# --- 2. Model (保持不变) ---
class DNN_Channel_Estimator(nn.Module):
    def __init__(self):
        super(DNN_Channel_Estimator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x): return self.layers(x)


# --- 3. 准备数据 ---
dataset = ChannelDataset("ofdm_dataset.npz")

# 划分 80% 训练, 20% 验证
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# --- 4. 初始化训练 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" 使用设备: {device}")

model = DNN_Channel_Estimator().to(device)
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率

# --- 5. 训练循环 (Training Loop) ---
epochs = 30  # 训练 30 轮
train_losses = []
val_losses = []

print(f"开始训练... (总共 {len(train_dataset)} 条训练数据)")

for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()  # 开启训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 1. 清空梯度
        outputs = model(inputs)  # 2. 前向传播 (猜)
        loss = criterion(outputs, labels)  # 3. 计算误差
        loss.backward()  # 4. 反向传播 (找原因)
        optimizer.step()  # 5. 更新参数 (改)

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- 验证阶段 ---
    model.eval()  # 开启评估模式 (不更新参数)
    val_running_loss = 0.0
    with torch.no_grad():  # 不算梯度，省内存
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

# --- 6. 保存模型 ---
torch.save(model.state_dict(), "dnn_model.pth")
print(" 模型已保存为 'dnn_model.pth'")

# --- 7. 画出 Loss 曲线 ---
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Process')
plt.show()