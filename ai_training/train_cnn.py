import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys, os

# 路径修复 (老规矩)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


# --- 1. 新的 Dataset: 输出 (2, 64) 的张量 ---
class CNNDataset(Dataset):
    def __init__(self, npz_file, K=64, P=8):
        data = np.load(npz_file)
        self.X = data['X']  # (N, 8) 导频
        self.Y = data['Y']  # (N, 64) 真值

        # 定义导频位置 (用于插值)
        self.allCarriers = np.arange(K)
        self.pilotCarriers = self.allCarriers[::K // P]

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        h_pilot = self.X[idx]  # (8,)
        h_true = self.Y[idx]  # (64,)

        # --- 核心变化: 先做个粗糙的线性插值 (Pre-processing) ---
        # 这样 AI 拿到的就是一张完整的图 (虽然不准)
        interp_func = interp1d(self.pilotCarriers, h_pilot, kind='linear', fill_value="extrapolate")
        h_coarse = interp_func(self.allCarriers)  # (64,)

        # --- 转成 CNN 喜欢的形状: (Channel, Length) ---
        # Channel 0: 实部, Channel 1: 虚部
        # Input (X): 粗糙估计
        x_tensor = torch.zeros(2, 64)
        x_tensor[0, :] = torch.from_numpy(h_coarse.real)
        x_tensor[1, :] = torch.from_numpy(h_coarse.imag)

        # Label (Y): 完美真值
        y_tensor = torch.zeros(2, 64)
        y_tensor[0, :] = torch.from_numpy(h_true.real)
        y_tensor[1, :] = torch.from_numpy(h_true.imag)

        return x_tensor.float(), y_tensor.float()


# --- 2. CNN 模型 (ResNet) ---
class CNN_Channel_Estimator(nn.Module):
    def __init__(self):
        super(CNN_Channel_Estimator, self).__init__()

        # 1. 第一层
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),  # <--- 新增
            nn.ReLU()
        )

        # 2. 中间层 (加 BN)
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),  # <--- 新增
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),  # <--- 新增
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  # <--- 新增
            nn.ReLU()
        )

        # 3. 输出层 (最后一层通常不加 BN 和 ReLU，直接输出)
        self.layer5 = nn.Conv1d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        residual = self.layer5(out)
        return x + residual


# --- 3. 训练流程 (和之前差不多) ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" CNN Training on {device}...")

    # 确保你有数据 (如果没有，请先运行 generate_dataset.py)
    if not os.path.exists("ofdm_dataset.npz"):
        print(" 没找到 ofdm_dataset.npz，请先运行 generate_dataset.py")
        return

    dataset = CNNDataset("ofdm_dataset.npz")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CNN_Channel_Estimator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20  # CNN 收敛通常比 DNN 快

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), "cnn_model.pth")
    print(" CNN 模型已保存: cnn_model.pth")


if __name__ == "__main__":
    train()