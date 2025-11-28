import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ofdm_system import OFDM_System
from channel import run_channel

# --- 1. 必须重新定义一遍模型结构才能加载 ---
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

# --- 2. 加载模型 ---
device = torch.device("cpu") # 推理用 CPU 足够了
model = DNN_Channel_Estimator()
try:
    model.load_state_dict(torch.load("dnn_model.pth", map_location=device))
    model.eval() # 切换到评估模式 (这一步很重要!)
    print(" AI 模型加载成功！")
except:
    print(" 找不到模型文件，请先运行 train_model.py")
    exit()

# --- 3. 准备测试环境 ---
K = 64; CP = 16; P = 8
ofdm = OFDM_System(K=K, CP=CP, P=P)
pilotCarriers = ofdm.pilotCarriers
allCarriers = ofdm.allCarriers

# --- 4. 生成一个恶劣的测试样本 ---
# 我们生成 1 个随机 OFDM 符号
bits_per_symbol = len(ofdm.dataCarriers) * ofdm.mu
tx_bits = np.random.randint(0, 2, bits_per_symbol)
tx_signal = ofdm.transmit(tx_bits)

# 强制使用随机信道，SNR=15dB (噪声适中，主要看多径)
rx_signal, h_time_true = run_channel(tx_signal, snr_db=15, random_channel=True)

# --- 5. 获取三种信道数据 ---

# A. 真实信道 (Ground Truth) - 上帝视角
H_true = np.fft.fft(h_time_true, n=K)

# B. 接收机的输入 (LS Estimation at Pilots)
# 模拟接收处理
rx_serial = rx_signal[:K+CP]
rx_no_cp = rx_serial[CP:]
rx_freq = np.fft.fft(rx_no_cp)
rx_pilots = rx_freq[pilotCarriers]
H_ls_pilots = rx_pilots / (1.0 + 0j) # 只有 8 个点

# C. 选手1: 传统线性插值 (Linear)
interp_func = interp1d(pilotCarriers, H_ls_pilots, kind='linear', fill_value="extrapolate")
H_linear = interp_func(allCarriers)

# D. 选手2: AI 模型 (Deep Learning)
# 预处理: 复数 -> 实数 (拼接)
input_feature = np.hstack([H_ls_pilots.real, H_ls_pilots.imag]) # Shape (16,)
input_tensor = torch.FloatTensor(input_feature).unsqueeze(0) # Add batch dim -> (1, 16)

# AI 推理
with torch.no_grad():
    output_tensor = model(input_tensor) # -> (1, 128)

# 后处理: 实数 -> 复数 (拆分)
output_np = output_tensor.numpy().flatten()
H_ai_real = output_np[:64]
H_ai_imag = output_np[64:]
H_ai = H_ai_real + 1j * H_ai_imag

# --- 6. 绘图决胜 ---
plt.figure(figsize=(10, 6))

# 画幅度响应 (Magnitude)
plt.plot(allCarriers, np.abs(H_true), 'k-', linewidth=2, label='Ground Truth (God View)')
plt.plot(allCarriers, np.abs(H_linear), 'g--', linewidth=1.5, label='Linear Interp (Traditional)')
plt.plot(allCarriers, np.abs(H_ai), 'r-', linewidth=2, label='AI Model (Ours)')

# 画出导频点 (接收机只能看到这些)
plt.plot(pilotCarriers, np.abs(H_ls_pilots), 'bo', markersize=8, label='Pilots Input')

plt.title("Channel Estimation: AI vs Linear")
plt.xlabel("Subcarrier Index")
plt.ylabel("Channel Magnitude |H|")
plt.legend()
plt.grid(True)
plt.show()

# 计算一下 MSE 误差对比
mse_linear = np.mean(np.abs(H_true - H_linear)**2)
mse_ai = np.mean(np.abs(H_true - H_ai)**2)
print(f" 误差对比 (越小越好):")
print(f"传统线性插值 MSE: {mse_linear:.5f}")
print(f"AI 模型预测 MSE: {mse_ai:.5f}")
if mse_ai < mse_linear:
    print(" 结论: AI 获胜！")
else:
    print(" 结论: AI 惜败 ")