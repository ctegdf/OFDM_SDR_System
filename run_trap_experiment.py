import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.interpolate import interp1d

# 路径修复
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from ofdm_system import OFDM_System


# --- 1. 定义模型 (DNN) ---
class DNN_Channel_Estimator(nn.Module):
    def __init__(self):
        super(DNN_Channel_Estimator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x): return self.layers(x)


# 加载模型
model = DNN_Channel_Estimator()
try:
    path = "dnn_model.pth"
    if not os.path.exists(path): path = "ai_training/dnn_model.pth"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
except:
    print("❌ 模型未加载，请确保 dnn_model.pth 存在")
    exit()

# --- 2. 准备系统 ---
ofdm = OFDM_System(K=64, CP=16, P=8, modulation='16QAM')
K, CP, P = 64, 16, 8

# --- 3. 构造“必死”信道 (陷阱) ---
# 这是一个极其极端的信道，两径几乎抵消
h_trap = np.array([1.0, 0.9])
h_trap = h_trap / np.linalg.norm(h_trap)  # 归一化

# 算出它的真实频域响应 (上帝视角)
H_true_trap = np.fft.fft(h_trap, n=K)

# --- 4. 发射数据 ---
# 发射足够多的数据来看清分布
N_BLOCKS = 50
bits = np.random.randint(0, 2, N_BLOCKS * len(ofdm.dataCarriers) * 4)
tx_signal = ofdm.transmit(bits)

# --- 5. 手动过信道 (不加噪声!) ---
# 重点：为了看清 AI 的几何能力，我们先把噪声关掉 (SNR=无穷大)
# 这样你看到的误差，全都是“算法太笨”导致的，而不是“环境太吵”导致的
rx_signal = np.convolve(tx_signal, h_trap, mode='full')[:len(tx_signal)]
# 这里不加 np.random.normal，只看畸变

# --- 6. 接收处理 ---
sl = K + CP
rx_matrix = rx_signal.reshape(sl, -1, order='F')[CP:, :]
rxf = np.fft.fft(rx_matrix, axis=0)

# 提取导频
rx_pilots = rxf[ofdm.pilotCarriers, :]
H_ls = rx_pilots / (1 + 0j)

# --- 方法 A: 线性插值 ---
H_linear = np.zeros((K, N_BLOCKS), dtype=complex)
for i in range(N_BLOCKS):
    func = interp1d(ofdm.pilotCarriers, H_ls[:, i], kind='linear', fill_value="extrapolate")
    H_linear[:, i] = func(ofdm.allCarriers)

# --- 方法 B: AI ---
feat = np.hstack([H_ls.T.real, H_ls.T.imag])
inp = torch.FloatTensor(feat)
with torch.no_grad():
    out = model(inp).numpy()
H_ai = (out[:, :64] + 1j * out[:, 64:]).T

# --- 7. 均衡后的星座图 ---
# 我们只看第 30 到 40 号子载波，因为那是“深坑”通常所在的地方
# 如果看全部子载波，好的点会掩盖坏的点
mask = ofdm.dataCarriers

sym_linear = (rxf / H_linear)[mask, :].flatten()
sym_ai = (rxf / H_ai)[mask, :].flatten()

# --- 8. 绘图：真相大白 ---
plt.figure(figsize=(14, 6))

# 图1: 信道长什么样？
plt.subplot(1, 3, 1)
plt.plot(np.abs(H_true_trap), 'k-', linewidth=2, label='True Channel (The Trap)')
plt.plot(np.abs(H_linear[:, 0]), 'g--', label='Linear Interp (Ignorant)')
plt.plot(np.abs(H_ai[:, 0]), 'r--', label='AI (Smart)')
plt.title("Frequency Response (Look at the DIP!)")
plt.legend()
plt.grid(True)

# 图2: 线性插值的星座图
plt.subplot(1, 3, 2)
plt.scatter(sym_linear.real, sym_linear.imag, c='r', s=2, alpha=0.5)
plt.title("Linear Interpolation\n(Distorted!)")
plt.grid(True);
plt.xlim(-2, 2);
plt.ylim(-2, 2)

# 图3: AI 的星座图
plt.subplot(1, 3, 3)
plt.scatter(sym_ai.real, sym_ai.imag, c='g', s=2, alpha=0.5)
plt.title("AI Estimator\n(Corrected!)")
plt.grid(True);
plt.xlim(-2, 2);
plt.ylim(-2, 2)

plt.tight_layout()
plt.show()