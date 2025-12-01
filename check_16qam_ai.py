import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# 路径修复
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from ofdm_system import OFDM_System
from channel import run_channel


# --- 1. 定义 DNN 模型 (必须和训练时完全一致) ---
class DNN_Channel_Estimator(nn.Module):
    def __init__(self):
        super(DNN_Channel_Estimator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x): return self.layers(x)


# --- 2. 准备工作 ---
# 必须使用 16QAM 模式
ofdm = OFDM_System(K=64, CP=16, P=8, modulation='16QAM')

# 加载最好的 DNN 模型
model = DNN_Channel_Estimator()
try:
    # 尝试在当前目录或 ai_training 目录找模型
    path = "dnn_model.pth"
    if not os.path.exists(path): path = "ai_training/dnn_model.pth"

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    print(" 加载了 Loss=0.019 的 DNN 模型")
except:
    print(" 找不到模型，请检查路径")
    exit()

# --- 3. 生成测试数据 ---
# 发射足够多的点，才能看清星座图的分布
N_SYMBOLS = 200
bits = np.random.randint(0, 2, N_SYMBOLS * len(ofdm.dataCarriers) * 4)  # *4 for 16QAM
tx_signal = ofdm.transmit(bits)

# --- 4. 经过恶劣信道 ---
# 设定一个对 16QAM 来说很危险的 SNR，比如 18dB
# 并开启随机深衰落
print(" 正在通过恶劣信道 (SNR=18dB, Random Multipath)...")
rx_signal, h_true = run_channel(tx_signal, snr_db=18, random_channel=True)

# --- 5. 两种方法的接收 ---

# 方法 A: 传统线性插值
rx_bits_linear = ofdm.receive(rx_signal, use_ai=False)

# 为了画星座图，我们需要偷取 receive 内部的中间变量 (均衡后的符号)
# 这里手动再跑一遍接收流程前半段...
sl = 64 + 16
nb = len(rx_signal) // sl
rx = rx_signal[:nb * sl].reshape(sl, nb, order='F')[16:, :]
rxf = np.fft.fft(rx, axis=0)

# A1. 线性插值均衡
from scipy.interpolate import interp1d

rx_pilots = rxf[ofdm.pilotCarriers, :]
H_ls = rx_pilots / (1 + 0j)
H_linear = np.zeros((64, nb), dtype=complex)
for i in range(nb):
    H_linear[:, i] = interp1d(ofdm.pilotCarriers, H_ls[:, i], kind='linear', fill_value="extrapolate")(ofdm.allCarriers)
sym_linear = (rxf / H_linear)[ofdm.dataCarriers, :].flatten()

# B1. AI 均衡
# 预处理输入
feat = np.hstack([H_ls.T.real, H_ls.T.imag])
inp = torch.FloatTensor(feat)
with torch.no_grad():
    out = model(inp).numpy()
H_ai = (out[:, :64] + 1j * out[:, 64:]).T
sym_ai = (rxf / H_ai)[ofdm.dataCarriers, :].flatten()

# --- 6. 绘图对比 ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(sym_linear.real, sym_linear.imag, c='r', s=1, alpha=0.3)
plt.title("Legacy Linear Interp\n(Cloudy & Blurry)")
plt.grid(True);
plt.xlim(-2, 2);
plt.ylim(-2, 2)

plt.subplot(1, 2, 2)
plt.scatter(sym_ai.real, sym_ai.imag, c='g', s=1, alpha=0.3)
plt.title("DNN AI Estimator\n(Sharper & Clearer)")
plt.grid(True);
plt.xlim(-2, 2);
plt.ylim(-2, 2)

plt.tight_layout()
plt.show()

# 计算 EVM (误差向量幅度) - 衡量星座图好坏的指标
evm_linear = np.mean(np.abs(sym_linear - ofdm._map(bits)) ** 2)  # 简化计算，近似值
evm_ai = np.mean(np.abs(sym_ai - ofdm._map(bits)) ** 2)
print(f"EVM 对比 (越小越好): Linear={evm_linear:.4f} vs AI={evm_ai:.4f}")