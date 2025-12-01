import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ofdm_system import OFDM_System
from channel import run_channel

# 1. 启动 16-QAM 系统
ofdm = OFDM_System(modulation='16QAM')

# 2. 发射大量随机数据
bits = np.random.randint(0, 2, 10000)
tx_signal = ofdm.transmit(bits)

# 3. 加一点点噪声 (SNR=20dB，这在QPSK下是很完美的，看看16QAM咋样)
rx_signal, _ = run_channel(tx_signal, snr_db=20)

# 4. 手动解调到均衡这一步，画星座图
# (为了偷懒，我们直接copy一下 receive 的前半部分逻辑)
sl = 64 + 16
nb = len(rx_signal) // sl
rx = rx_signal[:nb*sl].reshape(sl, nb, order='F')[16:, :]
rxf = np.fft.fft(rx, axis=0)
rx_pilots = rxf[ofdm.pilotCarriers, :]
H_est = rx_pilots / (1+0j) # 粗略估计
# 简单的把导频扩展到全频段 (Nearest Neighbor) 用于画图
H_full = np.repeat(H_est, 8, axis=0)
eq_symbols = (rxf / H_full)[ofdm.dataCarriers, :].flatten()

# 5. 绘图
plt.figure(figsize=(6, 6))
plt.scatter(eq_symbols.real, eq_symbols.imag, c='b', s=1, alpha=0.5)
plt.title("16-QAM Constellation (SNR=20dB)")
plt.grid(True)
plt.xlim(-2, 2); plt.ylim(-2, 2)
plt.show()