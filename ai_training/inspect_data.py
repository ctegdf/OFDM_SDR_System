import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
data = np.load("ofdm_dataset.npz")
X_complex = data['X']  # (10000, 8)
Y_complex = data['Y']  # (10000, 64)

print(f"原始数据形状: X={X_complex.shape}, Y={Y_complex.shape}")

# 2. 复数 -> 实数 (数据清洗)
# 策略：把实部和虚部拼接在一起
# X: (N, 8) -> (N, 16)
X_real = np.hstack([X_complex.real, X_complex.imag])
# Y: (N, 64) -> (N, 128)
Y_real = np.hstack([Y_complex.real, Y_complex.imag])

print(f"PyTorch就绪形状: X={X_real.shape}, Y={Y_real.shape}")

# 3. 可视化：看看我们造的信道长什么样
# 随机挑 3 个样本看看
sample_indices = np.random.choice(len(X_complex), 3)

plt.figure(figsize=(12, 8))

for i, idx in enumerate(sample_indices):
    h_true = Y_complex[idx]  # 真实的 64 点信道
    h_pilot = X_complex[idx]  # 接收到的 8 点导频 (含噪声)

    # 算出导频在 0-63 中的索引位置
    # (假设 K=64, P=8)
    pilot_idx = np.arange(0, 64, 8)

    plt.subplot(3, 1, i + 1)

    # 画出真实的信道幅度 (上帝视角)
    plt.plot(np.abs(h_true), 'b-', label='True Channel (Label)')

    # 画出导频看到的点 (输入特征)
    plt.plot(pilot_idx, np.abs(h_pilot), 'ro', label='Noisy Pilots (Input)')

    plt.title(f"Sample {idx} - Magnitude Response")
    plt.grid(True)
    if i == 0: plt.legend()

plt.tight_layout()
plt.show()