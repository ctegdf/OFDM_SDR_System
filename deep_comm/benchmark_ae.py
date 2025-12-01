import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 终极版 Autoencoder (固定星座点架构 - 无削波)
# ==========================================
class AutoencoderComm(nn.Module):
    def __init__(self, M=16):
        super(AutoencoderComm, self).__init__()
        self.M = M

        # --- 发射机 (Tx) ---
        # 直接定义 M 个可训练的坐标点 (16, 2)
        # 这种方式比全连接层更稳定，不会“手抖”
        self.constellation = nn.Parameter(torch.randn(M, 2))

        # --- 接收机 (Rx) ---
        self.receiver = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, M)
        )

    def forward(self, indices, noise_std):
        # 1. 查表 (Look-up)
        tx_signal = self.constellation[indices]

        # 2. 能量归一化 (Power Normalization)
        # 保证平均功率为 1
        avg_power = torch.mean(self.constellation ** 2)
        tx_signal = tx_signal / torch.sqrt(avg_power * 2)

        # 3. 信道 (Standard AWGN - 无削波)
        noise = torch.randn_like(tx_signal) * noise_std
        rx_signal = tx_signal + noise

        # 4. 解码
        output = self.receiver(rx_signal)
        return output, tx_signal


# ==========================================
# 2. 训练配置
# ==========================================
M = 16
model = AutoencoderComm(M=M)
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 学习率调度：后期降低学习率，精细微调
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 15000], gamma=0.2)
criterion = nn.CrossEntropyLoss()

print(" AI 正在接受基准特训 (Standard AWGN Training)...")

BATCH_SIZE = 5000
N_EPOCHS = 20000

for i in range(N_EPOCHS):
    # 生成随机索引
    idx = torch.randint(0, M, (BATCH_SIZE,))

    # 动态 SNR 训练 (0dB - 25dB)
    # 这能保证模型在全信噪比范围内都表现良好
    snr_now = np.random.uniform(0, 25)
    noise_std = np.sqrt(10 ** (-snr_now / 10) / 2)

    optimizer.zero_grad()
    outputs, _ = model(idx, noise_std=noise_std)
    loss = criterion(outputs, idx)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % 2000 == 0:
        print(f"Step {i}: Loss={loss.item():.5f}")

# ==========================================
# 3. 评测逻辑 (无削波)
# ==========================================
# 准备人类选手 (标准 16-QAM)
qam_points = np.array([r + 1j * i for r in [-3, -1, 1, 3] for i in [-3, -1, 1, 3]])
qam_points = qam_points / np.sqrt(10)
qam_tensor = torch.from_numpy(np.stack([qam_points.real, qam_points.imag], axis=1)).float()


def run_benchmark(snr_db):
    noise_power = 10 ** (-snr_db / 10)
    noise_std = np.sqrt(noise_power / 2)
    N_TEST = 100000

    labels = torch.randint(0, M, (N_TEST,))

    # --- AI 预测 ---
    with torch.no_grad():
        outputs, _ = model(labels, noise_std=noise_std)
        pred_ai = torch.argmax(outputs, dim=1)
        err_ai = torch.sum(pred_ai != labels).item()

    # --- QAM 预测 (标准环境) ---
    tx_qam = qam_tensor[labels]
    # 这里直接加噪声，没有削波 (Clipping)
    rx_qam = tx_qam + torch.randn_like(tx_qam) * noise_std

    # 最小距离判决
    dists = torch.sum((rx_qam.unsqueeze(1) - qam_tensor.unsqueeze(0)) ** 2, dim=2)
    pred_qam = torch.argmin(dists, dim=1)
    err_qam = torch.sum(pred_qam != labels).item()

    return err_ai / N_TEST, err_qam / N_TEST


# ==========================================
# 4. 开战 & 绘图
# ==========================================
snr_range = np.arange(0, 16, 1)
ser_ai = []
ser_qam = []

print(f"\n 决战时刻 (Testing SNR 0-15dB)...")
for snr in snr_range:
    err_ai, err_qam = run_benchmark(snr)
    ser_ai.append(err_ai)
    ser_qam.append(err_qam)

    diff = (err_qam - err_ai) / (err_qam + 1e-9) * 100
    win_str = "AI WIN" if err_ai < err_qam else "AI LOSS"
    print(f"SNR {snr}dB | AI: {err_ai:.5f} | QAM: {err_qam:.5f} | {win_str} ({diff:.1f}%)")

# 绘图
plt.figure(figsize=(8, 6))
plt.semilogy(snr_range, ser_qam, 'b-o', label='Standard 16-QAM')
plt.semilogy(snr_range, ser_ai, 'r-s', label='AI Autoencoder (APSK)')
plt.title("SER: AI (Clean AWGN) vs Human")
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate")
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.show()

# 额外福利：画出 AI 最终学到的星座图看看
plt.figure(figsize=(5, 5))
with torch.no_grad():
    # 传入 dummy input 只是为了触发 forward，实际上我们要看的是 self.constellation
    # 但由于 forward 里有归一化逻辑，我们最好走一遍 forward 获取 tx_signal
    # 我们输入 0-15 这 16 个索引
    indices = torch.arange(M)
    _, constellation = model(indices, noise_std=0.0)
    constellation = constellation.numpy()

plt.scatter(constellation[:, 0], constellation[:, 1], c='r', s=100)
# 画个单位圆
circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
plt.gca().add_patch(circle)
plt.title("Learned Constellation (Should be Ring-like)")
plt.grid(True)
plt.axis('equal')
plt.show()