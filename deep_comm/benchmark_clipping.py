import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. 终极版 Autoencoder (固定星座点架构)
# ==========================================
class AutoencoderComm(nn.Module):
    def __init__(self, M=16):
        super(AutoencoderComm, self).__init__()
        self.M = M

        # --- 发射机 (Tx) ---
        # 抛弃 Linear 层，直接定义 M 个可训练的坐标点
        # 形状: (16, 2) -> 16个点，每个点有实部和虚部
        # 初始化为随机，让 AI 自己去找最佳位置
        self.constellation = nn.Parameter(torch.randn(M, 2))

        # --- 接收机 (Rx) ---
        # 加宽网络，提升在高 SNR 下的判决精度
        self.receiver = nn.Sequential(
            nn.Linear(2, 128),  # 升维到 128
            nn.ReLU(),
            nn.Linear(128, 128),  # 再加一层深层特征
            nn.ReLU(),
            nn.Linear(128, M)  # 输出概率
        )

    def forward(self, indices, noise_std):
        # 1. 查表
        tx_signal = self.constellation[indices]

        # 2. 能量归一化
        avg_power = torch.mean(self.constellation ** 2)
        tx_signal = tx_signal / torch.sqrt(avg_power * 2)

        # --- 3. 新增：模拟廉价功放的削波 (Clipping) ---
        # 假设功放最大只能输出幅值为 0.9 的信号
        # 任何超过 0.9 的点都会被强行削平
        # 这对于标准 QAM/APSK 是毁灭性的
        CLIP_THRESHOLD = 0.9

        # 计算幅度
        amp = torch.sqrt(torch.sum(tx_signal ** 2, dim=1, keepdim=True))
        # 这是一个 differentiable 的写法：如果 amp > threshold, 就缩放它
        scale = torch.where(amp > CLIP_THRESHOLD, CLIP_THRESHOLD / amp, torch.ones_like(amp))
        tx_clipped = tx_signal * scale

        # ---------------------------------------------

        # 4. 信道 (AWGN)
        noise = torch.randn_like(tx_clipped) * noise_std
        rx_signal = tx_clipped + noise

        # 5. 解码
        output = self.receiver(rx_signal)
        return output, tx_clipped  # 返回被削波后的信号看看


# ==========================================
# 2. 训练配置
# ==========================================
M = 16
model = AutoencoderComm(M=M)
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 学习率衰减：最后冲刺阶段降速，求稳
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 15000], gamma=0.2)
criterion = nn.CrossEntropyLoss()

print(" AI 正在接受特训 (Fixed Constellation Training)...")

# 加大 Batch Size 到 5000，消除梯度噪声
BATCH_SIZE = 5000
N_EPOCHS = 20000

for i in range(N_EPOCHS):
    # 生成随机索引
    idx = torch.randint(0, M, (BATCH_SIZE,))

    # 动态 SNR 训练 (0dB - 20dB)
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
# 3. 评测逻辑
# ==========================================
# 准备人类选手 (标准 16-QAM)
qam_points = np.array([r + 1j * i for r in [-3, -1, 1, 3] for i in [-3, -1, 1, 3]])
qam_points = qam_points / np.sqrt(10)
qam_tensor = torch.from_numpy(np.stack([qam_points.real, qam_points.imag], axis=1)).float()


# --- 修正后的完整函数，请直接覆盖原来的 run_benchmark ---
def run_benchmark(snr_db):
    # SNR (dB) -> Noise Std
    noise_power = 10 ** (-snr_db / 10)
    noise_std = np.sqrt(noise_power / 2)

    N_TEST = 100000
    labels = torch.randint(0, M, (N_TEST,))

    # --- Round 1: AI (Autoencoder) ---
    with torch.no_grad():
        # AI 内部已经有了削波逻辑 (在 forward 函数里)
        outputs, _ = model(labels, noise_std=noise_std)
        pred_ai = torch.argmax(outputs, dim=1)
        err_ai = torch.sum(pred_ai != labels).item()

    # --- Round 2: 16-QAM (Human) ---
    tx_qam = qam_tensor[labels]

    # === 关键：手动给 QAM 加上同样的削波 (Clipping) ===
    # 否则对 QAM 太不公平了（AI 被削了，QAM 没被削）
    CLIP_THRESHOLD = 0.9

    # 计算幅度
    amp = torch.sqrt(torch.sum(tx_qam ** 2, dim=1, keepdim=True))
    # 如果幅度 > 0.9，就缩放它；否则保持原样
    scale = torch.where(amp > CLIP_THRESHOLD, CLIP_THRESHOLD / amp, torch.ones_like(amp))
    tx_qam_clipped = tx_qam * scale

    # 加噪声
    rx_qam = tx_qam_clipped + torch.randn_like(tx_qam_clipped) * noise_std

    # 计算接收到的点到 16 个标准 QAM 点的距离
    # rx_qam: (N, 2), qam_tensor: (16, 2)
    dists = torch.sum((rx_qam.unsqueeze(1) - qam_tensor.unsqueeze(0)) ** 2, dim=2)

    # 找距离最近的点作为判决结果
    pred_qam = torch.argmin(dists, dim=1)

    # 统计错误
    err_qam = torch.sum(pred_qam != labels).item()
    # ==========================================

    return err_ai / N_TEST, err_qam / N_TEST


# ==========================================
# 4. 开战
# ==========================================
snr_range = np.arange(0, 16, 1)  # 测到 15dB
ser_ai = []
ser_qam = []

print(f"\n  (Testing SNR 0-15dB)...")
for snr in snr_range:
    err_ai, err_qam = run_benchmark(snr)
    ser_ai.append(err_ai)
    ser_qam.append(err_qam)
    # 打印差异百分比
    diff = (err_qam - err_ai) / (err_qam + 1e-9) * 100
    win_str = "AI WIN" if err_ai < err_qam else "AI LOSS"
    print(f"SNR {snr}dB | AI: {err_ai:.5f} | QAM: {err_qam:.5f} | {win_str} ({diff:.1f}%)")

# 绘图
plt.figure(figsize=(8, 6))
plt.semilogy(snr_range, ser_qam, 'b-o', label='Standard 16-QAM')
plt.semilogy(snr_range, ser_ai, 'r-s', label='AI Autoencoder (Stable)')
plt.title("SER: AI (Trainable Constellation) vs Human")
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate")
plt.legend()
plt.grid(True, which="both", linestyle='--')
plt.show()