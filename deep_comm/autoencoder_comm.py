import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# --- 1. 定义通信系统 (Tx + Channel + Rx) ---
class AutoencoderComm(nn.Module):
    def __init__(self, M=16, n_channel=2):
        super(AutoencoderComm, self).__init__()
        self.M = M  # 消息种类数 (比如 16 种消息，相当于 16-QAM)
        self.n_channel = n_channel  # 信道维度 (2 代表实部和虚部，即 1个复数符号)

        # --- 发射机 (Transmitter) ---
        # 输入: M维 one-hot 向量 -> 输出: 2维 (I, Q)
        self.transmitter = nn.Sequential(
            nn.Linear(M, 32),
            nn.ReLU(),
            nn.Linear(32, n_channel)
            # 注意：这里没有 Tanh 或 Sigmoid，我们允许 AI 发射任意能量的信号
            # 但我们需要在 forward 里手动做归一化
        )

        # --- 接收机 (Receiver) ---
        # 输入: 2维 (I, Q) + 噪声 -> 输出: M维 (概率)
        self.receiver = nn.Sequential(
            nn.Linear(n_channel, 32),
            nn.ReLU(),
            nn.Linear(32, M),
            nn.Softmax(dim=1)  # 输出概率分布
        )

    def forward(self, x, noise_std=0.1):
        # 1. 编码 (Tx)
        tx_signal = self.transmitter(x)

        # 2. 能量归一化 (Power Normalization)
        # 这是一个物理约束：发射机的平均功率不能无限大，必须限制为 1
        # E[x^2] = 1
        # 计算当前 batch 的平均能量
        # 这种归一化技巧让 AI 必须在有限能量下优化分布
        n_power = torch.mean(tx_signal ** 2)
        tx_signal = tx_signal / torch.sqrt(n_power * 2)  # *2 是为了匹配复数功率定义

        # 3. 经过信道 (Channel)
        # 加高斯白噪声 (AWGN)
        noise = torch.randn_like(tx_signal) * noise_std
        rx_signal = tx_signal + noise

        # 4. 解码 (Rx)
        reconstructed = self.receiver(rx_signal)

        return reconstructed, tx_signal  # 把中间的发射信号也返回，我们要看星座图


# --- 2. 训练准备 ---
M = 16  # 试图模仿 16-QAM
model = AutoencoderComm(M=M)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # 分类任务常用的 Loss

# 准备 One-hot 数据 (单位矩阵就是最好的 One-hot 集合)
# 比如 M=4: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
data = torch.eye(M)

# --- 3. 训练循环 ---
print(f"🚀 AI 正在发明一种新的 {M}-点 调制方式...")
loss_history = []

for epoch in range(2000):
    # 随机生成一批消息索引
    batch_indices = torch.randint(0, M, (1000,))  # 随机选 1000 个数
    batch_inputs = data[batch_indices]  # 转成 One-hot: (1000, 16)

    # 训练 (设置一点噪声，逼迫 AI 把点拉开)
    # SNR = 7dB 左右
    optimizer.zero_grad()
    outputs, tx_sig = model(batch_inputs, noise_std=0.1)

    # 计算 Loss (输入是 label index，输出是概率)
    loss = criterion(outputs, batch_indices)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# --- 4. 绘图：AI 发明了什么？ ---
plt.figure(figsize=(12, 5))

# 图1: Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")

# 图2: AI 发明的星座图
plt.subplot(1, 2, 2)
# 这一步我们要把 16 个标准符号输进去，看看它映射到了哪里
with torch.no_grad():
    _, constellation = model(data, noise_std=0.0)  # 无噪声查看
    constellation = constellation.numpy()

plt.scatter(constellation[:, 0], constellation[:, 1], c='r', s=100, marker='o')
for i in range(M):
    plt.text(constellation[i, 0] + 0.1, constellation[i, 1] + 0.1, str(i))

# 画个圆圈表示能量边界
circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
plt.gca().add_patch(circle)

plt.title(f"AI-Learned Constellation (M={M})")
plt.grid(True)
plt.axis('equal')
plt.show()