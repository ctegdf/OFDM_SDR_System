import sys
import os
import numpy as np
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ofdm_system import OFDM_System
from channel import run_channel

# --- 配置参数 ---
NUM_SAMPLES = 50000  # 生成多少组数据 (建议先试 1000，确认没问题再加到 10000+)
SNR_RANGE = [5, 20]  # 训练时 SNR 随机范围 (让 AI 见过各种恶劣环境)
K = 64  # 子载波
CP = 16
P = 8  # 导频间隔


def generate_dataset():
    print(f" 正在启动数据工厂... 目标: {NUM_SAMPLES} 组数据")

    ofdm = OFDM_System(K=K, CP=CP, P=P)

    # 用来存数据的容器
    # X: 接收到的导频估计值 (Input) -> 形状 [N, 导频数量]
    # Y: 真实的信道响应 (Label)    -> 形状 [N, 所有子载波数量]
    X_list = []
    Y_list = []

    # 只需要生成一个 OFDM 符号的比特量即可 (因为我们只训练信道估计，不需要解调很长的图片)
    # 计算一个 OFDM 符号能装多少 bit
    bits_per_symbol = len(ofdm.dataCarriers) * ofdm.mu

    for i in tqdm(range(NUM_SAMPLES)):
        # 1. 生成随机比特
        tx_bits = np.random.randint(0, 2, bits_per_symbol)

        # 2. 发射
        tx_signal = ofdm.transmit(tx_bits)

        # 3. 随机信道 & 随机 SNR
        snr = np.random.uniform(SNR_RANGE[0], SNR_RANGE[1])
        # 注意：这里我们要拿到 h_time (真值)
        rx_signal, h_time = run_channel(tx_signal, snr, random_channel=True)

        # --- 4. 制作 Label (Y): 上帝视角 ---
        # 我们有了时域的 h_time，怎么知道频域 64 个子载波上的 H 是多少？
        # 答案：做 FFT！
        # 注意 FFT 的大小必须是 K (64)
        H_true = np.fft.fft(h_time, n=K)
        Y_list.append(H_true)

        # --- 5. 制作 Feature (X): 接收机的视角 ---
        # 我们需要手动做接收机的前半部分：去CP -> FFT -> 提取导频
        # (不想把这一步写在 ofdm_system.receive 里，因为我们要截取中间变量)

        # 简单模拟接收处理:
        rx_serial = rx_signal[:K + CP]  # 只取第一个符号
        rx_no_cp = rx_serial[CP:]  # 去 CP
        rx_freq = np.fft.fft(rx_no_cp)  # FFT

        # 提取导频位置的值
        rx_pilots = rx_freq[ofdm.pilotCarriers]

        # LS 估计: H_est = Rx / Tx (导频发送的是 1+0j)
        # 这就是 AI 能够看到的“带噪声的、稀疏的”输入
        H_ls = rx_pilots / (1.0 + 0j)
        X_list.append(H_ls)

    # 转换成 numpy 数组
    X_data = np.array(X_list)
    Y_data = np.array(Y_list)

    print(f"\n 数据生成完毕!")
    print(f"Feature (X) shape: {X_data.shape} (每样本 {X_data.shape[1]} 个导频点)")
    print(f"Label   (Y) shape: {Y_data.shape} (每样本 {Y_data.shape[1]} 个真实信道点)")

    # 保存文件
    np.savez("ofdm_dataset.npz", X=X_data, Y=Y_data)
    print(" 已保存为 'ofdm_dataset.npz'")


if __name__ == "__main__":
    generate_dataset()