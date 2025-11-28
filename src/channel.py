import numpy as np


def run_channel(tx_signal, snr_db, random_channel=False):
    """
    random_channel: 如果为 True，生成随机多径信道；否则使用固定的测试信道。
    返回:
      rx_signal: 接收到的时域信号
      h_time:    时域信道冲击响应 (用于计算 Label)
    """

    if random_channel:
        # --- 生成随机多径信道 ---
        # 1. 决定有多少条路径 (比如 2 到 6 条)
        n_taps = np.random.randint(2, 6)

        # 2. 生成每条路径的延迟功率谱 (指数衰减模型是常用的)
        # 也就是第一径最强，后面的越来越弱
        power_profile = np.exp(-np.arange(n_taps) / 1.5)

        # 3. 生成瑞利衰落系数 (复高斯分布)
        h_time = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) * np.sqrt(power_profile / 2)

        # 4. 归一化 (让总能量为 1，这样 SNR 计算才准)
        h_time = h_time / np.linalg.norm(h_time)
    else:
        # --- 之前的固定信道 (用于测试) ---
        h_time = np.array([1.0, 0.2, 0.1])
        # 也要归一化
        h_time = h_time / np.linalg.norm(h_time)

    # --- 卷积 ---
    rx_signal = np.convolve(tx_signal, h_time, mode='full')[:len(tx_signal)]

    # --- 加噪声 ---
    sig_power = np.mean(np.abs(rx_signal) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal)) + \
            1j * np.random.normal(0, np.sqrt(noise_power / 2), len(rx_signal))

    # 这里的 h_time 对我们训练 AI 至关重要！
    return rx_signal + noise, h_time