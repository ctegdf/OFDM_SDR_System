import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 导入 src ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ofdm_system import OFDM_System
from channel import run_channel

# 1. 准备图片
image_path = "test.png"
try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise Exception("No Image")
    img = cv2.resize(img, (100, 100))
    _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
except:
    print("使用默认十字架图案")
    img_bin = np.zeros((64, 64), dtype=int)
    img_bin[20:44, :] = 1;
    img_bin[:, 20:44] = 1

tx_bits = img_bin.flatten()

# 2. 初始化系统
ofdm = OFDM_System()
tx_signal = ofdm.transmit(tx_bits)

# 3. 设置动画画布
fig, ax = plt.subplots(figsize=(6, 6))
plt.axis('off')
title = ax.set_title("OFDM Transmission Simulation")
im_display = ax.imshow(img_bin, cmap='gray', vmin=0, vmax=1)

# 定义 SNR 范围: 从 -5dB 跑到 25dB，共 60 帧
snr_values = np.linspace(-5, 25, 60)


def update(frame_idx):
    current_snr = snr_values[frame_idx]

    # 跑一次仿真
    rx_signal = run_channel(tx_signal, current_snr)
    rx_bits = ofdm.receive(rx_signal)

    # 还原
    rx_bits = rx_bits[:len(tx_bits)]
    errors = np.sum(rx_bits != tx_bits)
    ber = errors / len(tx_bits)

    # 更新画面
    try:
        rx_img = rx_bits.reshape(img_bin.shape)
        im_display.set_data(rx_img)
        title.set_text(f"SNR: {current_snr:.1f} dB | BER: {ber:.4f}")
    except:
        pass
    return im_display, title


print("正在渲染 GIF，请耐心等待...")
ani = animation.FuncAnimation(fig, update, frames=len(snr_values), interval=100, blit=True)
ani.save('ofdm_evolution.gif', writer='pillow', fps=10)
print("✅ 完成！GIF 已保存。")