import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

# 引入积木
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ofdm_system import OFDM_System
from channel import run_channel



def main_comparison():
    # 准备图片
    img_path = "test.png"
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    except:
        img_bin = np.zeros((64, 64), dtype=int);
        img_bin[20:44, :] = 1;
        img_bin[:, 20:44] = 1
    tx_bits = img_bin.flatten()

    # 实例化两个系统
    legacy_system = OFDM_System(K=64, CP=16, P=8)
    ai_system = AI_OFDM_System("dnn_model.pth", K=64, CP=16, P=8)

    # 发射
    tx_signal = legacy_system.transmit(tx_bits)

    # --- 关键测试点：设置一个恶劣的信道 ---
    # SNR=10dB, 开启随机深衰落
    print(" 正在通过恶劣信道 (SNR=12dB, Random Deep Fading)...")
    rx_signal, _ = run_channel(tx_signal, snr_db=12, random_channel=True)

    # --- 选手 1: 传统方法接收 ---
    rx_bits_legacy = legacy_system.receive(rx_signal)[:len(tx_bits)]
    ber_legacy = np.sum(rx_bits_legacy != tx_bits) / len(tx_bits)

    # --- 选手 2: AI 方法接收 ---
    rx_bits_ai = ai_system.receive_with_ai(rx_signal)[:len(tx_bits)]
    ber_ai = np.sum(rx_bits_ai != tx_bits) / len(tx_bits)

    print(f"\n 最终战报:")
    print(f"传统算法 BER: {ber_legacy:.5f}")
    print(f"AI 算法 BER: {ber_ai:.5f}")

    # --- 绘图 ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_bin, cmap='gray');
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    rx_img_legacy = rx_bits_legacy.reshape(img_bin.shape)
    plt.imshow(rx_img_legacy, cmap='gray');
    plt.title(f"Legacy (Linear)\nBER={ber_legacy:.4f}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    rx_img_ai = rx_bits_ai.reshape(img_bin.shape)
    plt.imshow(rx_img_ai, cmap='gray');
    plt.title(f"AI Model (DNN)\nBER={ber_ai:.4f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_comparison()