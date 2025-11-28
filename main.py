import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- 关键修改：把 src 目录加入搜索路径，这样才能 import ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 从 src 文件夹导入我们的积木
from ofdm_system import OFDM_System
from channel import run_channel


def main_test(image_path, target_snr=25):
    print(f"--- 开始测试: {image_path} (SNR={target_snr}dB) ---")

    # 1. 读取并处理图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("错误：找不到图片，正在生成测试图...")
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:44, :] = 255;
        img[:, 20:44] = 255

    img = cv2.resize(img, (100, 100))
    _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    tx_bits = img_bin.flatten()
    print(f"1. 图片处理完毕: {len(tx_bits)} bits")

    # 2. 实例化系统
    ofdm = OFDM_System(K=64, CP=16, P=8)

    # 3. 发射
    print("2. OFDM 调制发射...")
    tx_signal = ofdm.transmit(tx_bits)

    # 4. 信道
    print(f"3. 通过信道 (SNR={target_snr}dB)...")
    rx_signal, _ = run_channel(tx_signal, target_snr)

    # 5. 接收
    print("4. 接收解调...")
    rx_bits = ofdm.receive(rx_signal)
    rx_bits = rx_bits[:len(tx_bits)]  # 截断 Padding

    # 6. 分析
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    print(f"5. 误码率(BER) = {ber:.6f}")

    # 7. 绘图
    try:
        rx_img = rx_bits.reshape(img.shape)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1);
        plt.imshow(img_bin, cmap='gray');
        plt.title("Tx")
        plt.subplot(1, 2, 2);
        plt.imshow(rx_img, cmap='gray');
        plt.title(f"Rx (SNR={target_snr}dB)")
        plt.show()
    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == "__main__":
    # 确保目录下有这张图，或者它会自动生成十字架
    main_test("test.png", target_snr=25)