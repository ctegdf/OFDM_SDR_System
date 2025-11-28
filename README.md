# 📡 PyOFDM: A Python-based OFDM Communication System

![System Evolution](ofdm_evolution.gif)

*(Above: Real-time visualization of image recovery as SNR increases from -5dB to 25dB)*

## 🧠 v2.0 Update: Deep Learning Powered Receiver

I have integrated a **Deep Neural Network (DNN)** to replace the traditional linear interpolation for channel estimation.

### AI vs Legacy Algorithm
The AI model successfully detects "Deep Fading" holes in the frequency domain that linear interpolation misses.

![AI Comparison](ai_comparison.png)
*(Result on 512x512 Lena image. **Legacy BER: 18.6% vs AI BER: 7.9%**. Note how the vertical stripes (deep fading errors) are removed by the AI.)*

### How to reproduce:
1. Generate training data: `python ai_training/generate_dataset.py`
2. Train the model: `python ai_training/train_model.py`
3. Run comparison: `python comparison_demo.py`

## 📖 Introduction
This project implements a full-chain **Orthogonal Frequency Division Multiplexing (OFDM)** communication system simulation in Python. It demonstrates how digital images are converted into waveforms, transmitted through a noisy multipath channel, and recovered at the receiver.

It is designed for students and researchers to understand the physical layer (PHY) of modern wireless standards like WiFi (802.11) and 5G NR.

## ✨ Key Features
- **Tx/Rx Chain**: Complete implementation of Mapping, IFFT/FFT, CP insertion/removal, and Equalization.
- **Modulation**: QPSK (Quadrature Phase Shift Keying).
- **Channel Model**:
  - Multipath Fading (ISI distortion).
  - AWGN (Additive White Gaussian Noise).
- **Visualization**: Real-time constellation diagrams and BER (Bit Error Rate) analysis.
- **Zero Dependencies**: Built purely on `numpy` and `scipy` (opencv only for image loading).

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ctegdf/OFDM_SDR_System.git