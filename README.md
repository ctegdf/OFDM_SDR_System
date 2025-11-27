# 📡 PyOFDM: A Python-based OFDM Communication System

![System Evolution](ofdm_evolution.gif)
*(Above: Real-time visualization of image recovery as SNR increases from -5dB to 25dB)*

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