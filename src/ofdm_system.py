import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import os

# AI Model Definition (DNN)
class DNN_Channel_Estimator(nn.Module):
    def __init__(self):
        super(DNN_Channel_Estimator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x): return self.layers(x)

# Core System (v3.0 - Supports 16QAM)
class OFDM_System:
    def __init__(self, K=64, CP=16, P=8, modulation='QPSK'):
        self.K = K
        self.CP = CP
        self.P = P
        self.allCarriers = np.arange(K)
        self.pilotCarriers = self.allCarriers[::K // P]
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)

        # --- v3.0: Modulation Config ---
        self.modulation = modulation
        if modulation == 'QPSK':
            self.mu = 2
            self.norm_factor = np.sqrt(2)
        elif modulation == '16QAM':
            self.mu = 4
            self.norm_factor = np.sqrt(10)
        else:
            raise ValueError(f"Unsupported modulation: {modulation}")

        print(f"System Initialized: K={K}, CP={CP}, Mod={modulation}")

    # --- 映射 Logic ---
    def _map(self, bits):
        if self.modulation == 'QPSK':
            # QPSK: 2 bits -> 1 symbol
            return np.array([1 + 1j if tuple(b) == (0, 0) else
                             -1 + 1j if tuple(b) == (0, 1) else
                             -1 - 1j if tuple(b) == (1, 0) else
                             1 - 1j for b in bits.reshape(-1, 2)]) / self.norm_factor
        elif self.modulation == '16QAM':
            # 16QAM: 4 bits -> 1 symbol (Simple Mapping)
            def bits2amp(b):
                if tuple(b) == (0, 0): return -3
                if tuple(b) == (0, 1): return -1
                if tuple(b) == (1, 0): return 1
                if tuple(b) == (1, 1): return 3

            symbols = []
            for b_chunk in bits.reshape(-1, 4):
                re = bits2amp(b_chunk[0:2])
                im = bits2amp(b_chunk[2:4])
                symbols.append(re + 1j * im)
            return np.array(symbols) / self.norm_factor

    # --- 解映射 Logic (最小距离法) ---
    def _demap(self, rx_symbols):
        # 还原幅度
        rx_scaled = rx_symbols * self.norm_factor

        # 定义标准星座点
        if self.modulation == 'QPSK':
            templates = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
            bit_map = [[0, 0], [0, 1], [1, 0], [1, 1]]
        elif self.modulation == '16QAM':
            templates = []
            bit_map = []
            levels = [-3, -1, 1, 3]
            bits_def = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for i in range(4):
                for j in range(4):
                    templates.append(levels[i] + 1j * levels[j])
                    bit_map.append(bits_def[i] + bits_def[j])
            templates = np.array(templates)

        # 暴力计算距离寻找最近点
        out_bits = []
        for r in rx_scaled:
            dists = np.abs(r - templates) ** 2
            idx = np.argmin(dists)
            out_bits.extend(bit_map[idx])

        return np.array(out_bits)

    def transmit(self, bit_stream):
        bits_per_ofdm = len(self.dataCarriers) * self.mu
        extra = len(bit_stream) % bits_per_ofdm
        if extra != 0: bit_stream = np.append(bit_stream, np.zeros(bits_per_ofdm - extra, dtype=int))

        symbols = self._map(bit_stream)
        n_blocks = len(symbols) // len(self.dataCarriers)

        sd = np.zeros((self.K, n_blocks), dtype=complex)
        sd[self.dataCarriers, :] = symbols.reshape(-1, n_blocks, order='F')
        sd[self.pilotCarriers, :] = 1 + 0j

        tx = np.fft.ifft(sd, axis=0)
        cp = tx[-self.CP:, :]
        return np.vstack([cp, tx]).flatten(order='F')

    def receive(self, rx_serial):
        # 默认只提供传统接收，AI 接收由子类实现
        sl = self.K + self.CP
        nb = len(rx_serial) // sl
        rx = rx_serial[:nb * sl].reshape(sl, nb, order='F')[self.CP:, :]
        rxf = np.fft.fft(rx, axis=0)

        rx_pilots = rxf[self.pilotCarriers, :]
        H_ls = rx_pilots / (1 + 0j)

        # 线性插值
        H_est = np.zeros((self.K, nb), dtype=complex)
        for i in range(nb):
            H_est[:, i] = interp1d(self.pilotCarriers, H_ls[:, i], kind='linear', fill_value="extrapolate")(
                self.allCarriers)

        eq = rxf / H_est
        data = eq[self.dataCarriers, :].flatten(order='F')
        return self._demap(data)


# ==========================================
# AI Extension (v2.0)
# ==========================================
class AI_OFDM_System(OFDM_System):
    def __init__(self, model_path, K=64, CP=16, P=8, modulation='QPSK'):
        super().__init__(K, CP, P, modulation)
        self.model = DNN_Channel_Estimator()
        self.device = torch.device("cpu")

        # 自动查找路径
        if not os.path.exists(model_path):
            alt_path = os.path.join("ai_training", model_path)
            if os.path.exists(alt_path): model_path = alt_path

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except:
            print(" Warning: AI Model not found or load failed.")

    def receive_with_ai(self, rx_serial):
        sl = self.K + self.CP
        nb = len(rx_serial) // sl
        rx = rx_serial[:nb * sl].reshape(sl, nb, order='F')[self.CP:, :]
        rxf = np.fft.fft(rx, axis=0)

        rx_pilots = rxf[self.pilotCarriers, :]
        H_ls = rx_pilots / (1 + 0j)

        # AI 推理
        feat = np.hstack([H_ls.T.real, H_ls.T.imag])
        inp = torch.FloatTensor(feat).to(self.device)
        with torch.no_grad():
            out = self.model(inp).numpy()
        H_est = (out[:, :64] + 1j * out[:, 64:]).T

        eq = rxf / H_est
        data = eq[self.dataCarriers, :].flatten(order='F')
        return self._demap(data)