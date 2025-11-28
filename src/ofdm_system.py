import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import os

class OFDM_System:
    def __init__(self, K=64, CP=16, P=8):
        self.K, self.CP, self.P = K, CP, P
        self.allCarriers = np.arange(K)
        self.pilotCarriers = self.allCarriers[::K // P]
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)
        self.mu = 2
        self.mapping_table = {(0, 0): 1 + 1j, (0, 1): -1 + 1j, (1, 0): -1 - 1j, (1, 1): 1 - 1j}

    def _qpsk_map(self, bits):
        return np.array([self.mapping_table[tuple(b)] for b in bits.reshape(-1, 2)])

    def _qpsk_demap(self, symbols):
        db = []
        for s in symbols:
            if s.real > 0 and s.imag > 0:
                db.extend([0, 0])
            elif s.real < 0 and s.imag > 0:
                db.extend([0, 1])
            elif s.real < 0 and s.imag < 0:
                db.extend([1, 0])
            else:
                db.extend([1, 1])
        return np.array(db)

    def transmit(self, bit_stream):
        bits_per_ofdm = len(self.dataCarriers) * self.mu
        if len(bit_stream) % bits_per_ofdm != 0: bit_stream = np.append(bit_stream, np.zeros(
            bits_per_ofdm - len(bit_stream) % bits_per_ofdm, dtype=int))
        qpsk = self._qpsk_map(bit_stream)
        nb = len(qpsk) // len(self.dataCarriers)
        sd = np.zeros((self.K, nb), dtype=complex)
        sd[self.dataCarriers, :] = qpsk.reshape(-1, nb, order='F')
        sd[self.pilotCarriers, :] = 1 + 0j
        tx = np.fft.ifft(sd, axis=0)
        return np.vstack([tx[-self.CP:, :], tx]).flatten(order='F')

    def receive(self, rx_serial):
        sl = self.K + self.CP
        nb = len(rx_serial) // sl
        rx = rx_serial[:nb * sl].reshape(sl, nb, order='F')[self.CP:, :]
        rxf = np.fft.fft(rx, axis=0)
        H = rxf[self.pilotCarriers, :] / (1 + 0j)
        # 简单插值
        from scipy.interpolate import interp1d
        He = np.zeros((self.K, nb), dtype=complex)
        for i in range(nb):
            He[:, i] = interp1d(self.pilotCarriers, H[:, i], kind='linear', fill_value="extrapolate")(self.allCarriers)
        return self._qpsk_demap((rxf / He)[self.dataCarriers, :].flatten(order='F'))

class DNN_Channel_Estimator(nn.Module):
    def __init__(self):
        super(DNN_Channel_Estimator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x): return self.layers(x)


class AI_OFDM_System(OFDM_System):
    def __init__(self, model_path, K=64, CP=16, P=8):
        super().__init__(K, CP, P)
        self.model = DNN_Channel_Estimator()
        self.device = torch.device("cpu")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except:
            print("找不到模型文件！")

    # 重写接收方法
    def receive_with_ai(self, rx_serial):
        symbol_len = self.K + self.CP
        n_blocks = len(rx_serial) // symbol_len
        rx_serial = rx_serial[:n_blocks * symbol_len]
        rx_matrix = rx_serial.reshape(symbol_len, n_blocks, order='F')
        rx_data_time = rx_matrix[self.CP:, :]
        rx_freq_data = np.fft.fft(rx_data_time, axis=0)

        # 1. 提取导频
        rx_pilots = rx_freq_data[self.pilotCarriers, :]
        H_ls_pilots = rx_pilots / (1.0 + 0j)  # LS 估计 (8, n_blocks)

        # 2. 准备 AI 输入 (Batch 处理)
        H_ls_transposed = H_ls_pilots.T
        features = np.hstack([H_ls_transposed.real, H_ls_transposed.imag])
        input_tensor = torch.FloatTensor(features).to(self.device)

        # 3. AI 推理
        with torch.no_grad():
            output_tensor = self.model(input_tensor)  # -> (n_blocks, 128)

        # 4. 还原为复数 H_est
        output_np = output_tensor.cpu().numpy()
        H_est_real = output_np[:, :64]
        H_est_imag = output_np[:, 64:]
        # 转置回 (64, n_blocks) 以匹配后续计算
        H_est_matrix = (H_est_real + 1j * H_est_imag).T

        rx_eq_data = rx_freq_data / H_est_matrix
        qpsk_symbols = rx_eq_data[self.dataCarriers, :]
        return self._qpsk_demap(qpsk_symbols.flatten(order='F'))