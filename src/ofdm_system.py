import numpy as np
from scipy.interpolate import interp1d

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
