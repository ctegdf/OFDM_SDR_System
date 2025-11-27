import numpy as np

def run_channel(tx, snr):
    h = np.array([1.0, 0.2, 0.1])
    rx = np.convolve(tx, h, mode='full')[:len(tx)]
    npow = np.mean(np.abs(rx) ** 2) / (10 ** (snr / 10))
    return rx + (np.random.normal(0, np.sqrt(npow / 2), len(rx)) + 1j * np.random.normal(0, np.sqrt(npow / 2), len(rx)))
