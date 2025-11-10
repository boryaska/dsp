import numpy as np
from diff_method import rrc_filter
import matplotlib.pyplot as plt

signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

rrc = rrc_filter(4, 10, 0.35)
signal_filtered = np.convolve(signal_iq, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)
signal_filtered = signal_filtered[800:-400]

for i in range(4):
    signal_filtered_phase = signal_filtered[i::4]
    plt.figure(figsize=(10, 10))
    plt.plot(signal_filtered_phase.real, signal_filtered_phase.imag, 'o', markersize=3, alpha=0.6)
    plt.title(f'Созвездие \n(f_rel = )')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imag)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()