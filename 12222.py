import numpy as np
import matplotlib.pyplot as plt
from diff_method import rrc_filter
from diff_method import find_preamble_offset
from gardner2 import gardner_timing_recovery

signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

rrc = rrc_filter(4, 10, 0.35)
signal_filtered = np.convolve(signal_iq, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)

offset, signal_iq, conv_results, conv_max = find_preamble_offset(signal_filtered, preamble_iq, 4)
print(conv_max)
# f_rel = -0.2335697446157038
f_rel = 0.016427416163163247
# f_rel = 0.016430255384296263


n = np.arange(len(signal_iq))
signal = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)

signal_for_f_rel = signal[::4]
plt.figure(figsize=(10, 10))
plt.plot(signal_for_f_rel.real, signal_for_f_rel.imag, 'o', markersize=3, alpha=0.6)
plt.title(f'Созвездие \n(f_rel = {f_rel:.6f})')
plt.xlabel('I (Real)')
plt.ylabel('Q (Imag)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

