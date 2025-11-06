import numpy as np
import matplotlib.pyplot as plt
from diff_method import find_preamble_offset
from diff_method import rrc_filter, find_optimal_sampling_phase
from gardner2 import gardner_timing_recovery

sps = 4

signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]
print(f"Длина исходного сигнала: {len(signal_iq)}")

f_rel = 0.016428
n = np.arange(signal_iq.size, dtype=np.float32)
signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)

preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

offset, signal_cutted, conv_results, conv_max = find_preamble_offset(signal_iq, preamble_iq, sps)

span = 10  
alpha = 0.35 

rrc = rrc_filter(sps, span, alpha)
signal_filtered = np.convolve(signal_cutted, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)

gardner_signal, errors, mu_history = gardner_timing_recovery(signal_filtered, sps, alpha=0.04)
print(f"Длина сигнала после Гарднера: {len(gardner_signal)}")
print(f"last mu: {mu_history[-1]}")

plt.figure(figsize=(10, 10))
plt.scatter(gardner_signal.real, gardner_signal.imag, alpha=0.5, s=10)
plt.grid(True)
plt.xlabel('I (In-phase)')
plt.ylabel('Q (Quadrature)')
plt.title(f'Созвездие ФМ4 (QPSK), {len(gardner_signal)} символов')
plt.axis('equal')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.show()