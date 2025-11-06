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
signal_iq = signal_iq[3::4]
signal_iq = signal_iq[:len(preamble_iq)]

phase_signal = signal_iq[:len(preamble_iq)] * np.conj(preamble_iq)
phases = np.angle(phase_signal)
# Разность фаз между соседними отсчетами
phase_diffs = np.diff(np.unwrap(phases))  # unwrap убирает скачки ±2π

# Усредняем изменение фазы
avg_phase_diff = np.mean(phase_diffs)

# Переводим в частоту: Δφ = 2π·f_rel
f_rel_method1 = avg_phase_diff / (2 * np.pi*4)

print(f_rel_method1)

# 4. МЕТОД 2: Линейная регрессия (более точный)
n = np.arange(len(phases))
unwrapped_phases = np.unwrap(phases)

# Наклон прямой φ(n) = 2π·f_rel·n + φ₀
# Используем МНК: slope = Σ(n·φ) / Σ(n²)
slope = np.polyfit(n, unwrapped_phases, 1)[0]  # коэффициент при n
f_rel_method2 = slope / (2 * np.pi*4)
print(f_rel_method2)

# 5. МЕТОД 3: Через произведение соседних отсчетов (без unwrap)
# Более устойчив к циклическим скачкам фазы
prod = phase_signal[1:] * np.conj(phase_signal[:-1])
avg_rotation = np.angle(np.mean(prod))
f_rel_method3 = avg_rotation / (2 * np.pi*4)
print(f_rel_method3)



signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]
signal_filtered = np.convolve(signal_iq, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)
signal_filtered = signal_filtered[offset*4::]
print(f"Длина signal_filtered: {len(signal_filtered)}")


# for i in [f_rel_method1, f_rel_method2, f_rel_method3]:
#     signal = signal_filtered.copy()
#     n = np.arange(len(signal))
#     signal = signal * np.exp(-1j * 2 * np.pi * i * n)
#     signal = signal[0::4]
 
#     plt.figure(figsize=(10, 10))
#     plt.plot(signal.real[:-100], signal.imag[:-100], 'o', markersize=3, alpha=0.6)
#     plt.title(f'Созвездие после коррекции частоты\n(f_rel = {i:.6f})')
#     plt.xlabel('I (Real)')
#     plt.ylabel('Q (Imag)')
#     plt.grid(True, alpha=0.3)
#     plt.axis('equal')
#     plt.tight_layout()
#     plt.show()

n = np.arange(len(signal_filtered))
print(f"Длина n: {len(n)}")
signal_filtred =  signal_filtered * np.exp(-1j * 2 * np.pi * f_rel_method2 * n)
plt.figure(figsize=(10, 10))
plt.plot(signal_filtred.real[:-400:4], signal_filtred.imag[:-400:4], 'o', markersize=3, alpha=0.6)
plt.title(f'Созвездие после коррекции частоты\n(f_rel = {f_rel_method2:.6f})')
plt.xlabel('I (Real)')
plt.ylabel('Q (Imag)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()   


signal_recovered, errors, mu_history = gardner_timing_recovery(signal_filtred[:-400], 4, alpha=0.06)
print(f"Финальное значение mu: {mu_history[-1]:.4f}")


plt.figure(figsize=(10, 10))
plt.plot(signal_recovered.real, signal_recovered.imag, 'o', markersize=3, alpha=0.6)
plt.title(f'Созвездие после коррекции частоты\n(f_rel = {f_rel_method2:.6f})')
plt.xlabel('I (Real)')
plt.ylabel('Q (Imag)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

# График mu (временной сдвиг)
plt.figure(figsize=(12, 4))
plt.plot(mu_history)
plt.title('История значений mu (временной сдвиг интерполяции)')
plt.xlabel('Номер символа')
plt.ylabel('mu')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



