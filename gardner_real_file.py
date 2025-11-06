import numpy as np
import matplotlib.pyplot as plt
from diff_method import find_preamble_offset
from diff_method import rrc_filter, find_optimal_sampling_phase
from gardner2 import gardner_timing_recovery




sps = 4

signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]


print(f"Длина исходного сигнала: {len(signal_iq)}")

# f_rel = 0.0164284
# n = np.arange(signal_iq.size, dtype=np.float32)
# signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)

# delay = -0.05829069139461425
# t = np.arange(len(signal_iq))
# signal_shifited = np.interp(t, t - delay, signal_iq)
# signal_iq = signal_shifited

preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

offset, signal_cutted, conv_results, conv_max = find_preamble_offset(signal_iq, preamble_iq, sps)

print(f"Преамбула найдена на позиции: {offset} символов")
print(f"Длина выровненного сигнала: {len(signal_cutted)}")


span = 10  
alpha = 0.35 

rrc = rrc_filter(sps, span, alpha)
signal_filtered = np.convolve(signal_cutted, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)
# signal_filtered = signal_cutted


best_phase, phase_correlations = find_optimal_sampling_phase(signal_filtered, preamble_iq, sps)
print(f"Оптимальная фаза семплирования: {best_phase}")
print(f"Значения корреляции по фазам: {phase_correlations}") 

offset, signal_cutted, conv_results, conv_max = find_preamble_offset(signal_filtered, preamble_iq, sps)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Результаты корреляции для каждой фазы (sps=4)', fontsize=14)

for idx in range(sps):
    row = idx // 2
    col = idx % 2
    ax = axes2[row, col]
            
    # Строим абсолютное значение корреляции
    ax.plot(np.abs(conv_results[idx]))
    ax.set_title(f'Фаза {idx}')
    ax.set_xlabel('Индекс')
    ax.set_ylabel('|Корреляция|')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

signal_filtered = signal_filtered[best_phase::sps]
signal_filtered = signal_filtered[:len(preamble_iq)]

signal_for_plot_1 = signal_filtered.copy()
print(f"Длина signal_for_plot_1: {len(signal_for_plot_1)}")

phases_symbols = signal_filtered * np.conj(preamble_iq)




phases = np.angle(phases_symbols)
phases_unwrapped = np.unwrap(phases)

n_symbols = len(phases_unwrapped)
n = np.arange(n_symbols)


phase_diff = phases_unwrapped[-1] - phases_unwrapped[0]
phase_per_symbol = phase_diff / (n_symbols - 1)


# coeffs = np.polyfit(n, phases_unwrapped, 1)
# phase_per_symbol = coeffs[0]
print(f"phase_per_symbol: {phase_per_symbol}")
f_rel = phase_per_symbol / (2 * np.pi * sps)
print(f"f_rel: {f_rel}")

signal_for_plot_2 = signal_for_plot_1 * np.exp(-1j * 2 * np.pi * (-f_rel) * n)

# Построение графиков на комплексной плоскости
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# График для signal_for_plot_1
ax1.scatter(phases_symbols.real[:60]   , phases_symbols.imag[:60], alpha=0.6, s=20, c='blue')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('I (In-phase)', fontsize=12)
ax1.set_ylabel('Q (Quadrature)', fontsize=12)
ax1.set_title(f'Созвездие до коррекции частоты\n({len(phases_symbols)} символов)', fontsize=14)
ax1.axis('equal')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)


plt.tight_layout()
plt.show()

# # График корреляций по фазам
# plt.figure(figsize=(10, 6))
# plt.bar(range(sps), phase_correlations)
# plt.xlabel('Фаза семплирования')
# plt.ylabel('Корреляция')
# plt.title(f'Корреляция с преамбулой для разных фаз семплирования\nОптимальная фаза: {best_phase}')
# plt.grid(True, alpha=0.3)
# plt.xticks(range(sps))
# plt.show()

# signal_gardner, errors, mu_history = gardner_timing_recovery(signal_corrected, sps, alpha=0.04)
# print(f"Длина сигнала после Гарднера: {len(signal_gardner)}")
# print(f"last mu: {mu_history[-1]}")


# symbols = signal_gardner

# # График ошибок Гарднера
# plt.figure(figsize=(12, 5))
# plt.plot(errors)
# plt.grid(True)
# plt.xlabel('Номер символа')
# plt.ylabel('Ошибка')
# plt.title('Ошибка синхронизации Гарднера (Timing Error)')
# plt.show()

# # График истории mu
# plt.figure(figsize=(12, 5))
# plt.plot(mu_history)
# plt.grid(True)
# plt.xlabel('Номер символа')
# plt.ylabel('μ (дробная часть)')
# plt.title('История изменения μ (дробная часть временного смещения)')
# plt.ylim([0, 1])
# plt.show()

# # Построение диаграммы созвездия
# plt.figure(figsize=(10, 10))
# plt.scatter(symbols.real, symbols.imag, alpha=0.5, s=10)
# plt.grid(True)
# plt.xlabel('I (In-phase)')
# plt.ylabel('Q (Quadrature)')
# plt.title(f'Созвездие ФМ4 (QPSK), {len(symbols)} символов')
# plt.axis('equal')
# plt.axhline(y=0, color='k', linewidth=0.5)
# plt.axvline(x=0, color='k', linewidth=0.5)
# plt.show()
