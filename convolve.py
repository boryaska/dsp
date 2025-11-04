import numpy as np
import matplotlib.pyplot as plt

# Загрузка сигнала с sps=4
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
# Преобразуем в комплексные числа (I и Q)
signal_iq = signal[::2] + 1j * signal[1::2]

# Загрузка преамбулы с sps=1
preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
# Преобразуем в комплексные числа (I и Q)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

# Создаем сопряженную и развернутую преамбулу для свертки (matched filter)
preamble_matched = np.conj(preamble_iq[::-1])

# Создаем 4 графика для разных фаз дискретизации
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for phase in range(4):
    # Берем каждый 4-й отсчет, начиная с индекса phase
    decimated_signal = signal_iq[phase::4]
    
    # Вычисляем свертку
    correlation = np.convolve(decimated_signal, preamble_matched, mode='full')
    
    # Берем абсолютное значение (амплитуду)
    correlation_abs = np.abs(correlation)
    
    # Строим график
    axes[phase].plot(correlation_abs)
    axes[phase].set_title(f'Фаза {phase} (начальный отсчет: {phase})')
    axes[phase].set_xlabel('Отсчет')
    axes[phase].set_ylabel('Амплитуда корреляции')
    axes[phase].grid(True, alpha=0.3)
    
    # Отмечаем максимум
    max_idx = np.argmax(correlation_abs)
    max_val = correlation_abs[max_idx]
    axes[phase].plot(max_idx, max_val, 'ro', markersize=8, 
                     label=f'Макс: {max_val:.2f} на отсчете {max_idx}')
    axes[phase].legend()

plt.tight_layout()
plt.suptitle('Свертка преамбулы с сигналом (4 фазы дискретизации)', 
             fontsize=14, y=1.02)
plt.savefig('correlation_phases.png', dpi=150, bbox_inches='tight')
print("График сохранен в файл correlation_phases.png")

# Найдем наилучшую фазу
print("\nРезультаты для каждой фазы:")
print("-" * 60)
best_phase = 0
best_correlation = 0

for phase in range(4):
    decimated_signal = signal_iq[phase::4]
    correlation = np.convolve(decimated_signal, preamble_matched, mode='full')
    correlation_abs = np.abs(correlation)
    max_val = np.max(correlation_abs)
    max_idx = np.argmax(correlation_abs)
    
    print(f"Фаза {phase}: Макс. корреляция = {max_val:.2f}, "
          f"позиция = {max_idx}")
    
    if max_val > best_correlation:
        best_correlation = max_val
        best_phase = phase

print("-" * 60)
print(f"Наилучшая фаза: {best_phase} с корреляцией {best_correlation:.2f}")

