"""
Построение графиков дифференциальной корреляции для 4 фаз семплирования
"""
import numpy as np
import matplotlib.pyplot as plt
from diff_method import find_preamble_offset

# Загружаем сигнал
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]
print(f"Длина исходного сигнала: {len(signal_iq)} отсчетов")

# Загружаем преамбулу
preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]
print(f"Длина преамбулы: {len(preamble_iq)} символов")

# Параметры
sps = 4  # samples per symbol

# Вызываем функцию поиска преамбулы
offset, signal_cutted, conv_results, conv_max = find_preamble_offset(signal_iq, preamble_iq, sps)

print(f"\n{'='*70}")
print(f"РЕЗУЛЬТАТЫ ПОИСКА ПРЕАМБУЛЫ")
print(f"{'='*70}")
print(f"Найденный offset (медиана): {offset} символов ({offset * sps} отсчетов)")
print(f"\nМаксимумы корреляции для каждой фазы:")
for phase in range(sps):
    max_pos = conv_max[phase]
    max_val = np.abs(conv_results[phase][max_pos])
    print(f"  Фаза {phase}: позиция {max_pos:4d} символов, значение {max_val:8.2f}")
print(f"{'='*70}\n")

# Построение графиков для 4 фаз
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Дифференциальная корреляция для разных фаз семплирования', 
             fontsize=16, fontweight='bold')

colors = ['blue', 'green', 'red', 'purple']

for idx in range(sps):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Берем результаты корреляции для текущей фазы
    corr_abs = np.abs(conv_results[idx])
    x = np.arange(len(corr_abs))
    
    # Строим модуль корреляции
    ax.plot(x, corr_abs, linewidth=1.2, color=colors[idx], alpha=0.8, label='|Корреляция|')
    
    # Отмечаем максимум для этой фазы
    max_pos = conv_max[idx]
    max_val = corr_abs[max_pos]
    ax.axvline(max_pos, color='red', linestyle='--', linewidth=2.5, 
               label=f'Макс. фазы: {max_pos}')
    ax.scatter([max_pos], [max_val], color='red', s=150, zorder=5, marker='o', 
               edgecolors='darkred', linewidths=2)
    
    # Отмечаем общий offset (медиану)
    ax.axvline(offset, color='darkgreen', linestyle=':', linewidth=2.5, 
               label=f'Медиана: {offset}', alpha=0.8)
    
    # Настройка графика
    ax.set_title(f'Фаза {idx} (начало с отсчета {idx}, шаг {sps})',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Позиция (символы)', fontsize=12)
    ax.set_ylabel('|Нормализованная корреляция|', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=11, loc='upper right')
    
    # Добавляем текстовую аннотацию с максимальным значением
    ax.text(max_pos, max_val, f'  {max_val:.1f}', 
           fontsize=10, verticalalignment='bottom', 
           color='red', fontweight='bold')
    
    # Устанавливаем разумные пределы по оси X
    x_margin = 150
    x_min = max(0, offset - x_margin)
    x_max = min(len(corr_abs), offset + x_margin)
    ax.set_xlim(x_min, x_max)
    
    # Добавляем второстепенную сетку
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.minorticks_on()

plt.tight_layout()
plt.savefig('conv_4_phases.png', dpi=150, bbox_inches='tight')
print(f"График сохранен: conv_4_phases.png\n")

# Дополнительно: все фазы на одном графике для сравнения
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Верхний график: полный диапазон
for idx in range(sps):
    corr_abs = np.abs(conv_results[idx])
    x = np.arange(len(corr_abs))
    ax1.plot(x, corr_abs, linewidth=1.5, alpha=0.8, 
            color=colors[idx], label=f'Фаза {idx}')
    # Отмечаем максимум
    ax1.scatter([conv_max[idx]], [corr_abs[conv_max[idx]]], 
               s=120, zorder=5, color=colors[idx], edgecolors='black', linewidths=1.5)

ax1.axvline(offset, color='black', linestyle='--', linewidth=3, 
           label=f'Медиана offset={offset}', alpha=0.7)
ax1.set_xlabel('Позиция (символы)', fontsize=12)
ax1.set_ylabel('|Нормализованная корреляция|', fontsize=12)
ax1.set_title('Сравнение всех 4 фаз (полный диапазон)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.legend(fontsize=11, loc='upper right')

# Нижний график: увеличенная область вокруг максимума
x_margin = 150
x_min = max(0, offset - x_margin)
x_max = min(len(conv_results[0]), offset + x_margin)

for idx in range(sps):
    corr_abs = np.abs(conv_results[idx])
    x = np.arange(x_min, x_max)
    ax2.plot(x, corr_abs[x_min:x_max], linewidth=2, alpha=0.8, 
            color=colors[idx], label=f'Фаза {idx}')
    # Отмечаем максимум, если он в диапазоне
    if x_min <= conv_max[idx] < x_max:
        ax2.scatter([conv_max[idx]], [corr_abs[conv_max[idx]]], 
                   s=120, zorder=5, color=colors[idx], edgecolors='black', linewidths=1.5)
        # Добавляем аннотацию
        ax2.text(conv_max[idx], corr_abs[conv_max[idx]], f'  {corr_abs[conv_max[idx]]:.1f}',
                fontsize=9, verticalalignment='bottom', color=colors[idx], fontweight='bold')

ax2.axvline(offset, color='black', linestyle='--', linewidth=3, 
           label=f'Медиана offset={offset}', alpha=0.7)
ax2.set_xlabel('Позиция (символы)', fontsize=12)
ax2.set_ylabel('|Нормализованная корреляция|', fontsize=12)
ax2.set_title(f'Увеличенная область (±{x_margin} символов от медианы)', 
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.legend(fontsize=11, loc='upper right')

# Добавляем второстепенную сетку
ax1.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
ax1.minorticks_on()
ax2.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
ax2.minorticks_on()

plt.tight_layout()
plt.savefig('conv_4_phases_comparison.png', dpi=150, bbox_inches='tight')
print(f"График сохранен: conv_4_phases_comparison.png\n")

# Показываем графики
plt.show()

print(f"\n{'='*70}")
print(f"Графики успешно построены!")
print(f"{'='*70}")

