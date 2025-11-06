"""
Пример использования универсальных функций для построения графиков
"""
import numpy as np
from diff_method import find_preamble_offset
from plot_utils import plot_signal, plot_correlation_results

# Загружаем данные
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

sps = 4

print("=" * 70)
print("Демонстрация универсальных функций построения графиков")
print("=" * 70)

# Пример 1: График исходного сигнала (модуль)
print("\n1. Построение модуля исходного сигнала...")
plot_signal(signal_iq[:2000], 
           plot_type='abs',
           titles='Модуль исходного сигнала (первые 2000 отсчетов)',
           xlabel='Отсчет',
           ylabel='Модуль',
           save_path='signal_magnitude.png',
           show=False)

# Пример 2: I/Q компоненты
print("\n2. Построение I/Q компонент...")
plot_signal(signal_iq[:2000],
           plot_type='iq',
           titles='I/Q компоненты сигнала',
           xlabel='Отсчет',
           save_path='signal_iq.png',
           show=False)

# Пример 3: Несколько сигналов (модуль, фаза, I, Q)
print("\n3. Построение нескольких графиков одновременно...")
data_to_plot = [
    np.abs(signal_iq[:2000]),
    np.angle(signal_iq[:2000]),
    signal_iq[:2000].real,
    signal_iq[:2000].imag
]
plot_signal(data_to_plot,
           titles=['Модуль', 'Фаза', 'I (Real)', 'Q (Imag)'],
           ylabel=['Модуль', 'Фаза (рад)', 'I', 'Q'],
           xlabel='Отсчет',
           save_path='signal_components.png',
           show=False)

# Пример 4: Поиск преамбулы и построение графиков корреляции
print("\n4. Поиск преамбулы и построение результатов корреляции...")
offset, signal_cutted, conv_results, conv_max = find_preamble_offset(
    signal_iq, preamble_iq, sps
)

print(f"\nРезультаты поиска преамбулы:")
print(f"  Offset: {offset} символов ({offset * sps} отсчетов)")
print(f"  Максимумы по фазам: {conv_max}")

# Используем специализированную функцию для графиков корреляции
plot_correlation_results(
    conv_results, 
    conv_max, 
    offset, 
    sps,
    save_path='correlation_results.png',
    window=100,
    show=False
)

# Пример 5: Созвездие сигнала после выравнивания
print("\n5. Построение созвездия...")
# Берем символы из выровненного сигнала
symbols = signal_cutted[::sps][:1000]  # Первые 1000 символов
plot_signal(symbols,
           plot_type='constellation',
           titles=f'Созвездие сигнала (1000 символов)',
           alpha=0.5,
           s=15,
           save_path='constellation.png',
           show=False)

# Пример 6: График с маркерами
print("\n6. График с маркерами важных точек...")
markers = [
    {'x': offset, 'label': f'Преамбула (offset={offset})', 'color': 'red', 'linewidth': 2}
]
plot_signal(np.abs(signal_iq[:offset*sps+1000]),
           plot_type='line',
           titles='Сигнал с отметкой позиции преамбулы',
           xlabel='Отсчет',
           ylabel='Модуль',
           markers=markers,
           save_path='signal_with_markers.png',
           show=False)

print("\n" + "=" * 70)
print("Все графики успешно построены и сохранены!")
print("=" * 70)
print("\nСозданные файлы:")
print("  - signal_magnitude.png")
print("  - signal_iq.png")
print("  - signal_components.png")
print("  - correlation_results_phases.png")
print("  - correlation_results_comparison.png")
print("  - constellation.png")
print("  - signal_with_markers.png")


