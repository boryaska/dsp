"""
Примеры использования функций из spectrum_utils.py
"""
import numpy as np
from spectrum_utils import plot_spectrum, plot_psd, plot_spectrogram, compare_spectra

# Загрузка вашего сигнала
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

print(f"Длина сигнала: {len(signal_iq)}")

# ============================================
# Пример 1: Простой спектр
# ============================================
print("\n1. Построение спектра сигнала")
plot_spectrum(signal_iq, 
              sample_rate=1.0,  # Нормализованная частота
              title='Спектр QPSK сигнала (полный)',
              window='hann',
              scale='db',
              show=False,
              save_path='spectrum_full.png')

# ============================================
# Пример 2: Спектр с ограничением по частоте
# ============================================
print("\n2. Спектр с ограничением частоты")
plot_spectrum(signal_iq, 
              sample_rate=1.0,
              title='Спектр QPSK (увеличенная область)',
              window='hann',
              scale='db',
              xlim=(-0.3, 0.3),  # Ограничиваем диапазон частот
              show=False)

# ============================================
# Пример 3: Спектральная плотность мощности (PSD)
# ============================================
print("\n3. PSD методом Уэлча")
plot_psd(signal_iq,
         sample_rate=1.0,
         title='Спектральная плотность мощности',
         nfft=1024,
         window='hann',
         show=False)

# ============================================
# Пример 4: Спектрограмма (изменение спектра во времени)
# ============================================
print("\n4. Спектрограмма")
plot_spectrogram(signal_iq[:10000],  # Берем часть сигнала
                 sample_rate=1.0,
                 title='Спектрограмма QPSK',
                 nfft=256,
                 cmap='viridis',
                 show=False)

# ============================================
# Пример 5: Сравнение спектров до и после фильтрации
# ============================================
print("\n5. Сравнение спектров")

# Создаем простой фильтр для примера
from diff_method import rrc_filter

sps = 4
span = 10
alpha = 0.35
rrc = rrc_filter(sps, span, alpha)

# Фильтруем сигнал
signal_filtered = np.convolve(signal_iq, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)

# Сравниваем
compare_spectra([signal_iq, signal_filtered],
                labels=['Исходный сигнал', 'После RRC фильтра'],
                sample_rate=1.0,
                title='Сравнение спектров до и после фильтрации',
                window='hann',
                xlim=(-0.5, 0.5),
                show=True,
                save_path='spectrum_comparison.png')

print("\n✓ Все графики построены!")


