"""
Быстрый пример: добавьте эти строки в ваш код для построения спектра
"""
import numpy as np
from spectrum_utils import plot_spectrum, compare_spectra

# ============================================
# Загрузка сигнала (ваш существующий код)
# ============================================
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

# ============================================
# ПОСТРОЕНИЕ СПЕКТРА - добавьте эту строку!
# ============================================
plot_spectrum(signal_iq, 
              sample_rate=1.0,
              title='Спектр QPSK сигнала',
              window='hann',
              scale='db')

# Если нужно ограничить частотный диапазон:
plot_spectrum(signal_iq, 
              sample_rate=1.0,
              title='Спектр QPSK (увеличено)',
              window='hann',
              scale='db',
              xlim=(-0.3, 0.3))  # Показать только от -0.3 до 0.3





