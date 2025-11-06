# Руководство по использованию spectrum_utils.py

## Установленные функции

### 1. `compute_spectrum()` - Вычисление спектра
Базовая функция для вычисления FFT сигнала.

```python
from spectrum_utils import compute_spectrum

freq, spectrum = compute_spectrum(signal_iq, sample_rate=1.0, window='hann')
```

### 2. `plot_spectrum()` - Построение спектра
Главная функция для визуализации спектра.

```python
from spectrum_utils import plot_spectrum

# Базовое использование
plot_spectrum(signal_iq, sample_rate=1.0, title='Спектр сигнала')

# С дополнительными параметрами
plot_spectrum(signal_iq, 
              sample_rate=1.0,
              window='hann',      # Оконная функция: 'hann', 'hamming', 'blackman'
              scale='db',          # Масштаб: 'db', 'linear', 'power'
              xlim=(-0.5, 0.5),   # Ограничение по частоте
              save_path='spectrum.png')
```

### 3. `plot_psd()` - Спектральная плотность мощности
Использует метод Уэлча для более гладкой оценки спектра.

```python
from spectrum_utils import plot_psd

plot_psd(signal_iq,
         sample_rate=1.0,
         nfft=1024,          # Размер FFT для каждого сегмента
         window='hann')
```

### 4. `plot_spectrogram()` - Спектрограмма
Показывает изменение спектра во времени.

```python
from spectrum_utils import plot_spectrogram

plot_spectrogram(signal_iq,
                 sample_rate=1.0,
                 nfft=256,
                 cmap='viridis')  # Цветовая схема
```

### 5. `compare_spectra()` - Сравнение нескольких спектров
Построение нескольких спектров на одном графике.

```python
from spectrum_utils import compare_spectra

compare_spectra([signal1, signal2, signal3],
                labels=['Сигнал 1', 'Сигнал 2', 'Сигнал 3'],
                sample_rate=1.0,
                title='Сравнение сигналов')
```

## Практические примеры

### Пример 1: Быстрый анализ спектра
```python
import numpy as np
from spectrum_utils import plot_spectrum

# Загрузка сигнала
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

# Построение спектра
plot_spectrum(signal_iq, title='Мой спектр', window='hann')
```

### Пример 2: Анализ с фильтром
```python
from spectrum_utils import compare_spectra
from diff_method import rrc_filter

# Создаем фильтр
rrc = rrc_filter(sps=4, span=10, alpha=0.35)

# Фильтруем
signal_filtered = np.convolve(signal_iq, rrc, mode='same')

# Сравниваем
compare_spectra([signal_iq, signal_filtered],
                labels=['До фильтра', 'После фильтра'],
                xlim=(-0.5, 0.5))
```

### Пример 3: Детальный анализ
```python
from spectrum_utils import plot_spectrum, plot_psd, plot_spectrogram
import matplotlib.pyplot as plt

# Создаем общую фигуру
fig = plt.figure(figsize=(16, 12))

# Спектр
ax1 = plt.subplot(3, 1, 1)
plot_spectrum(signal_iq, ax=ax1, show=False, 
              title='Спектр', window='hann')

# PSD
ax2 = plt.subplot(3, 1, 2)
plot_psd(signal_iq, show=False, title='PSD')

# Спектрограмма
ax3 = plt.subplot(3, 1, 3)
plot_spectrogram(signal_iq, show=True, title='Спектрограмма')
```

## Параметры

### Оконные функции (`window`)
- `None` - без окна (прямоугольное окно)
- `'hann'` - окно Ханна (рекомендуется)
- `'hamming'` - окно Хэмминга
- `'blackman'` - окно Блэкмана

### Масштаб (`scale`)
- `'db'` - логарифмическая шкала в дБ (20*log10)
- `'linear'` - линейная шкала
- `'power'` - мощность (квадрат амплитуды)

### Частота дискретизации (`sample_rate`)
- `1.0` - нормализованная частота (по умолчанию)
- Любое другое значение - частота в Гц

## Советы

1. **Используйте оконные функции** для уменьшения утечки спектра:
   ```python
   plot_spectrum(signal, window='hann')
   ```

2. **Для длинных сигналов** используйте PSD вместо простого FFT:
   ```python
   plot_psd(signal, nfft=1024)  # Метод Уэлча
   ```

3. **Для анализа изменений во времени** используйте спектрограмму:
   ```python
   plot_spectrogram(signal, nfft=256)
   ```

4. **Ограничивайте диапазон частот** для детального просмотра:
   ```python
   plot_spectrum(signal, xlim=(-0.3, 0.3))
   ```

5. **Сохраняйте графики** для отчетов:
   ```python
   plot_spectrum(signal, save_path='my_spectrum.png', show=False)
   ```

## Интеграция с существующим кодом

Можно добавить в ваш `gardner_real_file.py`:

```python
from spectrum_utils import plot_spectrum, compare_spectra

# После загрузки сигнала
signal_iq = signal[::2] + 1j * signal[1::2]

# Построить спектр
plot_spectrum(signal_iq, title='Спектр исходного сигнала', window='hann')

# После фильтрации
signal_filtered = np.convolve(signal_cutted, rrc, mode='same')

# Сравнить
compare_spectra([signal_cutted, signal_filtered],
                labels=['До RRC', 'После RRC'],
                title='Влияние RRC фильтра')
```

## Примечания

- Все функции работают как с **комплексными**, так и с **вещественными** сигналами
- Для комплексных сигналов спектр автоматически центрируется (fftshift)
- Можно передавать существующую ось (`ax`) для построения нескольких графиков


