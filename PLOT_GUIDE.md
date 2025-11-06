# Руководство по использованию универсальных функций построения графиков

## Обзор

Модуль `plot_utils.py` содержит универсальные функции для построения различных типов графиков при анализе сигналов.

## Основные функции

### 1. `plot_signal()` - Универсальная функция построения графиков

Поддерживает различные типы графиков и данных.

#### Базовое использование

```python
from plot_utils import plot_signal
import numpy as np

# Простой график одного сигнала
plot_signal(signal, titles='Мой сигнал')
```

#### Типы графиков (`plot_type`)

- **`'line'`** (по умолчанию) - Линейный график
- **`'scatter'`** - Точечный график
- **`'stem'`** - Стебельковая диаграмма
- **`'abs'`** - Модуль комплексного сигнала
- **`'angle'`** - Фаза комплексного сигнала
- **`'iq'`** - Разделение на I и Q компоненты (2 графика)
- **`'constellation'`** - Созвездие (I vs Q на одном графике)

#### Примеры использования

##### Модуль комплексного сигнала
```python
plot_signal(complex_signal, 
           plot_type='abs',
           titles='Модуль сигнала',
           xlabel='Отсчет',
           ylabel='Модуль')
```

##### I/Q компоненты
```python
plot_signal(complex_signal, 
           plot_type='iq',
           titles='Комплексный сигнал')
```

##### Несколько графиков одновременно
```python
plot_signal([signal1, signal2, signal3],
           titles=['График 1', 'График 2', 'График 3'],
           ylabel=['Амплитуда', 'Частота', 'Фаза'])
```

##### Созвездие
```python
plot_signal(symbols, 
           plot_type='constellation',
           titles='QPSK созвездие',
           alpha=0.6,  # прозрачность точек
           s=20)       # размер точек
```

##### График с маркерами
```python
markers = [
    {'x': 100, 'label': 'Начало преамбулы', 'color': 'red'},
    {'x': 250, 'label': 'Конец преамбулы', 'color': 'green'}
]
plot_signal(signal, 
           markers=markers,
           titles='Сигнал с отметками')
```

##### Сохранение графика
```python
plot_signal(signal,
           save_path='my_plot.png',
           show=False)  # не показывать, только сохранить
```

### 2. `plot_correlation_results()` - Специализированная функция для корреляции

Создает подробные графики результатов дифференциальной корреляции из функции `find_preamble_offset`.

#### Использование

```python
from diff_method import find_preamble_offset
from plot_utils import plot_correlation_results

# Поиск преамбулы
offset, signal_cutted, conv_results, conv_max = find_preamble_offset(
    signal_iq, preamble_iq, sps
)

# Построение графиков
plot_correlation_results(
    conv_results,      # результаты корреляции для каждой фазы
    conv_max,          # позиции максимумов
    offset,            # найденное смещение
    sps,               # samples per symbol
    save_path='correlation.png',
    window=100         # окно вокруг максимума
)
```

Функция создает **два набора графиков**:
1. Отдельные графики для каждой фазы семплирования (4 графика)
2. Сравнительные графики всех фаз (2 графика: полный диапазон и увеличенная область)

## Параметры функции `plot_signal()`

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | array или list | Данные для отображения (одиночный массив или список массивов) |
| `titles` | str или list | Заголовок(и) графика(ов) |
| `xlabel` | str | Подпись оси X |
| `ylabel` | str или list | Подпись(и) оси Y |
| `plot_type` | str | Тип графика ('line', 'scatter', 'abs', 'iq', 'constellation' и т.д.) |
| `figsize` | tuple | Размер фигуры (ширина, высота), по умолчанию (14, 6) |
| `save_path` | str | Путь для сохранения графика |
| `show` | bool | Показывать ли график (True/False) |
| `grid` | bool | Отображать ли сетку (True/False) |
| `markers` | list | Список маркеров для отметки точек |
| `layout` | tuple | Расположение подграфиков (rows, cols) |
| `**kwargs` | dict | Дополнительные параметры matplotlib |

## Полный пример использования

```python
import numpy as np
from diff_method import find_preamble_offset
from plot_utils import plot_signal, plot_correlation_results

# Загружаем данные
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

sps = 4

# 1. График модуля сигнала
plot_signal(signal_iq[:2000], 
           plot_type='abs',
           titles='Модуль сигнала',
           save_path='magnitude.png')

# 2. I/Q компоненты
plot_signal(signal_iq[:2000],
           plot_type='iq',
           save_path='iq_components.png')

# 3. Поиск преамбулы
offset, signal_cutted, conv_results, conv_max = find_preamble_offset(
    signal_iq, preamble_iq, sps
)

# 4. Графики корреляции
plot_correlation_results(conv_results, conv_max, offset, sps,
                        save_path='correlation.png')

# 5. Созвездие
symbols = signal_cutted[::sps][:1000]
plot_signal(symbols, 
           plot_type='constellation',
           titles='Созвездие QPSK',
           save_path='constellation.png')
```

## Запуск примера

```bash
cd /home/user/Documents/Dev/dsp
source venv/bin/activate
python example_plot_usage.py
```

Это создаст набор различных графиков, демонстрирующих возможности функций.

## Советы

1. **Для больших данных** используйте срезы: `signal[:1000]` вместо полного сигнала
2. **Для сохранения без показа** используйте `show=False`
3. **Для настройки размера** меняйте `figsize=(ширина, высота)`
4. **Для прозрачности** используйте параметр `alpha` (0.0 - 1.0)
5. **Для изменения цвета** используйте параметр `color` или `c`

## Дополнительные параметры matplotlib

Все дополнительные параметры передаются напрямую в matplotlib:

```python
plot_signal(signal,
           linewidth=2,      # толщина линии
           alpha=0.7,        # прозрачность
           color='red',      # цвет
           linestyle='--')   # стиль линии
```


