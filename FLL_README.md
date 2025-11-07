# 🔄 Frequency Locked Loop (FLL) — Реализация

## 📁 Файлы проекта

### Основные модули

| Файл | Описание |
|------|----------|
| **`fll.py`** | 🔧 Основная реализация FLL<br>Содержит классы: `FrequencyErrorDetector`, `LoopFilter`, `NCO`, `FrequencyLockedLoop` |
| **`fll_real_signal.py`** | 📊 Применение FLL к реальным данным<br>Работает с файлами `qpsk_high_snr_sps_4_float32.pcm` |
| **`test_fll.py`** | ✅ Юнит-тесты для проверки работоспособности<br>Тестирует все блоки FLL |

### Документация

| Файл | Назначение |
|------|-----------|
| **`FLL_GUIDE.md`** | 📖 Полное руководство с теорией и примерами |
| **`FLL_CHEATSHEET.md`** | 🎯 Краткая шпаргалка для быстрого старта |
| **`FLL_README.md`** | 📌 Этот файл — обзор проекта |

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install numpy matplotlib
```

### 2. Базовое использование

```python
from fll import FrequencyLockedLoop
import numpy as np

# Загрузка сигнала
signal = np.fromfile('signal.pcm', dtype=np.complex64)

# Создание FLL
fll = FrequencyLockedLoop(
    detector_method='cross_product',
    Kp=0.001,
    Ki=0.00005
)

# Обработка
corrected_signal, freq_estimate = fll.process_signal(signal)

print(f"Частотный сдвиг: {freq_estimate:.8f}")
```

### 3. Запуск примеров

```bash
# Тестовый сигнал
python fll.py

# Реальные данные
python fll_real_signal.py

# Юнит-тесты
python test_fll.py
```

---

## 🏗️ Архитектура FLL

### Структура блоков

```
┌────────────────────────────────────────────────────────┐
│                 FrequencyLockedLoop                    │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │                                                  │ │
│  │              Feedback Loop                       │ │
│  │                                                  │ │
│  │  Input ──> ⊗ ──> [Detector] ──> [Filter] ──> [NCO] │
│  │            │                                      │ │
│  │            └──────────────────────────────────────┘ │
│  │                   exp(-j·φ)                         │
│  └──────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────┘
```

### Основные классы

#### 1. `FrequencyErrorDetector`

**Методы детектирования:**
- `cross_product` — быстрый, для малых ошибок
- `atan2` — для больших ошибок
- `decision_directed` — для высокого SNR

**Пример:**
```python
detector = FrequencyErrorDetector(method='cross_product')
error = detector.detect(sample)
```

#### 2. `LoopFilter`

**PI-контроллер:**
- `Kp` — пропорциональный коэффициент
- `Ki` — интегральный коэффициент

**Пример:**
```python
loop_filter = LoopFilter(Kp=0.001, Ki=0.00005)
correction = loop_filter.update(error)
```

#### 3. `NCO`

**Numerically Controlled Oscillator:**
- Генерирует `exp(-j·φ)` для коррекции
- Накапливает фазу и частоту

**Пример:**
```python
nco = NCO()
nco.update_freq(freq_correction)
correction_signal = nco.step()
```

#### 4. `FrequencyLockedLoop`

**Интегрирует все блоки:**

**Пример:**
```python
fll = FrequencyLockedLoop(
    detector_method='cross_product',
    Kp=0.001,
    Ki=0.00005,
    freq_limit=0.1
)

# Обработка сигнала
corrected, freq = fll.process_signal(signal)

# Визуализация
fll.plot_performance()
```

---

## 📊 Результаты тестов

### Тест 1: NCO
✅ **Работает корректно**
- Генерирует комплексный экспоненциал с заданной частотой
- Точность: < 0.001 рад

### Тест 2: Frequency Error Detector
✅ **Все методы работают**
- `cross_product`: точность для малых сдвигов
- `atan2`: работает для больших сдвигов
- `decision_directed`: лучшая точность при высоком SNR

### Тест 3: Loop Filter
✅ **PI-контроллер работает**
- Интегратор корректно накапливает ошибку
- Ограничение (anti-windup) работает

### Тест 4: Сходимость FLL
⚠️ **Зависит от параметров**
- Требуется настройка `Kp`, `Ki` под конкретный сигнал
- `decision_directed` показывает лучшие результаты для QPSK

### Тест 5: Сравнение методов
🏆 **Лучший метод: decision_directed** (для QPSK с высоким SNR)
- Ошибка: 0.0002 (0.02%)
- Быстрая сходимость

---

## ⚙️ Настройка параметров

### Выбор метода детектирования

| Условие | Рекомендация |
|---------|-------------|
| SNR > 15 дБ, QPSK | `decision_directed` ✅ |
| SNR < 10 дБ | `cross_product` |
| Большая ошибка (>5%) | `atan2` |
| Неизвестная модуляция | `cross_product` или `atan2` |

### Выбор коэффициентов

**Формула расчёта:**

```python
BnT = 0.01  # полоса петли (1% от символьной частоты)
zeta = 0.707  # критическое демпфирование

theta = BnT / (zeta + 1/(4*zeta))
Kp = theta
Ki = theta**2 / 4
```

**Типичные значения:**

| Сценарий | Kp | Ki | Комментарий |
|----------|----|----|-------------|
| Высокий SNR | 0.005 | 0.0001 | Быстрая сходимость |
| Средний SNR | 0.001 | 0.00005 | Баланс |
| Низкий SNR | 0.0002 | 0.00001 | Стабильность |
| Большой сдвиг | 0.002 | 0.0001 | Увеличенная полоса |

---

## 🔍 Примеры использования

### Пример 1: Базовый QPSK

```python
from fll import FrequencyLockedLoop
import numpy as np

# Генерация QPSK с частотным сдвигом
symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 1000)
signal = np.repeat(symbols, 4)  # sps=4

f_offset = 0.01
n = np.arange(len(signal))
signal_shifted = signal * np.exp(1j * 2*np.pi * f_offset * n)

# FLL
fll = FrequencyLockedLoop(
    detector_method='decision_directed',
    Kp=0.005,
    Ki=0.0001
)

corrected, freq = fll.process_signal(signal_shifted)
print(f"Истинный сдвиг: {f_offset}")
print(f"Оценка FLL: {freq}")
```

### Пример 2: Реальный сигнал с преамбулой

```python
from fll import FrequencyLockedLoop
from diff_method import rrc_filter, find_preamble_offset
import numpy as np

# Загрузка
signal = np.fromfile('qpsk_signal.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

# RRC фильтрация
rrc = rrc_filter(sps=4, span=10, alpha=0.35)
signal_filtered = np.convolve(signal_iq, rrc, mode='same')

# Поиск преамбулы
preamble = np.fromfile('preamble.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

offset, signal_aligned, _, _, phase = find_preamble_offset(
    signal_filtered, preamble_iq, sps=4
)

# FLL
fll = FrequencyLockedLoop(
    detector_method='cross_product',
    Kp=0.001,
    Ki=0.00005
)

corrected, freq = fll.process_signal(signal_aligned)
print(f"Частотный сдвиг: {freq:.8f}")

# Децимация и визуализация
symbols = corrected[::4]
import matplotlib.pyplot as plt
plt.plot(symbols.real, symbols.imag, 'o')
plt.title(f'Созвездие после FLL (f={freq:.6f})')
plt.show()
```

### Пример 3: Адаптивная обработка в реальном времени

```python
fll = FrequencyLockedLoop(
    detector_method='cross_product',
    Kp=0.001,
    Ki=0.00005
)

# Обработка отсчёт за отсчётом
output_buffer = []

for i, sample in enumerate(signal):
    corrected_sample = fll.process_sample(sample)
    output_buffer.append(corrected_sample)
    
    # Мониторинг каждые 100 отсчётов
    if i % 100 == 0:
        current_freq = fll.get_frequency_estimate()
        print(f"Отсчёт {i}: f = {current_freq:.6f}")
```

### Пример 4: Комбинация методов

```python
# Шаг 1: Грубая оценка по преамбуле
signal_preamble = signal_aligned[:len(preamble_iq)]
phase_signal = signal_preamble * np.conj(preamble_iq)
phases = np.angle(phase_signal)
slope = np.polyfit(np.arange(len(phases)), np.unwrap(phases), 1)[0]
f_coarse = slope / (2 * np.pi * 4)

# Шаг 2: Применяем грубую коррекцию
n = np.arange(len(signal_aligned))
signal_coarse_corrected = signal_aligned * np.exp(-1j * 2*np.pi * f_coarse * n)

# Шаг 3: FLL для точной коррекции
fll = FrequencyLockedLoop(Kp=0.0005, Ki=0.00001)
signal_fine_corrected, f_fine = fll.process_signal(signal_coarse_corrected)

# Итоговая частота
f_total = f_coarse + f_fine
print(f"Грубая оценка: {f_coarse:.8f}")
print(f"Точная оценка: {f_fine:.8f}")
print(f"Итого: {f_total:.8f}")
```

---

## 📈 Визуализация

### Созвездие до и после

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# До FLL
ax1.plot(signal[::4].real, signal[::4].imag, 'o', alpha=0.6)
ax1.set_title('До FLL')
ax1.grid(True)

# После FLL
ax2.plot(corrected[::4].real, corrected[::4].imag, 'o', alpha=0.6)
ax2.set_title(f'После FLL (f={freq:.6f})')
ax2.grid(True)

plt.show()
```

### Анализ сходимости

```python
fll.plot_performance()  # встроенная функция
```

Отображает:
1. Ошибку детектора во времени
2. Оценку частоты NCO
3. Фазу NCO

---

## ❓ FAQ

### В: FLL не сходится. Что делать?

**О:** Попробуйте:
1. Уменьшить `Kp` и `Ki` в 10 раз
2. Изменить метод: `cross_product` → `atan2`
3. Увеличить `freq_limit`
4. Проверить SNR сигнала

### В: Ошибка оценки слишком большая

**О:** 
1. Для высокого SNR используйте `decision_directed`
2. Увеличьте длину сигнала для лучшей сходимости
3. Используйте грубую оценку (преамбула/FFT) для инициализации

### В: FLL медленно сходится

**О:**
1. Увеличьте `Kp` (но не более 0.01)
2. Увеличьте `Ki`
3. Используйте двухступенчатую коррекцию (грубая + точная)

### В: В чём разница между FLL и PLL?

**О:**
- **FLL** отслеживает частоту (быстрый захват больших ошибок)
- **PLL** отслеживает фазу (точная коррекция малых ошибок)
- Обычно используют **FLL → PLL** (каскад)

---

## 📚 Литература

1. **Michael Rice**, "Digital Communications: A Discrete-Time Approach" (Chapter 7)
2. **Fredric Harris**, "Multirate Signal Processing for Communication Systems"
3. **Floyd Gardner**, "Phaselock Techniques" (3rd edition)
4. **Umberto Mengali**, "Synchronization Techniques for Digital Receivers"

---

## 🎓 Теоретическая справка

### Формулы

**Cross-Product Detector:**
```
e[n] = Im(s[n] · s*[n-1])
```

**Atan2 Detector:**
```
e[n] = atan2(Im(s[n]·s*[n-1]), Re(s[n]·s*[n-1]))
```

**PI Loop Filter:**
```
integrator[n] = integrator[n-1] + Ki·e[n]
u[n] = Kp·e[n] + integrator[n]
```

**NCO:**
```
φ[n+1] = φ[n] + 2π·f[n]
correction[n] = exp(-j·φ[n])
```

### Полоса петли

```
Bn ≈ (Kp + Ki/4) / (2π)
```

Типично: `Bn = 0.01...0.05` от символьной частоты

### Время установления

```
T_settling ≈ 4 / (2π·Bn)
```

---

## 🛠️ Дополнительные инструменты

### Утилиты в проекте

| Файл | Функция |
|------|---------|
| `f_rel.py` | Альтернативные методы оценки частоты (преамбула, регрессия) |
| `diff_method.py` | RRC фильтр, поиск преамбулы, дифференциальная корреляция |
| `gardner2.py` | Синхронизация тактовой частоты (Gardner TED) |

### Интеграция с другими блоками

```python
# Полный receiver pipeline
from diff_method import rrc_filter, find_preamble_offset
from fll import FrequencyLockedLoop
from gardner2 import gardner_timing_recovery

# 1. Фильтрация
signal_filtered = np.convolve(signal, rrc, mode='same')

# 2. Поиск преамбулы и выравнивание фазы
offset, signal_aligned, _, _, _ = find_preamble_offset(
    signal_filtered, preamble, sps=4
)

# 3. Коррекция частоты (FLL)
fll = FrequencyLockedLoop(Kp=0.001, Ki=0.00005)
signal_freq_corrected, freq = fll.process_signal(signal_aligned)

# 4. Синхронизация тактовой частоты
signal_recovered, errors, mu = gardner_timing_recovery(
    signal_freq_corrected, sps=4, alpha=0.05
)

# 5. Демодуляция (решение по символам)
symbols = np.sign(signal_recovered.real) + 1j * np.sign(signal_recovered.imag)
```

---

## 🔧 Расширение функциональности

### Добавление нового детектора

```python
class FrequencyErrorDetector:
    def __init__(self, method='cross_product'):
        self.method = method
        self.prev_sample = 0.0 + 0.0j
    
    def _my_new_detector(self, sample):
        """Ваш новый метод детектирования"""
        # Реализация
        error = ...
        return error
    
    def detect(self, sample):
        if self.method == 'my_new_method':
            error = self._my_new_detector(sample)
        # ... остальные методы
        
        self.prev_sample = sample
        return error
```

### Добавление PID контроллера

```python
class LoopFilter:
    def __init__(self, Kp=0.01, Ki=0.001, Kd=0.0, freq_limit=0.5):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd  # дифференциальный коэффициент
        self.integrator = 0.0
        self.prev_error = 0.0
    
    def update(self, error):
        self.integrator += error * self.Ki
        self.integrator = np.clip(self.integrator, -self.freq_limit, self.freq_limit)
        
        # PID = P + I + D
        derivative = (error - self.prev_error) * self.Kd
        freq_correction = error * self.Kp + self.integrator + derivative
        
        self.prev_error = error
        return np.clip(freq_correction, -self.freq_limit, self.freq_limit)
```

---

## 📞 Контакты и поддержка

**Проект:** DSP Signal Processing  
**Модуль:** Frequency Locked Loop (FLL)  
**Версия:** 1.0  
**Дата:** Ноябрь 2025

---

## 📝 Лицензия

Проект создан в образовательных целях.

---

**🎉 Спасибо за использование FLL!**

Для детальной информации см. `FLL_GUIDE.md`  
Для быстрого старта см. `FLL_CHEATSHEET.md`

