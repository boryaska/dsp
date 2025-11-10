"""
Frequency Locked Loop (FLL) - Петля автоподстройки частоты
Модульная реализация с отдельными блоками

КАК РАБОТАЕТ КОРРЕКЦИЯ ЧАСТОТЫ:
=================================

Проблема:
---------
Сигнал приходит с частотным сдвигом: s(t) = s₀(t)·exp(j·2π·Δf·t)
Это заставляет созвездие вращаться → демодуляция невозможна

Решение - замкнутая петля обратной связи:
------------------------------------------

┌─────────────────────────────────────────────────────────┐
│                    FLL ПЕТЛЯ                             │
│                                                          │
│  Вход: signal[n]                                         │
│     ↓                                                    │
│  [×] ← exp(-j·phase_nco)  ← NCO (шаг 4)                 │
│     ↓                             ↑                      │
│  corrected_signal          freq_nco += Δf                │
│     ↓                             ↑                      │
│  [Detector] → Δφ (фаза)    [Loop Filter]                │
│     измеряет ошибку        PI-контроллер:                │
│     Δφ = angle(rx) -       Δf = Kp·Δφ + Ki·Σ(Δφ)        │
│          angle(ideal)                                    │
│                                                          │
└─────────────────────────────────────────────────────────┘

Ключевая идея:
--------------
Фазовая ошибка Δφ ≈ 2π·Δf_остаточная·T_symbol

При частотном сдвиге фаза накапливается линейно.
Измеряя фазовую ошибку между символами, мы получаем информацию о частотном сдвиге.

Логика работы:
- Если Δφ > 0: остаточная частота > 0 → увеличиваем freq_nco
- Если Δφ < 0: остаточная частота < 0 → уменьшаем freq_nco

PI-контроллер настраивает NCO так, чтобы:
    freq_nco → f_error (сходимость к истинной частоте)

Результат:
----------
После сходимости NCO генерирует точно обратную частоту:
    signal · exp(-j·2π·f_nco·t) ≈ signal · exp(-j·2π·f_error·t)⁻¹
    
Частотный сдвиг компенсируется → созвездие стабилизируется!
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# БЛОК 1: FREQUENCY ERROR DETECTOR (Детектор ошибки частоты)
# ============================================================================

class FrequencyErrorDetector:
    """
    Детектор ошибки частоты на основе фазовой ошибки
    (Phase Error Frequency Discriminator)
    
    Принцип работы:
    1. Определяем ближайший идеальный символ созвездия (decision-directed)
    2. Вычисляем фазовую ошибку: Δφ = angle(received) - angle(decided)
    3. При частотном сдвиге Δf фаза накапливается: φ(t) = 2π·Δf·t
    4. Фазовая ошибка между символами пропорциональна частотной ошибке
    5. PI-контроллер интегрирует эти ошибки → компенсирует частоту
    
    Защита от выбросов:
    - Если |Δφ| > π/9 (20°) - используем предыдущую ошибку
    - Это защищает от шума и неправильных решений
    """
    
    def __init__(self, method='decision_directed'):
        """
        Args:
            method: метод детектирования
                'decision_directed' - с принятием решений (для высоких SNR)
        """
        self.method = method
        # self.prev_sample = 0.0 + 0.0j
        self.previous_error = 0.0
        
    def reset(self):
        """Сброс состояния детектора"""
        self.prev_sample = 0.0 + 0.0j
    
    def detect(self, sample):
        """
        Детектирует ошибку частоты для одного отсчета
        
        Args:
            sample: комплексный отсчет сигнала
            
        Returns:
            frequency_error: оценка ошибки частоты (в радианах)
        """
        error = self._decision_directed(sample)
        
        # self.prev_sample = sample
        return error
    
    def _decision_directed(self, sample):
        """
        Метод с принятием решений (Decision-Directed)
        
        Для QPSK определяем ближайшую точку созвездия:
        - I: sign(Re) → ±1/√2
        - Q: sign(Im) → ±1/√2
        
        Фазовая ошибка как индикатор частоты:
        -----------------------------------------------
        Δφ = angle(received) - angle(ideal)
        
        При частотном сдвиге Δf:
        - Фаза вращается: φ(t) = 2π·Δf·t
        - Между символами: Δφ ≈ 2π·Δf·T_symbol
        - Эта Δφ используется для коррекции частоты
        
        Защита от выбросов:
        - Если |Δφ| > 20° → ошибка слишком большая (шум/плохое решение)
        - Используем предыдущее значение вместо текущего
        """
        # Находим ближайший идеальный символ QPSK
        decided_sample = (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample)))/np.sqrt(2)
        
        # Вычисляем фазовую ошибку
        # angle = np.angle(sample) - np.angle(decided_sample)
        angle = np.angle(sample * np.conj(decided_sample))
        
        # # Защита от выбросов (outlier rejection)
    
        # if abs(angle) > np.pi/9:  # > 45 градусов
        #     error = self.previous_error
        # else:
        #     error = angle

        self.previous_error = error

        return error
    
    def detect_batch(self, samples):
        """
        Детектирует ошибки для массива отсчетов
        
        Args:
            samples: массив комплексных отсчетов
            
        Returns:
            errors: массив оценок ошибок частоты
        """
        errors = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            errors[i] = self.detect(sample)
        return errors


# ============================================================================
# БЛОК 2: LOOP FILTER (Петлевой фильтр)
# ============================================================================

class LoopFilter:
    """
    Петлевой фильтр для FLL
    Реализует PI-контроллер (пропорционально-интегральный)
    
    Как фазовая ошибка превращается в коррекцию частоты:
    =====================================================
    
    Вход: Δφ[n] - фазовая ошибка от детектора
    
    Выход: Δf[n] - коррекция частоты для NCO
           Δf[n] = Kp·Δφ[n] + Ki·Σ(Δφ[k])
    
    Логика обратной связи:
    - Если Δφ > 0: остаточная частота положительная (сигнал вращается)
      → нужно УВЕЛИЧИТЬ freq_nco, чтобы больше компенсировать
      → phase_nco растет быстрее → вычитаем больше
    
    - Если Δφ < 0: остаточная частота отрицательная (перекомпенсация)
      → нужно УМЕНЬШИТЬ freq_nco
      → phase_nco растет медленнее → вычитаем меньше
    
    Защита от насыщения (anti-windup):
    - Ограничиваем интегратор: [-freq_limit, +freq_limit]
    - Предотвращает "зависание" при больших ошибках
    """
    
    def __init__(self, Kp=0.01, Ki=0.001, freq_limit=0.5):
        """
        Args:
            Kp: пропорциональный коэффициент усиления
            Ki: интегральный коэффициент усиления
            freq_limit: ограничение на частоту (в долях от частоты дискретизации)
        """
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.freq_limit = freq_limit
        
        # Состояние интегратора
        self.integrator = 0.0
        
    def reset(self):
        """Сброс состояния фильтра"""
        self.integrator = 0.0
    
    def update(self, error):
        """
        Обновляет состояние фильтра на основе фазовой ошибки
        
        Args:
            error: Δφ - фазовая ошибка от детектора (в радианах)
            
        Returns:
            freq_correction: Δf - коррекция частоты для NCO
        
        Преобразование: Δφ → Δf
        ------------------------
        1. P-часть: мгновенный отклик = Kp·Δφ
        2. I-часть: накопление = Σ(Ki·Δφ)
        3. Итого: Δf = Kp·Δφ + integrator
        
        Логика:
        - Δφ > 0: остаточная частота > 0 → увеличиваем freq
        - Δφ < 0: остаточная частота < 0 → уменьшаем freq
        
        NCO использует Δf для изменения своей частоты:
        freq_nco += Δf
        """
        # Интегрируем фазовую ошибку
        # Δφ > 0 → увеличиваем freq_nco для компенсации
        self.integrator += error * self.Ki
        
        # Ограничиваем интегратор (anti-windup)
        # Предотвращает слишком большие накопленные значения
        # self.integrator = np.clip(self.integrator, -self.freq_limit, self.freq_limit)
        
        # PI-контроллер
        # Пропорциональная часть (Kp·error): быстрая реакция
        # Интегральная часть (integrator): устранение постоянной ошибки
        freq_correction = error * self.Kp + self.integrator
        # freq_correction = error * self.Kp
        
        # Ограничиваем итоговую коррекцию
        # freq_correction = np.clip(freq_correction, -self.freq_limit, self.freq_limit)
        
        return freq_correction


# ============================================================================
# БЛОК 3: NCO (Numerically Controlled Oscillator)
# ============================================================================

class NCO:
    """
    Численно-управляемый генератор (NCO)
    Генерирует комплексный экспоненциал для коррекции частоты
    
    Принцип работы:
    ===============
    
    1. Накапливает частоту: freq += Δf (от loop filter)
    2. Накапливает фазу: phase += 2π·freq
    3. Генерирует: exp(-j·phase)
    4. Умножаем на сигнал: signal·exp(-j·phase)
    
    Результат: убираем частотный сдвиг
    
    Пример:
    -------
    Сигнал: s(t) = exp(j·2π·f_error·t)  ← частотный сдвиг
    NCO:    c(t) = exp(-j·2π·f_nco·t)    ← компенсация
    Выход:  s·c  = exp(j·2π·(f_error - f_nco)·t)
    
    Когда f_nco → f_error, выход становится постоянным (нет вращения)
    """
    
    def __init__(self, initial_freq=0.0):
        """
        Args:
            initial_freq: начальная частота (в долях от частоты дискретизации)
        """
        self.freq = initial_freq
        self.phase = 0.0
        self.n = 0
        
    def reset(self):
        """Сброс состояния NCO"""
        self.phase = 0.0
    
    def update_freq(self, freq_correction):
        """
        Обновляет частоту NCO
        
        Args:
            freq_correction: коррекция частоты от петлевого фильтра
        """
        self.freq += freq_correction
    
    def step(self):
        """
        Один шаг NCO - генерирует следующий отсчет
        
        Returns:
            correction_signal: комплексный экспоненциал для коррекции
        """
        # Генерируем e^(-j*phase) для удаления частотного сдвига
        correction_signal = np.exp(-1j * self.phase)
        self.n += 1
        
        # Обновляем фазу
        self.phase = 2 * np.pi * self.freq * n
        
        # Держим фазу в пределах [-π, π]
        # self.phase = np.arctan2(np.sin(self.phase), np.cos(self.phase))
        
        return correction_signal
    
    def get_current_freq(self):
        """Возвращает текущую оценку частоты"""
        return self.freq


# ============================================================================
# БЛОК 4: FLL (Frequency Locked Loop) - Интеграция всех блоков
# ============================================================================

class FrequencyLockedLoop:
    """
    Полная реализация Frequency Locked Loop
    Объединяет детектор, фильтр и NCO
    """
    
    def __init__(self, 
                 detector_method='cross_product',
                 Kp=0.01, 
                 Ki=0.001,
                 freq_limit=0.5,
                 auto_acquire=False,
                 acquire_samples=50):
        """
        Args:
            detector_method: метод детектирования ошибки
            Kp: пропорциональный коэффициент
            Ki: интегральный коэффициент
            freq_limit: ограничение частоты
            auto_acquire: автоматически оценить начальную частоту перед tracking
            acquire_samples: количество символов для оценки начальной частоты (по умолчанию 50)
        """
        self.detector = FrequencyErrorDetector(method=detector_method)
        self.loop_filter = LoopFilter(Kp=Kp, Ki=Ki, freq_limit=freq_limit)
        self.nco = NCO()
        
        # Параметры автоматической оценки частоты
        self.auto_acquire = auto_acquire
        self.acquire_samples = acquire_samples
        
        # История для отладки
        self.history = {
            'errors': [],
            'freq_estimates': [],
            'phases': []
        }
        
    def reset(self):
        """Сброс всех блоков петли"""
        self.detector.reset()
        self.loop_filter.reset()
        self.nco.reset()
        self.history = {
            'errors': [],
            'freq_estimates': [],
            'phases': []
        }
    
    def process_sample(self, sample):
        """
        Обрабатывает один отсчет сигнала
        
        Args:
            sample: входной комплексный отсчет
            
        Returns:
            corrected_sample: скорректированный отсчет
        
        Полный цикл коррекции частоты:
        ================================
        
        Вход: signal[n] с частотным сдвигом Δf_error
        
        Шаг 1: NCO генерирует компенсацию
               correction = exp(-j·phase_nco)
               corrected = signal·correction
        
        Шаг 2: Детектор измеряет фазовую ошибку
               Δφ = angle(corrected) - angle(ideal_symbol)
               
        Шаг 3: Loop Filter преобразует Δφ → Δf
               Δf = Kp·Δφ + Ki·Σ(Δφ)
               
               Логика: Δφ > 0 (остаточная частота > 0) → Δf > 0 (увеличиваем freq_nco)
                       Δφ < 0 (остаточная частота < 0) → Δf < 0 (уменьшаем freq_nco)
               
        Шаг 4: NCO обновляет частоту
               freq_nco += Δf
               phase_nco += 2π·freq_nco
        
        Результат: freq_nco → f_error (петля сходится к истинной частоте)
        """
        # ============== ШАГ 1: КОРРЕКЦИЯ СИГНАЛА ==============
        # NCO генерирует exp(-j·phase) для компенсации частотного сдвига
        correction = self.nco.step()
        corrected_sample = sample * correction
        
        # ============== ШАГ 2: ДЕТЕКЦИЯ ОШИБКИ ==============
        # Измеряем фазовую ошибку: Δφ = angle(received) - angle(decided)
        error = self.detector.detect(corrected_sample)
        
        # ============== ШАГ 3: ФИЛЬТРАЦИЯ (Δφ → Δf) ==============
        # PI-контроллер преобразует фазовую ошибку в частотную коррекцию
        freq_correction = self.loop_filter.update(error)
        
        # ============== ШАГ 4: ОБНОВЛЕНИЕ NCO ==============
        # Обновляем частоту NCO: freq += Δf
        self.nco.update_freq(freq_correction)
        
        # Сохраняем историю для анализа
        self.history['errors'].append(error)
        self.history['freq_estimates'].append(self.nco.get_current_freq())
        self.history['phases'].append(self.nco.phase)
        
        return corrected_sample
    
    def process_signal(self, signal, debug=False, debug_samples=50):
        """
        Обрабатывает весь сигнал
        
        Args:
            signal: массив комплексных отсчетов
            debug: если True, выводит отладочную информацию
            debug_samples: количество первых отсчетов для отладки
            
        Returns:
            corrected_signal: скорректированный сигнал
            freq_estimate: финальная оценка частоты
        """
        corrected_signal = np.zeros_like(signal, dtype=complex)
        
        if debug:
            print(f"\n{'='*100}")
            print(f"{'ОТЛАДКА FLL - Первые {debug_samples} итераций':^100}")
            print(f"{'='*100}")
            print("\nЛегенда:")
            print("  #       - номер итерации (отсчет/символ)")
            print("  |Sample| - амплитуда входного отсчета")
            print("  ∠Sample - фаза входного отсчета (радианы)")
            print("  NCO_phase - фаза NCO перед коррекцией")
            print("  |Corr|  - амплитуда после коррекции")
            print("  ∠Corr   - фаза после коррекции")
            print("  Δφ      - фазовая ошибка (angle(received) - angle(decided))")
            print("  Δf      - изменение частоты NCO на этом шаге")
            print("  f_NCO   - текущая частота NCO")
            print(f"{'-'*100}")
            print(f"{'#':>4} {'|Sample|':>10} {'∠Sample':>10} {'NCO_phase':>12} "
                  f"{'|Corr|':>10} {'∠Corr':>10} {'Δφ':>10} {'Δf':>10} {'f_NCO':>10}")
            print(f"{'-'*100}")
        
        # Автоматическая оценка начальной частоты (frequency acquisition)
        if self.auto_acquire and len(signal) > self.acquire_samples:
            initial_freq = self._estimate_initial_frequency(signal, self.acquire_samples)
            self.nco.freq = initial_freq
            if debug:
                print(f"[Auto-Acquire] Оценка начальной частоты: {initial_freq:.6f} "
                      f"(по первым {self.acquire_samples} символам)")
                print(f"{'-'*100}")
        
        for i, sample in enumerate(signal):
            # Сохраняем состояние до обработки
            if debug and i < debug_samples:
                phase_before = self.nco.phase
                freq_before = self.nco.freq
            
            corrected_signal[i] = self.process_sample(sample)
            
            # Выводим информацию после обработки
            if debug and i < debug_samples:
                error = self.history['errors'][-1]
                freq_now = self.history['freq_estimates'][-1]
                
                print(f"{i:4d} "
                      f"{abs(sample):10.4f} "
                      f"{np.angle(sample):10.4f} "
                      f"{phase_before:12.4f} "
                      f"{abs(corrected_signal[i]):10.4f} "
                      f"{np.angle(corrected_signal[i]):10.4f} "
                      f"{error:10.6f} "
                      f"{freq_now - freq_before:10.6f} "
                      f"{freq_now:10.6f}")
        
        if debug:
            print(f"{'-'*100}\n")
        
        freq_estimate = self.nco.get_current_freq()
        
        return corrected_signal, freq_estimate
    
    def get_frequency_estimate(self):
        """Возвращает текущую оценку частоты"""
        return self.nco.get_current_freq()
    
    def _estimate_initial_frequency(self, signal, num_samples):
        """
        Грубая оценка частотного сдвига по первым num_samples символам
        
        Метод: Fourth-power method для QPSK
        -------------------------------------
        1. Возводим сигнал в 4-ю степень: z[n] = s[n]^4
        2. Для QPSK это полностью убирает модуляцию:
           s[n] = a[n]·exp(j·2π·Δf·n), где a[n] ∈ {±1±j}/√2
           s[n]^4 = a[n]^4·exp(j·8π·Δf·n)
           a[n]^4 = const для любого QPSK символа
        3. Вычисляем автокорреляцию с задержкой 1:
           R = Σ z[n]·conj(z[n-1]) = exp(j·8π·Δf)
        4. Частота: Δf = angle(R) / (8π)
        
        Преимущества:
        - Полностью убирает QPSK модуляцию
        - Точность не зависит от случайности символов
        - Стандартный метод в DSP
        
        Args:
            signal: массив комплексных символов
            num_samples: количество символов для анализа
            
        Returns:
            freq_estimate: оценка частотного сдвига
        """
        # Используем только доступные отсчеты
        n = min(num_samples, len(signal))
        if n < 2:
            return 0.0
        
        # Возводим сигнал в 4-ю степень для удаления QPSK модуляции
        signal_4th = signal[:n] ** 4
        
        # Вычисляем автокорреляцию с задержкой 1
        autocorr = 0.0 + 0.0j
        for i in range(1, n):
            autocorr += signal_4th[i] * np.conj(signal_4th[i-1])
        
        # Усредняем
        autocorr /= (n - 1)
        
        # Извлекаем фазовый сдвиг
        # s^4 создает 4x частотное умножение: phase = 8π·Δf
        phase_shift = np.angle(autocorr)
        
        # Частота = фазовый сдвиг / (8π)
        # Делим на 8π, т.к. s^4 умножает частоту на 4, 
        # а автокорреляция с задержкой 1 добавляет еще x2
        freq_estimate = phase_shift / (8 * np.pi)
        
        print(f"[Auto-Acquire Debug] phase_shift={phase_shift:.6f}, freq_estimate={freq_estimate:.6f}")
        
        return freq_estimate
    
    def plot_performance(self):
        """Визуализирует работу петли"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # График ошибки частоты
        axes[0].plot(self.history['errors'])
        axes[0].set_title('Ошибка частоты')
        axes[0].set_xlabel('Номер отсчета')
        axes[0].set_ylabel('Ошибка')
        axes[0].grid(True, alpha=0.3)
        
        # График оценки частоты
        axes[1].plot(self.history['freq_estimates'])
        axes[1].set_title('Оценка частоты NCO')
        axes[1].set_xlabel('Номер отсчета')
        axes[1].set_ylabel('Частота (относительная)')
        axes[1].grid(True, alpha=0.3)
        
        # График фазы NCO
        axes[2].plot(self.history['phases'])
        axes[2].set_title('Фаза NCO')
        axes[2].set_xlabel('Номер отсчета')
        axes[2].set_ylabel('Фаза (радианы)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Создаем тестовый QPSK сигнал с частотным сдвигом
    from diff_method import rrc_filter
    np.random.seed(42)
    
    # Параметры
    num_symbols = 6000
    sps = 2  # samples per symbol
    f_offset = 0.004  # частотный сдвиг (10% от частоты символов)
    
    # Генерация QPSK символов (нормализованные)
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols) / np.sqrt(2)
    
    # Повышение частоты дискретизации
    signal = np.zeros(num_symbols*sps, dtype=complex)
    # signal = np.repeat(symbols, sps)
    for i in range(num_symbols):
        signal[i*sps] = symbols[i]

    # rrc = rc_filter(sps, 10, 0.35)
    # signal = np.convolve(signal, rrc, mode='same')
    # signal = signal / np.std(signal)

    
    # ============================================================================
    # ВАЖНО: Применяем частотный сдвиг К СИМВОЛАМ (после децимации)
    # ============================================================================
    # Если применять сдвиг к oversampled сигналу, а потом брать [::sps],
    # то эффективная частота масштабируется в sps раз!
    
    # Берем только символы (каждый sps-й отсчет)

    n = np.arange(len(signal))
    signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
    signal_with_offset = signal_with_offset[::sps].copy()  # .copy() для создания непрерывного массива
    signal_with_offset = np.asarray(signal_with_offset, dtype=complex).flatten()  # Убедимся что это 1D массив
    
    # Применяем частотный сдвиг к символам
    # n_symbols = np.arange(len(symbols_signal))
    # signal_with_offset = symbols_signal * np.exp(1j * 2 * np.pi * f_offset * n_symbols)
    
    # Добавляем шум
    # noise = (np.random.randn(len(signal_with_offset)) + 1j * np.random.randn(len(signal_with_offset))) * 0.01

    # signal_with_offset += noise

    # signal_with_offset = signal_with_offset * np.exp(1j * 2 * np.pi * 0.05)

    # Визуализация созвездия ДО коррекции (закомментировано для отладки)
    plt.figure(figsize=(10, 10))
    plt.plot(signal_with_offset.real, signal_with_offset.imag, 'o', markersize=3, alpha=0.6)
    plt.title(f'Созвездие ДО FLL\n(f_offset={f_offset:.4f})')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imag)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    print(f"Истинный частотный сдвиг: {f_offset:.6f}")
    
    # Информация о сигнале
    print(f"\n{'='*60}")
    print("ИНФОРМАЦИЯ О СИГНАЛЕ:")
    print(f"{'='*60}")
    print(f"Количество символов:              {num_symbols}")
    print(f"Samples per symbol (sps):         {sps}")
    print(f"Всего отсчетов (oversampled):     {len(signal)}")
    print(f"Частотный сдвиг применен к:       СИМВОЛАМ (после децимации) ✓")
    print(f"Частота сдвига:                   {f_offset:.6f} (относительно частоты символов)")
    print(f"\nПервые 5 символов (до частотного сдвига):")
    for i in range(min(5, len(symbols))):
        print(f"  [{i}] = {symbols[i].real:+.4f} {symbols[i].imag:+.4f}j  "
              f"(|s|={abs(symbols[i]):.4f}, ∠={np.angle(symbols[i]):.4f})")
    
    print(f"\nПервые 5 символов ПОСЛЕ частотного сдвига:")
    for i in range(min(5, len(signal_with_offset))):
        print(f"  [{i}] = {signal_with_offset[i].real:+.4f} {signal_with_offset[i].imag:+.4f}j  "
              f"(|s|={abs(signal_with_offset[i]):.4f}, ∠={np.angle(signal_with_offset[i]):.4f})")
    
    # Создаем FLL
    method = 'decision_directed'
    print(f"\n{'='*60}")
    print(f"Метод: {method}")
    print(f"{'='*60}")

    # Создаем FLL с уменьшенными коэффициентами для стабильности
    fll = FrequencyLockedLoop(
        detector_method=method,
        Kp=0.00001 ,   # пропорциональный коэффициент (уменьшен, чтобы избежать перерегулирования)
        Ki=0.00001, # интегральный коэффициент (медленное накопление для стабильности)
        freq_limit=0.15,
        auto_acquire=False,      # Автоматическая оценка начальной частоты
        acquire_samples=50      # Количество символов для оценки
    )

    # Обрабатываем сигнал с символами (включаем отладку)
    # signal_with_offset уже содержит только символы, не нужна децимация [::sps]
    corrected_signal, freq_estimate = fll.process_signal(
        signal_with_offset, 
        debug=True, 
        debug_samples=800
    )

    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"{'='*60}")
    print(f"Истинный частотный сдвиг:  {f_offset:.6f}")
    print(f"Оценка частоты FLL:        {freq_estimate:.6f}")
    print(f"Абсолютная ошибка:         {abs(freq_estimate - f_offset):.6f}")
    print(f"Относительная ошибка:      {100*abs(freq_estimate - f_offset)/f_offset:.2f}%")
    
    # Дополнительная диагностика
    print(f"\nСтатистика сходимости:")
    print(f"  - Частота применена к символам напрямую ✓")
    print(f"  - FLL обрабатывает {len(signal_with_offset)} символов")
    print(f"  - Финальная оценка: {freq_estimate:.6f}")
    print(f"  - Ошибка: {abs(freq_estimate - f_offset)*1000:.2f} ×10⁻³")

    # Визуализация (раскомментируйте при необходимости)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # # До коррекции - показываем исходные символы с частотным сдвигом
    skip = 100  # пропускаем переходный процесс
    axes[0].plot(signal_with_offset[skip:].real,
                signal_with_offset[skip:].imag,
                'o', markersize=4, alpha=0.5)
    axes[0].set_title(f'До FLL\n(f_offset={f_offset:.4f})')
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # После коррекции - пропускаем переходный процесс FLL
    axes[1].plot(corrected_signal[skip:].real,
                corrected_signal[skip:].imag,
                'o', markersize=4, alpha=0.5)
    axes[1].set_title(f'После FLL\n(метод: {method}, оценка={freq_estimate:.6f})')
    axes[1].set_xlabel('I')
    axes[1].set_ylabel('Q')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()

    # Показываем работу петли
    fll.plot_performance()

