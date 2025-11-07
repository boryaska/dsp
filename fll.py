"""
Frequency Locked Loop (FLL) - Петля фазовой автоподстройки частоты
Модульная реализация с отдельными блоками
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# БЛОК 1: FREQUENCY ERROR DETECTOR (Детектор ошибки частоты)
# ============================================================================

class FrequencyErrorDetector:
    """
    Детектор ошибки частоты на основе перекрестного произведения
    (Cross-Product Frequency Discriminator)
    
    Принцип работы:
    - Берем текущий символ и предыдущий: s[n] и s[n-1]
    - Вычисляем: e[n] = imag(s[n] * conj(s[n-1]))
    - Это дает оценку мгновенной частотной ошибки
    """
    
    def __init__(self, method='cross_product'):
        """
        Args:
            method: метод детектирования
                'cross_product' - перекрестное произведение (классика)
                'atan2' - арктангенс (более точный, но медленнее)
                'decision_directed' - с принятием решений (для высоких SNR)
        """
        self.method = method
        self.prev_sample = 0.0 + 0.0j
        
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
        if self.method == 'cross_product':
            error = self._cross_product(sample)
        elif self.method == 'atan2':
            error = self._atan2_method(sample)
        elif self.method == 'decision_directed':
            error = self._decision_directed(sample)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        self.prev_sample = sample
        return error
    
    def _cross_product(self, sample):
        """
        Классический метод перекрестного произведения
        Простой и быстрый, работает для малых ошибок
        """
        # e = Im[s[n] * conj(s[n-1])]
        product = sample * np.conj(self.prev_sample)
        error = np.imag(product)
        return error
    
    def _atan2_method(self, sample):
        """
        Метод с использованием atan2
        Более точный для больших ошибок частоты
        """
        product = sample * np.conj(self.prev_sample)
        error = np.angle(product)  # эквивалентно atan2(imag, real)
        return error
    
    def _decision_directed(self, sample):
        """
        Метод с принятием решений (Decision-Directed)
        Для QPSK: определяем ближайший символ и используем его как reference
        """
        # Жесткое решение для QPSK (приводим к ±1±j)
        decided_sample = np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample))
        decided_prev = np.sign(np.real(self.prev_sample)) + 1j * np.sign(np.imag(self.prev_sample))
        
        # Вычисляем ошибку относительно идеальных символов
        product = (sample * np.conj(decided_sample)) * np.conj(self.prev_sample * np.conj(decided_prev))
        error = np.imag(product)
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
    
    Выход фильтра управляет NCO для коррекции частоты
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
        Обновляет состояние фильтра на основе ошибки
        
        Args:
            error: ошибка частоты от детектора
            
        Returns:
            freq_correction: коррекция частоты для NCO
        """
        # Интегрируем ошибку
        self.integrator += error * self.Ki
        
        # Ограничиваем интегратор (anti-windup)
        self.integrator = np.clip(self.integrator, -self.freq_limit, self.freq_limit)
        
        # Выход = пропорциональная часть + интегральная часть
        freq_correction = error * self.Kp + self.integrator
        
        # Ограничиваем выход
        freq_correction = np.clip(freq_correction, -self.freq_limit, self.freq_limit)
        
        return freq_correction


# ============================================================================
# БЛОК 3: NCO (Numerically Controlled Oscillator)
# ============================================================================

class NCO:
    """
    Численно-управляемый генератор (NCO)
    Генерирует комплексный экспоненциал для коррекции частоты
    """
    
    def __init__(self, initial_freq=0.0):
        """
        Args:
            initial_freq: начальная частота (в долях от частоты дискретизации)
        """
        self.freq = initial_freq
        self.phase = 0.0
        
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
        
        # Обновляем фазу
        self.phase += 2 * np.pi * self.freq
        
        # Держим фазу в пределах [-π, π]
        self.phase = np.arctan2(np.sin(self.phase), np.cos(self.phase))
        
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
                 freq_limit=0.5):
        """
        Args:
            detector_method: метод детектирования ошибки
            Kp: пропорциональный коэффициент
            Ki: интегральный коэффициент
            freq_limit: ограничение частоты
        """
        self.detector = FrequencyErrorDetector(method=detector_method)
        self.loop_filter = LoopFilter(Kp=Kp, Ki=Ki, freq_limit=freq_limit)
        self.nco = NCO()
        
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
        """
        # 1. Корректируем входной сигнал текущей оценкой
        correction = self.nco.step()
        corrected_sample = sample * correction
        
        # 2. Детектируем ошибку частоты
        error = self.detector.detect(corrected_sample)
        
        # 3. Фильтруем ошибку
        freq_correction = self.loop_filter.update(error)
        
        # 4. Обновляем NCO
        self.nco.update_freq(freq_correction)
        
        # Сохраняем историю
        self.history['errors'].append(error)
        self.history['freq_estimates'].append(self.nco.get_current_freq())
        self.history['phases'].append(self.nco.phase)
        
        return corrected_sample
    
    def process_signal(self, signal):
        """
        Обрабатывает весь сигнал
        
        Args:
            signal: массив комплексных отсчетов
            
        Returns:
            corrected_signal: скорректированный сигнал
            freq_estimate: финальная оценка частоты
        """
        corrected_signal = np.zeros_like(signal, dtype=complex)
        
        for i, sample in enumerate(signal):
            corrected_signal[i] = self.process_sample(sample)
        
        freq_estimate = self.nco.get_current_freq()
        
        return corrected_signal, freq_estimate
    
    def get_frequency_estimate(self):
        """Возвращает текущую оценку частоты"""
        return self.nco.get_current_freq()
    
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
    np.random.seed(42)
    
    # Параметры
    num_symbols = 1000
    sps = 4  # samples per symbol
    f_offset = 0.01  # частотный сдвиг (10% от частоты символов)
    
    # Генерация QPSK символов
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
    
    # Повышение частоты дискретизации (без фильтрации для простоты)
    signal = np.repeat(symbols, sps)
    
    # Добавляем частотный сдвиг
    n = np.arange(len(signal))
    signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
    
    # Добавляем шум
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * 0.1
    signal_with_offset += noise
    
    print(f"Истинный частотный сдвиг: {f_offset:.6f}")
    
    # Создаем FLL с разными методами
    methods = ['cross_product', 'atan2', 'decision_directed']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Метод: {method}")
        print(f"{'='*60}")
        
        # Создаем FLL
        fll = FrequencyLockedLoop(
            detector_method=method,
            Kp=0.005,  # пропорциональный коэффициент
            Ki=0.0001,  # интегральный коэффициент
            freq_limit=0.1
        )
        
        # Обрабатываем сигнал
        corrected_signal, freq_estimate = fll.process_signal(signal_with_offset)
        
        print(f"Оценка частоты FLL: {freq_estimate:.6f}")
        print(f"Ошибка оценки: {abs(freq_estimate - f_offset):.6f}")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # До коррекции
        axes[0].plot(signal_with_offset[::sps].real, 
                    signal_with_offset[::sps].imag, 
                    'o', markersize=4, alpha=0.5)
        axes[0].set_title(f'До FLL (f_offset={f_offset:.4f})')
        axes[0].set_xlabel('I')
        axes[0].set_ylabel('Q')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # После коррекции
        axes[1].plot(corrected_signal[::sps].real, 
                    corrected_signal[::sps].imag, 
                    'o', markersize=4, alpha=0.5)
        axes[1].set_title(f'После FLL (метод: {method}, оценка={freq_estimate:.6f})')
        axes[1].set_xlabel('I')
        axes[1].set_ylabel('Q')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Показываем работу петли
        fll.plot_performance()

