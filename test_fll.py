"""
Тесты для проверки работоспособности FLL
"""

import numpy as np
import matplotlib.pyplot as plt
from fll import FrequencyLockedLoop, FrequencyErrorDetector, LoopFilter, NCO


def test_nco():
    """Тест NCO (Numerically Controlled Oscillator)"""
    print("=" * 60)
    print("ТЕСТ 1: NCO")
    print("=" * 60)
    
    nco = NCO(initial_freq=0.0)
    
    # Устанавливаем частоту 0.01
    nco.freq = 0.01
    
    # Генерируем 100 отсчётов
    output = []
    for _ in range(100):
        output.append(nco.step())
    
    output = np.array(output)
    
    # Проверяем, что это комплексный экспоненциал
    phases = np.angle(output)
    phases_unwrapped = np.unwrap(phases)
    phase_diffs = np.diff(phases_unwrapped)
    
    # Разность фаз должна быть примерно -2π * 0.01 (минус, т.к. exp(-j*phase))
    expected_phase_diff = -2 * np.pi * 0.01
    actual_phase_diff = np.mean(phase_diffs)
    
    print(f"Установленная частота: 0.01")
    print(f"Ожидаемый сдвиг фазы: {expected_phase_diff:.6f} рад (с минусом)")
    print(f"Фактический сдвиг фазы: {actual_phase_diff:.6f} рад")
    print(f"Ошибка: {abs(actual_phase_diff - expected_phase_diff):.6f} рад")
    
    if abs(actual_phase_diff - expected_phase_diff) < 0.001:
        print("✅ NCO работает корректно!")
    else:
        print("❌ NCO работает некорректно!")
    
    print()


def test_frequency_error_detector():
    """Тест детектора ошибки частоты"""
    print("=" * 60)
    print("ТЕСТ 2: Frequency Error Detector")
    print("=" * 60)
    
    # Создаём QPSK сигнал с частотным сдвигом
    f_offset = 0.05  # 5% частотный сдвиг
    n = np.arange(1000)
    signal = np.exp(1j * 2 * np.pi * f_offset * n)
    
    methods = ['cross_product', 'atan2']
    
    for method in methods:
        detector = FrequencyErrorDetector(method=method)
        
        errors = []
        for sample in signal[:100]:
            error = detector.detect(sample)
            errors.append(error)
        
        # Средняя ошибка должна быть пропорциональна f_offset
        avg_error = np.mean(errors[10:])  # пропускаем первые отсчёты
        
        print(f"\nМетод: {method}")
        print(f"Средняя ошибка: {avg_error:.6f}")
        print(f"Частотный сдвиг: {f_offset}")
        
        if method == 'cross_product':
            # Для cross_product: error ≈ sin(2π·f) ≈ 2π·f для малых f
            expected_error = np.sin(2 * np.pi * f_offset)
            print(f"Ожидаемая ошибка: {expected_error:.6f}")
            
            if abs(avg_error - expected_error) < 0.1:
                print(f"✅ {method} работает корректно!")
            else:
                print(f"⚠️ {method} показывает отклонение")
        
        elif method == 'atan2':
            # Для atan2: error = atan2(sin, cos) ≈ 2π·f
            expected_error = 2 * np.pi * f_offset
            print(f"Ожидаемая ошибка: {expected_error:.6f}")
            
            if abs(avg_error - expected_error) < 0.1:
                print(f"✅ {method} работает корректно!")
            else:
                print(f"⚠️ {method} показывает отклонение")
    
    print()


def test_loop_filter():
    """Тест петлевого фильтра"""
    print("=" * 60)
    print("ТЕСТ 3: Loop Filter")
    print("=" * 60)
    
    loop_filter = LoopFilter(Kp=0.1, Ki=0.01, freq_limit=0.5)
    
    # Подаём постоянную ошибку
    constant_error = 0.1
    outputs = []
    
    for _ in range(100):
        output = loop_filter.update(constant_error)
        outputs.append(output)
    
    outputs = np.array(outputs)
    
    # Выход должен расти (из-за интегратора)
    print(f"Начальный выход: {outputs[0]:.6f}")
    print(f"Конечный выход: {outputs[-1]:.6f}")
    
    if outputs[-1] > outputs[0]:
        print("✅ Интегратор работает!")
    else:
        print("❌ Интегратор не работает!")
    
    # Проверяем ограничение
    if abs(outputs[-1]) <= 0.5:
        print("✅ Ограничение работает!")
    else:
        print("❌ Ограничение не работает!")
    
    print()


def test_fll_convergence():
    """Тест сходимости FLL"""
    print("=" * 60)
    print("ТЕСТ 4: Сходимость FLL")
    print("=" * 60)
    
    # Параметры сигнала
    num_symbols = 500
    sps = 4
    f_offset = 0.02  # 2% частотный сдвиг
    
    # Генерация QPSK
    np.random.seed(42)
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
    signal = np.repeat(symbols, sps)
    
    # Добавляем частотный сдвиг
    n = np.arange(len(signal))
    signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
    
    # Добавляем шум
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * 0.1
    signal_with_offset += noise
    
    # Создаём FLL
    fll = FrequencyLockedLoop(
        detector_method='cross_product',
        Kp=0.005,
        Ki=0.0001,
        freq_limit=0.1
    )
    
    # Обрабатываем
    corrected_signal, freq_estimate = fll.process_signal(signal_with_offset)
    
    print(f"Истинный частотный сдвиг: {f_offset:.6f}")
    print(f"Оценка FLL: {freq_estimate:.6f}")
    print(f"Ошибка оценки: {abs(freq_estimate - f_offset):.6f}")
    
    # Проверяем точность
    error = abs(freq_estimate - f_offset)
    if error < 0.005:  # ошибка меньше 0.5%
        print("✅ FLL сходится с высокой точностью!")
    elif error < 0.01:  # ошибка меньше 1%
        print("⚠️ FLL сходится, но с умеренной точностью")
    else:
        print("❌ FLL не сходится!")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # До коррекции
    axes[0, 0].plot(signal_with_offset[::sps].real[:200], 
                    signal_with_offset[::sps].imag[:200], 
                    'o', markersize=4, alpha=0.6)
    axes[0, 0].set_title(f'До FLL (f_offset={f_offset:.4f})')
    axes[0, 0].set_xlabel('I')
    axes[0, 0].set_ylabel('Q')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # После коррекции
    axes[0, 1].plot(corrected_signal[::sps].real[:200], 
                    corrected_signal[::sps].imag[:200], 
                    'o', markersize=4, alpha=0.6)
    axes[0, 1].set_title(f'После FLL (оценка={freq_estimate:.6f})')
    axes[0, 1].set_xlabel('I')
    axes[0, 1].set_ylabel('Q')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Оценка частоты во времени
    axes[1, 0].plot(fll.history['freq_estimates'])
    axes[1, 0].axhline(f_offset, color='r', linestyle='--', label='Истинное значение')
    axes[1, 0].axhline(freq_estimate, color='g', linestyle='--', label='Финальная оценка')
    axes[1, 0].set_title('Сходимость оценки частоты')
    axes[1, 0].set_xlabel('Номер отсчёта')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Ошибка частоты
    axes[1, 1].plot(fll.history['errors'])
    axes[1, 1].set_title('Ошибка детектора')
    axes[1, 1].set_xlabel('Номер отсчёта')
    axes[1, 1].set_ylabel('Ошибка')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_fll_convergence.png', dpi=150)
    print("\nСохранено: test_fll_convergence.png")
    plt.show()
    
    print()


def test_different_methods():
    """Сравнение разных методов детектирования"""
    print("=" * 60)
    print("ТЕСТ 5: Сравнение методов детектирования")
    print("=" * 60)
    
    # Параметры
    num_symbols = 300
    sps = 4
    f_offset = 0.015
    
    # Сигнал
    np.random.seed(42)
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
    signal = np.repeat(symbols, sps)
    n = np.arange(len(signal))
    signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
    
    # Шум
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * 0.1
    signal_with_offset += noise
    
    # Тестируем методы
    methods = ['cross_product', 'atan2', 'decision_directed']
    results = {}
    
    for method in methods:
        fll = FrequencyLockedLoop(
            detector_method=method,
            Kp=0.005,
            Ki=0.0001,
            freq_limit=0.1
        )
        
        _, freq_estimate = fll.process_signal(signal_with_offset)
        error = abs(freq_estimate - f_offset)
        
        results[method] = {
            'estimate': freq_estimate,
            'error': error
        }
        
        print(f"\n{method}:")
        print(f"  Оценка: {freq_estimate:.6f}")
        print(f"  Ошибка: {error:.6f}")
        
        if error < 0.005:
            print(f"  ✅ Отлично!")
        elif error < 0.01:
            print(f"  ⚠️ Приемлемо")
        else:
            print(f"  ❌ Плохо")
    
    # Лучший метод
    best_method = min(results.keys(), key=lambda k: results[k]['error'])
    print(f"\n🏆 Лучший метод для этого сигнала: {best_method}")
    
    print()


def run_all_tests():
    """Запуск всех тестов"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "ТЕСТИРОВАНИЕ FLL" + " " * 27 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    test_nco()
    test_frequency_error_detector()
    test_loop_filter()
    test_fll_convergence()
    test_different_methods()
    
    print("=" * 60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

