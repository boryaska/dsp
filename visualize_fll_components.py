"""
Визуализация работы отдельных компонентов FLL
Показывает как работает каждый блок по отдельности
"""

import numpy as np
import matplotlib.pyplot as plt
from fll import FrequencyErrorDetector, LoopFilter, NCO


def visualize_frequency_error_detector():
    """Визуализирует работу детектора ошибки частоты"""
    print("Визуализация Frequency Error Detector...")
    
    # Создаём сигналы с разными частотными сдвигами
    freq_offsets = [0.01, 0.03, 0.05, 0.1]
    n = np.arange(500)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    methods = ['cross_product', 'atan2']
    
    for row, method in enumerate(methods):
        for col, f_offset in enumerate([0.01, 0.05]):
            ax = axes[row, col]
            
            # Генерация сигнала
            signal = np.exp(1j * 2 * np.pi * f_offset * n)
            
            # Детектирование
            detector = FrequencyErrorDetector(method=method)
            errors = []
            for sample in signal:
                error = detector.detect(sample)
                errors.append(error)
            
            # Визуализация
            ax.plot(errors, alpha=0.7, linewidth=1)
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            
            if method == 'cross_product':
                expected = np.sin(2 * np.pi * f_offset)
                ax.axhline(expected, color='r', linestyle='--', 
                          label=f'Ожидаемое: {expected:.3f}')
            else:
                expected = 2 * np.pi * f_offset
                ax.axhline(expected, color='r', linestyle='--', 
                          label=f'Ожидаемое: {expected:.3f}')
            
            ax.set_title(f'{method}, f_offset={f_offset}')
            ax.set_xlabel('Отсчёт')
            ax.set_ylabel('Ошибка')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('fll_detector_visualization.png', dpi=150, bbox_inches='tight')
    print("Сохранено: fll_detector_visualization.png\n")
    plt.show()


def visualize_loop_filter():
    """Визуализирует работу петлевого фильтра"""
    print("Визуализация Loop Filter...")
    
    # Тест 1: Постоянная ошибка
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Разные сценарии
    scenarios = [
        {'error_type': 'constant', 'value': 0.1, 'title': 'Постоянная ошибка'},
        {'error_type': 'step', 'value': 0.1, 'title': 'Ступенчатое изменение'},
        {'error_type': 'ramp', 'value': 0.001, 'title': 'Линейное изменение (дрейф)'},
        {'error_type': 'noise', 'value': 0.05, 'title': 'Шумовая ошибка'}
    ]
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        
        # Создаём сигнал ошибки
        n = np.arange(500)
        if scenario['error_type'] == 'constant':
            errors = np.ones(500) * scenario['value']
        elif scenario['error_type'] == 'step':
            errors = np.zeros(500)
            errors[250:] = scenario['value']
        elif scenario['error_type'] == 'ramp':
            errors = n * scenario['value']
        else:  # noise
            errors = np.random.randn(500) * scenario['value']
        
        # Применяем фильтр
        loop_filter = LoopFilter(Kp=0.01, Ki=0.001, freq_limit=0.5)
        outputs = []
        for error in errors:
            output = loop_filter.update(error)
            outputs.append(output)
        
        # Визуализация
        ax.plot(errors, alpha=0.5, label='Входная ошибка', linewidth=1)
        ax.plot(outputs, alpha=0.8, label='Выход фильтра', linewidth=2)
        ax.set_title(scenario['title'])
        ax.set_xlabel('Отсчёт')
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('fll_filter_visualization.png', dpi=150, bbox_inches='tight')
    print("Сохранено: fll_filter_visualization.png\n")
    plt.show()


def visualize_nco():
    """Визуализирует работу NCO"""
    print("Визуализация NCO...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Разные частоты
    frequencies = [0.01, 0.05, 0.1]
    
    for col, freq in enumerate(frequencies):
        # Генерация NCO
        nco = NCO(initial_freq=0.0)
        nco.freq = freq
        
        output = []
        phases = []
        for _ in range(200):
            signal = nco.step()
            output.append(signal)
            phases.append(nco.phase)
        
        output = np.array(output)
        phases = np.array(phases)
        
        # График 1: Комплексный сигнал (I/Q)
        axes[0, col].plot(output.real, output.imag, 'o-', markersize=3, alpha=0.6)
        axes[0, col].set_title(f'NCO output (f={freq})')
        axes[0, col].set_xlabel('I (Real)')
        axes[0, col].set_ylabel('Q (Imag)')
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].axis('equal')
        
        # График 2: Фаза во времени
        axes[1, col].plot(phases, linewidth=2)
        axes[1, col].set_title(f'Фаза NCO (f={freq})')
        axes[1, col].set_xlabel('Отсчёт')
        axes[1, col].set_ylabel('Фаза (радианы)')
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].axhline(np.pi, color='r', linestyle='--', alpha=0.3)
        axes[1, col].axhline(-np.pi, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fll_nco_visualization.png', dpi=150, bbox_inches='tight')
    print("Сохранено: fll_nco_visualization.png\n")
    plt.show()


def visualize_full_loop_dynamics():
    """Визуализирует динамику полной петли с разными параметрами"""
    print("Визуализация динамики полной петли...")
    
    from fll import FrequencyLockedLoop
    
    # Создаём тестовый сигнал
    np.random.seed(42)
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 300)
    signal = np.repeat(symbols, 4)
    f_offset = 0.02
    n = np.arange(len(signal))
    signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * 0.1
    signal_with_offset += noise
    
    # Тестируем разные параметры
    configs = [
        {'Kp': 0.001, 'Ki': 0.00005, 'name': 'Медленная (Kp=0.001)'},
        {'Kp': 0.005, 'Ki': 0.0001, 'name': 'Быстрая (Kp=0.005)'},
        {'Kp': 0.01, 'Ki': 0.001, 'name': 'Очень быстрая (Kp=0.01)'},
        {'Kp': 0.0001, 'Ki': 0.000001, 'name': 'Очень медленная (Kp=0.0001)'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, config in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        
        fll = FrequencyLockedLoop(
            detector_method='cross_product',
            Kp=config['Kp'],
            Ki=config['Ki'],
            freq_limit=0.1
        )
        
        _, freq_estimate = fll.process_signal(signal_with_offset)
        
        # График сходимости частоты
        ax.plot(fll.history['freq_estimates'], linewidth=2, label='Оценка')
        ax.axhline(f_offset, color='r', linestyle='--', linewidth=2, 
                  label=f'Истинное значение ({f_offset})')
        ax.axhline(freq_estimate, color='g', linestyle='--', 
                  label=f'Финальная оценка ({freq_estimate:.6f})')
        
        ax.set_title(config['name'])
        ax.set_xlabel('Отсчёт')
        ax.set_ylabel('Частота')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([-0.05, 0.1])
    
    plt.tight_layout()
    plt.savefig('fll_loop_dynamics.png', dpi=150, bbox_inches='tight')
    print("Сохранено: fll_loop_dynamics.png\n")
    plt.show()


def visualize_snr_impact():
    """Визуализирует влияние SNR на работу FLL"""
    print("Визуализация влияния SNR...")
    
    from fll import FrequencyLockedLoop
    
    # Разные уровни SNR
    snr_levels = [20, 10, 5, 0]  # дБ
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, snr_db in enumerate(snr_levels):
        ax = axes[idx // 2, idx % 2]
        
        # Создаём сигнал с заданным SNR
        np.random.seed(42)
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 300)
        signal = np.repeat(symbols, 4)
        f_offset = 0.02
        n = np.arange(len(signal))
        signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
        
        # Добавляем шум с заданным SNR
        signal_power = np.mean(np.abs(signal_with_offset)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * np.sqrt(noise_power/2)
        signal_with_offset += noise
        
        # FLL
        fll = FrequencyLockedLoop(
            detector_method='cross_product',
            Kp=0.001,
            Ki=0.00005,
            freq_limit=0.1
        )
        
        _, freq_estimate = fll.process_signal(signal_with_offset)
        error = abs(freq_estimate - f_offset)
        
        # График
        ax.plot(fll.history['freq_estimates'], linewidth=2, alpha=0.8)
        ax.axhline(f_offset, color='r', linestyle='--', linewidth=2, 
                  label=f'Истинное ({f_offset})')
        ax.set_title(f'SNR = {snr_db} дБ (ошибка: {error:.6f})')
        ax.set_xlabel('Отсчёт')
        ax.set_ylabel('Частота')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([-0.05, 0.1])
    
    plt.tight_layout()
    plt.savefig('fll_snr_impact.png', dpi=150, bbox_inches='tight')
    print("Сохранено: fll_snr_impact.png\n")
    plt.show()


def create_summary_diagram():
    """Создаёт итоговую диаграмму сравнения методов"""
    print("Создание итоговой диаграммы...")
    
    from fll import FrequencyLockedLoop
    
    # Тестовый сигнал
    np.random.seed(42)
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 300)
    signal = np.repeat(symbols, 4)
    f_offset = 0.015
    n = np.arange(len(signal))
    signal_with_offset = signal * np.exp(1j * 2 * np.pi * f_offset * n)
    noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * 0.1
    signal_with_offset += noise
    
    # Три метода
    methods = ['cross_product', 'atan2', 'decision_directed']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    for row, method in enumerate(methods):
        fll = FrequencyLockedLoop(
            detector_method=method,
            Kp=0.005,
            Ki=0.0001,
            freq_limit=0.1
        )
        
        corrected, freq_estimate = fll.process_signal(signal_with_offset)
        error = abs(freq_estimate - f_offset)
        
        # Созвездие до
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(signal_with_offset[::4].real[:100], 
                signal_with_offset[::4].imag[:100], 
                'o', markersize=3, alpha=0.6)
        ax1.set_title(f'{method}\nДо FLL')
        ax1.set_xlabel('I')
        ax1.set_ylabel('Q')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Созвездие после
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(corrected[::4].real[:100], 
                corrected[::4].imag[:100], 
                'o', markersize=3, alpha=0.6)
        ax2.set_title(f'После FLL\n(оценка: {freq_estimate:.6f})')
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Сходимость
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.plot(fll.history['freq_estimates'], linewidth=2)
        ax3.axhline(f_offset, color='r', linestyle='--', 
                   label=f'Истинное ({f_offset})')
        ax3.set_title(f'Сходимость\n(ошибка: {error:.6f})')
        ax3.set_xlabel('Отсчёт')
        ax3.set_ylabel('Частота')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    plt.savefig('fll_methods_summary.png', dpi=150, bbox_inches='tight')
    print("Сохранено: fll_methods_summary.png\n")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ КОМПОНЕНТОВ FLL")
    print("=" * 70 + "\n")
    
    visualize_frequency_error_detector()
    visualize_loop_filter()
    visualize_nco()
    visualize_full_loop_dynamics()
    visualize_snr_impact()
    create_summary_diagram()
    
    print("=" * 70)
    print("ВСЕ ВИЗУАЛИЗАЦИИ ЗАВЕРШЕНЫ")
    print("=" * 70)
    print("\nСозданные файлы:")
    print("  - fll_detector_visualization.png")
    print("  - fll_filter_visualization.png")
    print("  - fll_nco_visualization.png")
    print("  - fll_loop_dynamics.png")
    print("  - fll_snr_impact.png")
    print("  - fll_methods_summary.png")

