"""
Применение FLL к реальному сигналу из файла
Использует данные QPSK с частотным сдвигом
"""

import numpy as np
import matplotlib.pyplot as plt
from fll import FrequencyLockedLoop
from diff_method import rrc_filter, find_preamble_offset


def apply_fll_to_real_signal():
    """
    Применяет FLL к реальному QPSK сигналу
    """
    print("=" * 70)
    print("ПРИМЕНЕНИЕ FLL К РЕАЛЬНОМУ СИГНАЛУ")
    print("=" * 70)
    
    # Загружаем сигнал
    print("\n1. Загрузка сигнала...")
    signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
    signal_iq = signal[::2] + 1j * signal[1::2]
    print(f"   Длина сигнала: {len(signal_iq)} отсчетов")
    
    # Загружаем преамбулу
    preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
    preamble_iq = preamble[::2] + 1j * preamble[1::2]
    print(f"   Длина преамбулы: {len(preamble_iq)} символов")
    
    # Применяем RRC фильтр
    print("\n2. Применение RRC фильтра...")
    rrc = rrc_filter(sps=4, span=10, alpha=0.35)
    signal_filtered = np.convolve(signal_iq, rrc, mode='same')
    signal_filtered = signal_filtered / np.std(signal_filtered)
    
    # Находим преамбулу и выравниваем фазу
    print("\n3. Поиск преамбулы и выравнивание фазы...")
    offset, signal_aligned, conv_results, conv_max, phase_offset = \
        find_preamble_offset(signal_filtered, preamble_iq, sps=4)
    print(f"   Смещение преамбулы: {offset}")
    print(f"   Фазовый сдвиг: {phase_offset:.4f} радиан")
    
    # Визуализация ДО FLL
    print("\n4. Созвездие ДО коррекции частоты...")
    signal_decimated = signal_aligned[::4]  # Децимация до символьной частоты
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(signal_decimated[:500].real, signal_decimated[:500].imag, 
             'o', markersize=3, alpha=0.6, label='Сигнал')
    plt.title('Созвездие ДО FLL\n(видно вращение из-за частотного сдвига)')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imag)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    # =====================================================================
    # ПРИМЕНЕНИЕ FLL
    # =====================================================================
    print("\n5. Применение FLL...")
    
    # Создаем FLL с оптимальными параметрами
    fll = FrequencyLockedLoop(
        detector_method='decision_directed',  # Лучший метод для высокого SNR
        Kp=0.001,   # Пропорциональный коэффициент (небольшой для плавности)
        Ki=0.00005, # Интегральный коэффициент (еще меньше)
        freq_limit=0.05  # Ограничение частоты
    )
    
    # Обрабатываем сигнал
    signal_corrected, freq_estimate = fll.process_signal(signal_aligned)
    
    print(f"   Оценка частотного сдвига: {freq_estimate:.8f}")
    print(f"   В герцах (относительно частоты символов): {freq_estimate * 100:.6f}%")
    
    # Децимируем скорректированный сигнал
    signal_corrected_decimated = signal_corrected[::4]
    
    # Визуализация ПОСЛЕ FLL
    plt.subplot(1, 2, 2)
    plt.plot(signal_corrected_decimated[:500].real, 
             signal_corrected_decimated[:500].imag, 
             'o', markersize=3, alpha=0.6, label='Скорректированный')
    plt.title(f'Созвездие ПОСЛЕ FLL\n(f_rel = {freq_estimate:.8f})')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imag)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fll_constellation_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   Сохранено: fll_constellation_comparison.png")
    plt.show()
    
    # =====================================================================
    # АНАЛИЗ РАБОТЫ FLL
    # =====================================================================
    print("\n6. Анализ сходимости FLL...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # График 1: Ошибка частоты
    axes[0].plot(fll.history['errors'], alpha=0.7)
    axes[0].set_title('Ошибка частоты во времени', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Номер отсчета')
    axes[0].set_ylabel('Ошибка (радианы)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='r', linestyle='--', alpha=0.5)
    
    # График 2: Оценка частоты NCO
    axes[1].plot(fll.history['freq_estimates'], linewidth=2)
    axes[1].set_title('Сходимость оценки частоты', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Номер отсчета')
    axes[1].set_ylabel('Частота (относительная)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(freq_estimate, color='r', linestyle='--', 
                    label=f'Финальная оценка: {freq_estimate:.8f}')
    axes[1].legend()
    
    # График 3: Фаза NCO
    axes[2].plot(fll.history['phases'], alpha=0.7, linewidth=1)
    axes[2].set_title('Фаза NCO (для коррекции)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Номер отсчета')
    axes[2].set_ylabel('Фаза (радианы)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fll_performance.png', dpi=150, bbox_inches='tight')
    print("   Сохранено: fll_performance.png")
    plt.show()
    
    # =====================================================================
    # СРАВНЕНИЕ С МЕТОДАМИ ИЗ f_rel.py
    # =====================================================================
    print("\n7. Сравнение с другими методами оценки частоты...")
    
    # Используем только часть сигнала с преамбулой для честного сравнения
    signal_for_comparison = signal_decimated[:len(preamble_iq)]
    phase_signal = signal_for_comparison * np.conj(preamble_iq)
    
    # Метод 1: Усреднение разностей фаз
    phases = np.angle(phase_signal)
    phase_diffs = np.diff(np.unwrap(phases))
    f_rel_method1 = np.mean(phase_diffs) / (2 * np.pi * 4)
    
    # Метод 2: Линейная регрессия
    n = np.arange(len(phases))
    unwrapped_phases = np.unwrap(phases)
    slope = np.polyfit(n, unwrapped_phases, 1)[0]
    f_rel_method2 = slope / (2 * np.pi * 4)
    
    # Метод 3: Произведение соседних отсчетов
    prod = phase_signal[1:] * np.conj(phase_signal[:-1])
    avg_rotation = np.angle(np.mean(prod))
    f_rel_method3 = avg_rotation / (2 * np.pi * 4)
    
    print(f"\n   Метод 1 (усреднение разностей фаз): {f_rel_method1:.8f}")
    print(f"   Метод 2 (линейная регрессия):       {f_rel_method2:.8f}")
    print(f"   Метод 3 (произведение соседних):    {f_rel_method3:.8f}")
    print(f"   FLL (адаптивный):                   {freq_estimate:.8f}")
    
    print("\n" + "=" * 70)
    print("ПРЕИМУЩЕСТВА FLL:")
    print("=" * 70)
    print("✓ Работает БЕЗ знания преамбулы (слепой метод)")
    print("✓ Адаптируется к изменениям частоты в реальном времени")
    print("✓ Отслеживает дрейф частоты во время приема")
    print("✓ Можно использовать на всем сигнале, не только на преамбуле")
    print("=" * 70)
    
    return signal_corrected, freq_estimate, fll


def compare_fll_methods():
    """
    Сравнивает разные методы детектирования в FLL
    """
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МЕТОДОВ ДЕТЕКТИРОВАНИЯ В FLL")
    print("=" * 70)
    
    # Загружаем и подготавливаем сигнал (как выше)
    signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
    signal_iq = signal[::2] + 1j * signal[1::2]
    
    preamble = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
    preamble_iq = preamble[::2] + 1j * preamble[1::2]
    
    rrc = rrc_filter(sps=4, span=10, alpha=0.35)
    signal_filtered = np.convolve(signal_iq, rrc, mode='same')
    signal_filtered = signal_filtered / np.std(signal_filtered)
    
    offset, signal_aligned, _, _, _ = \
        find_preamble_offset(signal_filtered, preamble_iq, sps=4)
    
    # Тестируем все методы
    methods = ['cross_product', 'atan2', 'decision_directed']
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, method in enumerate(methods):
        print(f"\nТестирование метода: {method}")
        
        fll = FrequencyLockedLoop(
            detector_method=method,
            Kp=0.001,
            Ki=0.00005,
            freq_limit=0.05
        )
        
        signal_corrected, freq_estimate = fll.process_signal(signal_aligned)
        signal_corrected_decimated = signal_corrected[::4]
        
        results[method] = {
            'freq': freq_estimate,
            'signal': signal_corrected_decimated
        }
        
        print(f"   Оценка частоты: {freq_estimate:.8f}")
        
        # Визуализация
        axes[idx].plot(signal_corrected_decimated[:500].real,
                      signal_corrected_decimated[:500].imag,
                      'o', markersize=3, alpha=0.6)
        axes[idx].set_title(f'{method}\nf = {freq_estimate:.8f}')
        axes[idx].set_xlabel('I')
        axes[idx].set_ylabel('Q')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')
    
    plt.tight_layout()
    plt.savefig('fll_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   Сохранено: fll_methods_comparison.png")
    plt.show()
    
    return results


if __name__ == "__main__":
    # Основное применение FLL
    signal_corrected, freq_estimate, fll = apply_fll_to_real_signal()
    
    # Сравнение методов
    results = compare_fll_methods()
    
    print("\n" + "=" * 70)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 70)

