import numpy as np
# from numpy.fft import fft
import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
import matplotlib.pyplot as plt
from gardner2 import gardner_timing_recovery

def rrc_filter(sps, span, alpha):
    """
    Создание Root Raised Cosine фильтра
    
    Параметры:
    sps - samples per symbol (отсчетов на символ)
    span - длина фильтра в символах
    alpha - коэффициент скругления (roll-off factor)
    
    Возвращает:
    h - импульсная характеристика фильтра
    """
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    h = np.zeros(len(n))
    
    for i, t in enumerate(n):
        if t == 0:
            h[i] = (1 + alpha * (4 / np.pi - 1))
        elif abs(t) == sps / (4 * alpha):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            numerator = np.sin(np.pi * t / sps * (1 - alpha)) + \
                       4 * alpha * t / sps * np.cos(np.pi * t / sps * (1 + alpha))
            denominator = np.pi * t / sps * (1 - (4 * alpha * t / sps) ** 2)
            h[i] = numerator / denominator
    
    # Нормализация
    h = h / np.sqrt(np.sum(h ** 2))
    return h

def find_optimal_sampling_phase(signal_filtered, preamble_symbols, sps, preamble_length=None):
    """
    Поиск оптимальной фазы семплирования (от 0 до sps-1) путем корреляции с преамбулой.
    
    Параметры:
    -----------
    signal_filtered : array
        Отфильтрованный сигнал (после RRC)
    preamble_symbols : array
        Известные символы преамбулы
    sps : int
        Количество отсчетов на символ
    preamble_length : int, optional
        Длина преамбулы для анализа (по умолчанию вся преамбула)
    
    Возвращает:
    -----------
    best_phase : int
        Оптимальная фаза семплирования (0..sps-1)
    correlations : array
        Значения корреляции для каждой фазы
    """
    if preamble_length is None:
        preamble_length = len(preamble_symbols)
    else:
        preamble_length = min(preamble_length, len(preamble_symbols))
    
    preamble_ref = preamble_symbols[:preamble_length]
    correlations = np.zeros(sps)
    
    # Перебираем все возможные фазы от 0 до sps-1
    for phase in range(sps):
        # Децимация с текущей фазой
        sampled = signal_filtered[phase::sps]
        
        # Берем участок длиной с преамбулу
        if len(sampled) < preamble_length:
            correlations[phase] = 0
            continue
            
        sampled_preamble = sampled[:preamble_length]
        
        # Вычисляем корреляцию (нормализованную)
        correlation = np.abs(np.sum(sampled_preamble * np.conj(preamble_ref)))
        correlations[phase] = correlation
    
    best_phase = np.argmax(correlations)
    
    return best_phase, correlations


def find_preamble_offset(signal_iq, preamble_iq, sps):
    """
    Поиск начала преамбулы в сигнале методом дифференциальной корреляции.
    
    Параметры:
    ----------
    signal_iq : np.ndarray
        Входной сигнал (комплексный IQ)
    preamble_iq : np.ndarray
        Известная преамбула (комплексные символы)
    sps : int
        Samples per symbol (количество отсчетов на символ)
    
    Возвращает:
    -----------
    offset : int
        Смещение в символах, где найдена преамбула
    signal_aligned : np.ndarray
        Сигнал, обрезанный с найденного offset
    conv_results : list
        Результаты корреляции для каждой фазы (для отладки/визуализации)
    conv_max : list
        Позиции максимумов корреляции для каждой фазы
    
    Пример использования:
    --------------------
    >>> offset, aligned_signal, corr_results, corr_max = find_preamble_offset(
    ...     signal_iq, preamble_iq, sps=4
    ... )
    >>> print(f"Преамбула найдена на позиции {offset} символов")
    """
    # Дифференциальное произведение для преамбулы
    d_pre = preamble_iq[1:] * np.conj(preamble_iq[:-1])
    
    # Дифференциальная корреляция для каждой фазы
    d_sig_list = []
    for phase in range(sps):
        # Берем символы с шагом sps, начиная с фазы phase
        sig_symbols = signal_iq[phase::sps]
        d_sig = sig_symbols[1:] * np.conj(sig_symbols[:-1])
        d_sig_list.append(d_sig)
    
    # Корреляция с нормализацией для каждой фазы
    conv_results = []
    conv_max = []
    for phase in range(sps):
        signal = d_sig_list[phase]
        conv = np.zeros(len(signal), dtype=np.complex64)
        
        # Скользящая корреляция с нормализацией на СКО
        for k in range(len(signal) - len(d_pre) + 1):
            curr_signal = signal[k:k + len(d_pre)]
            Y = np.std(curr_signal)
            if Y > 0:  # Защита от деления на ноль
                conv[k] = np.sum(curr_signal * np.conj(d_pre)) / Y
            else:
                conv[k] = 0
        
        conv_results.append(conv)
        conv_max.append(np.argmax(np.abs(conv)))
    
    # Находим медиану максимумов по всем фазам
    offset = int(np.median(conv_max))
    
    # Обрезаем сигнал с найденного смещения
    signal_aligned = signal_iq[offset * sps:]
    
    return offset, signal_aligned, conv_results, conv_max

if __name__ == "__main__":
    signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
    signal_iq = signal[::2] + 1j * signal[1::2]
    sps = 4

    preamble_data = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
    preamble_iq = preamble_data[::2] + 1j * preamble_data[1::2]

    # Поиск преамбулы и выравнивание сигнала
    offset, signal_iq, conv_results, conv_max = find_preamble_offset(signal_iq, preamble_iq, sps)
    print(conv_results)
    print(conv_max)
    print(offset)

    print(f"Преамбула найдена на позиции: {offset} символов")
    print(f"Длина выровненного сигнала: {len(signal_iq)}")

    # f_rel = 0.0164284
    # n = np.arange(signal_iq.size, dtype=np.float32)
    # signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)

    sps = 4  # samples per symbol
    span = 10  # длина фильтра в символах
    alpha = 0.35  # roll-off factor

    rrc = rrc_filter(sps, span, alpha)
    signal_filtered = np.convolve(signal_iq, rrc, mode='same')
    signal_filtered = signal_filtered / np.std(signal_filtered)
    

    signal_gardner, errors, mu_history = gardner_timing_recovery(signal_filtered, sps, alpha=0.04)

    print(f"Длина signal_filtered: {len(signal_filtered)}")
    print(f"Тип данных: {signal_filtered.dtype}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Созвездия для каждой фазы (sps=4)', fontsize=14)

    for i in range(sps):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        signal_iq_symbols = signal_filtered[i::sps]
        print(f"Фаза {i}: количество символов = {len(signal_iq_symbols)}")
        
        ax.plot(signal_iq_symbols.real, signal_iq_symbols.imag, 'o', markersize=2, alpha=0.5)
        ax.set_title(f'Фаза {i} (N={len(signal_iq_symbols)})')
        ax.set_xlabel('I (Real)')
        ax.set_ylabel('Q (Imag)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()


    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Результаты корреляции для каждой фазы (sps=4)', fontsize=14)

    for idx in range(sps):
        row = idx // 2
        col = idx % 2
        ax = axes2[row, col]
        
        # Строим абсолютное значение корреляции
        ax.plot(np.abs(conv_results[idx]))
        ax.set_title(f'Фаза {idx}')
        ax.set_xlabel('Индекс')
        ax.set_ylabel('|Корреляция|')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()



    print(f"После Гарднера: {len(signal_gardner)} символов")

    fig_const, axes_const = plt.subplots(2, 2, figsize=(12, 10))
    fig_const.suptitle(f'Созвездие после Гарднера ({len(signal_gardner)} символов)', fontsize=14)

    # После Gardner у нас уже downsampled сигнал (по 1 отсчету на символ)
    # Поэтому просто строим весь сигнал на одном графике
    ax = axes_const[0, 0]
    ax.plot(signal_gardner.real, signal_gardner.imag, 'o', markersize=3, alpha=0.6)
    ax.set_title('Все символы')
    ax.set_xlabel('I (Real)')
    ax.set_ylabel('Q (Imag)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # График ошибок синхронизации
    ax = axes_const[0, 1]
    ax.plot(errors)
    ax.set_title('Ошибки синхронизации')
    ax.set_xlabel('Символ')
    ax.set_ylabel('Ошибка')
    ax.grid(True, alpha=0.3)

    # График истории mu
    ax = axes_const[1, 0]
    ax.plot(mu_history)
    ax.set_title('История mu (дробная задержка)')
    ax.set_xlabel('Символ')
    ax.set_ylabel('mu')
    ax.grid(True, alpha=0.3)

    # Скрыть неиспользуемый график
    axes_const[1, 1].axis('off')

    plt.tight_layout()

    # Показать все графики одновременно
    plt.show()
    print(mu_history[-1])