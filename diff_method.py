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
    # """
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
        conv_max.append((np.argmax(np.abs(conv)),np.max(np.abs(conv))))
    
    # Находим фазу с максимальным значением корреляции
    values = [val for idx, val in conv_max]
    offset_index = int(np.argmax(values))
    offset = conv_max[offset_index][0]
    phase_offset = offset_index
    # Обрезаем сигнал с найденного смещения
    signal_aligned = signal_iq[offset * sps + phase_offset:]
    
    return offset, signal_aligned, conv_results, conv_max, phase_offset

if __name__ == "__main__":
    signal = np.fromfile('files/sig_symb_x4_ncr_77671952566_logon_id_1_tx_id_7616.pcm', dtype=np.float32)
    signal_iq = signal[::2] + 1j * signal[1::2]
    sps = 4

    preamble_data = np.fromfile('files/preambule_logon_id_2_tx_id_7616_float32.pcm', dtype=np.float32)
    preamble_iq = preamble_data[::2] + 1j * preamble_data[1::2]

    signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * 0.0105 * np.arange(len(signal_iq)))
    # Добавление ФНЧ (низкочастотного фильтра) к сигналу

    from scipy.signal import firwin, lfilter

    fs = 1.0   # частота дискретизации (нормирована, тк частоты в спектре нормированы)
    cutoff_hz = 0.14  # допустим, пропускаем до F/Fd = 0.15
    numtaps = 101     # число коэффициентов ФНЧ (чем больше, тем круче фильтр, например, 51-201)

    # Расчет коэффициентов ФНЧ
    lpf_coeff = firwin(numtaps, cutoff=cutoff_hz, window='hamming', fs=fs)

    # Применение фильтра к сигналу
    signal_iq = lfilter(lpf_coeff, 1.0, signal_iq)
    signal_iq = signal_iq/np.std(signal_iq)


    # Поиск преамбулы и выравнивание сигнала
    offset, signal_iq, conv_results, conv_max , phase_offset= find_preamble_offset(signal_iq, preamble_iq, sps)
    print('--------------------------------')
    print(offset)
    print(phase_offset)
    print(conv_results)
    print(conv_max)
    print(offset)
    print('--------------------------------')

    print(f"Преамбула найдена на позиции: {offset} символов")
    print(f"Длина выровненного сигнала: {len(signal_iq)}")

    # f_rel = 0.0164284
    # n = np.arange(signal_iq.size, dtype=np.float32)
    # signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)

    sps = 4  # samples per symbol
    span = 10  # длина фильтра в символах
    alpha = 0.35  # roll-off factor

    # rrc = rrc_filter(sps, span, alpha)
    # signal_filtered = np.convolve(signal_iq, rrc, mode='same')
    # signal_filtered = signal_filtered / np.std(signal_filtered)
    
    signal_filtered = signal_iq

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Созвездия для каждой фазы (sps=4)', fontsize=14)

    for i in range(sps):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        signal_iq_symbols = signal_filtered[i::sps]
        print(f"Фаза {i}: количество символов = {len(signal_iq_symbols)}")
        
        ax.plot(signal_iq_symbols[:-400].real, signal_iq_symbols[:-400].imag, 'o', markersize=2, alpha=0.5)
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
    plt.show()


    signal_for_f_rel = signal_filtered[::4]
    # plt.figure(figsize=(10, 10))
    # plt.plot(signal_for_f_rel.real, signal_for_f_rel.imag, 'o', markersize=3, alpha=0.6)
    # plt.title(f'Созвездие \n(f_rel = )')
    # plt.xlabel('I (Real)')
    # plt.ylabel('Q (Imag)')
    # plt.grid(True, alpha=0.3)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()

    signal_for_f_rel = signal_for_f_rel[:len(preamble_iq)]
    phase_signal = signal_for_f_rel * np.conj(preamble_iq)
    phases = np.angle(phase_signal)
    phase_diffs = np.diff((np.unwrap(phases)))
    print(f"phase_diffs: {phase_diffs}")
    avg_phase_diff = np.mean(phase_diffs)
    print(f"avg_phase_diff: {avg_phase_diff}")
    f_rel_method1 = avg_phase_diff / (2 * np.pi * 4)
    print(f"f_rel_method ----: {f_rel_method1}")

        # 4. МЕТОД 2: Линейная регрессия (более точный)
    n = np.arange(len(phases))
    unwrapped_phases = np.unwrap(phases)

    # Наклон прямой φ(n) = 2π·f_rel·n + φ₀
    # Используем МНК: slope = Σ(n·φ) / Σ(n²)
    slope = np.polyfit(n, unwrapped_phases, 1)[0]  # коэффициент при n
    f_rel_method2 = slope / (2 * np.pi*4)
    print(f_rel_method2)

    # 5. МЕТОД 3: Через произведение соседних отсчетов (без unwrap)
    # Более устойчив к циклическим скачкам фазы
    prod = phase_signal[1:] * np.conj(phase_signal[:-1])
    avg_rotation = np.angle(np.mean(prod))
    f_rel_method3 = avg_rotation / (2 * np.pi*4)
    print(f_rel_method3)


    for i in [f_rel_method1, f_rel_method2, f_rel_method3]:
        signal = signal_iq.copy()
        n = np.arange(len(signal))
        signal = signal * np.exp(-1j * 2 * np.pi * i * n)
        rrc = rrc_filter(sps, span, alpha)
        signal_filtered = np.convolve(signal, rrc, mode='same')
        signal_filtered = signal_filtered / np.std(signal_filtered)
        # signal_filtered = signal

        signal = signal_filtered[0::4]

        from fll_func import fll_func
        signal_list, curr_freq, curr_freq_list, ef_n_list = fll_func(signal, 0.000001, 1)

    
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Созвездие после коррекции (signal_list)
        axs[0].plot(signal_list.real, signal_list.imag, 'o', markersize=3, alpha=0.6)
        axs[0].set_title(f'Созвездие после коррекции\n(f_rel = {i:.6f})')
        axs[0].set_xlabel('I (Real)')
        axs[0].set_ylabel('Q (Imag)')
        axs[0].grid(True, alpha=0.3)
        axs[0].axis('equal')

        # Текущая накопленная оценка частоты
        axs[1].plot(curr_freq_list)  # или просто значением, если curr_freq скаляр
        axs[1].set_title("curr_freq")
        axs[1].set_xlabel('n')
        axs[1].set_ylabel('curr_freq')

        # ef_n_list по времени
        axs[2].plot(ef_n_list)
        axs[2].set_title("Эфф. изменение частоты (ef_n_list)")
        axs[2].set_xlabel('n')
        axs[2].set_ylabel('ef_n')

        plt.tight_layout()
        plt.show()
    



