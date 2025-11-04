import numpy as np
from scipy import signal as sp_signal

def generate_mpsk_signal(modulation_type='bpsk', Fs=100000, Fc=1000, Sps=10, num_symbols=100, 
                         use_pulse_shaping=True):
    """
    Генерация M-PSK модулированного сигнала
    
    Параметры:
    modulation_type : str
        Тип модуляции ('bpsk', 'qpsk', '8psk')
    Fs : int
        Частота дискретизации, Гц
    Fc : int
        Частота несущей, Гц
    Sps : int
        Количество отсчетов на символ
    num_symbols : int
        Количество символов для генерации
    use_pulse_shaping : bool, default=True
        True - создает импульсный сигнал (дельта-функции) для pulse shaping фильтров
        False - создает прямоугольные импульсы (np.repeat) для простой демонстрации
    
    Возвращает:
    dict : Словарь с результатами:
        't' : временная ось
        'signal' : модулированный сигнал
        'carrier' : несущий сигнал
        'symbols' : символы до расширения
        'symbols_expanded' : расширенные символы (импульсы или прямоугольники)
        'bits' : исходные биты
        'Sps' : количество отсчетов на символ
        'Fs' : частота дискретизации
        'Fc' : частота несущей
        'use_pulse_shaping' : режим формирования сигнала
    """
    # Определяем параметры в зависимости от типа модуляции
    modulation_params = {
        'bpsk': {'M': 2, 'bits_per_symbol': 1},
        'qpsk': {'M': 4, 'bits_per_symbol': 2},
        '8psk': {'M': 8, 'bits_per_symbol': 3}
    }
    
    if modulation_type.lower() not in modulation_params:
        raise ValueError(f"Неподдерживаемый тип модуляции. Поддерживаются: {list(modulation_params.keys())}")
    
    M = modulation_params[modulation_type.lower()]['M']
    bits_per_symbol = modulation_params[modulation_type.lower()]['bits_per_symbol']
    
    # Генерация случайных битов
    num_bits = num_symbols * bits_per_symbol
    bits = np.random.randint(0, 2, num_bits)
    
    # Создание временной шкалы
    Tb = Sps/Fs  # длительность символа
    t = np.arange(0, Tb*num_symbols, 1/Fs)
    
    # Преобразование битов в символы
    if modulation_type.lower() == 'bpsk':
        # BPSK: 0 -> -1, 1 -> 1
        symbols = 2*bits - 1
    else:
        # Группируем биты по bits_per_symbol
        bits_grouped = bits.reshape(-1, bits_per_symbol)
        # Преобразуем группы битов в десятичные числа
        decimal_values = np.zeros(num_symbols, dtype=int)
        for i in range(num_symbols):
            decimal_values[i] = int(''.join(map(str, bits_grouped[i])), 2)
        # Создаем символы с соответствующими фазами
        phases = 2 * np.pi * decimal_values / M
        symbols = np.exp(1j * phases)
    
    # Расширяем символы (выбор между импульсами и прямоугольниками)
    if use_pulse_shaping:
        # ПРАВИЛЬНЫЙ подход: импульсы (дельта-функции) для pulse shaping фильтров
        symbols_expanded = np.zeros(Sps * num_symbols, dtype=complex)
        for i in range(num_symbols):
            symbols_expanded[i * Sps] = symbols[i]
    else:
        # ПРОСТОЙ подход: прямоугольные импульсы для базовой демонстрации
        symbols_expanded = np.repeat(symbols, Sps)
    
    # Создаем несущий сигнал
    carrier = np.exp(2j * np.pi * Fc * t)
    
    # Модулируем несущий сигнал
    modulated_signal = carrier * symbols_expanded
    
    # Для реального сигнала берем действительную часть
    modulated_signal_real = np.real(modulated_signal)
    
    return {
        't': t,
        'signal': modulated_signal_real,
        'carrier': np.real(carrier),
        'symbols': symbols,
        'symbols_expanded': symbols_expanded,
        'bits': bits,
        'Sps': Sps,
        'Fs': Fs,
        'Fc': Fc,
        'use_pulse_shaping': use_pulse_shaping
    }

def resample_signal(input_signal, original_Sps, target_Sps, original_Fs=None, 
                   num_symbols=None, apply_filter=True):
    """
    Изменение SPS (Samples Per Symbol) сигнала с автоматической фильтрацией
    
    Параметры:
    input_signal : array_like или dict
        Входной сигнал (массив numpy) или словарь из generate_mpsk_signal
    original_Sps : int
        Исходное количество отсчетов на символ
    target_Sps : int или float
        Целевое количество отсчетов на символ
    original_Fs : float, optional
        Исходная частота дискретизации (нужна для расчета новой Fs)
    num_symbols : int, optional
        Количество символов в сигнале (если не указано, вычисляется автоматически)
    apply_filter : bool, default=True
        Применять ли anti-aliasing фильтр
    
    Возвращает:
    dict : Словарь с результатами:
        'signal' : передискретизированный сигнал
        'original_length' : длина исходного сигнала
        'new_length' : длина нового сигнала
        'original_Sps' : исходный SPS
        'target_Sps' : целевой SPS
        'resampling_ratio' : коэффициент передискретизации
        'Fs' : новая частота дискретизации (если original_Fs был указан)
        'num_symbols' : количество символов
    """
    
    # Если на вход пришел словарь из generate_mpsk_signal
    if isinstance(input_signal, dict):
        signal_array = input_signal['signal']
        if original_Fs is None and 'Fs' in input_signal:
            original_Fs = input_signal['Fs']
        if original_Sps != input_signal.get('Sps'):
            print(f"Внимание: original_Sps ({original_Sps}) не совпадает с SPS в словаре ({input_signal.get('Sps')})")
    else:
        signal_array = np.array(input_signal)
    
    # Вычисляем количество символов
    if num_symbols is None:
        num_symbols = len(signal_array) / original_Sps
        if not num_symbols.is_integer():
            print(f"Предупреждение: длина сигнала ({len(signal_array)}) не кратна original_Sps ({original_Sps})")
            print(f"Используется дробное количество символов: {num_symbols:.2f}")
    
    # Вычисляем параметры передискретизации
    resampling_ratio = target_Sps / original_Sps
    original_length = len(signal_array)
    new_length = int(np.round(original_length * resampling_ratio))
    
    # Определяем метод передискретизации
    if np.isclose(resampling_ratio, 1.0, rtol=1e-6):
        # SPS не меняется
        resampled_signal = signal_array.copy()
        method = "без изменений"
    
    elif resampling_ratio > 1:
        # Интерполяция (увеличение SPS)
        # Пытаемся найти целочисленные множители для resample_poly
        up, down = _find_rational_approximation(target_Sps, original_Sps)
        
        if apply_filter and up <= 100 and down <= 100:
            # Используем resample_poly (быстрее и эффективнее для целочисленных коэффициентов)
            resampled_signal = sp_signal.resample_poly(signal_array, up, down)
            method = f"resample_poly (up={up}, down={down})"
        else:
            # Используем универсальный resample
            resampled_signal = sp_signal.resample(signal_array, new_length)
            method = "resample (интерполяция)"
    
    else:
        # Децимация (уменьшение SPS)
        # Пытаемся найти целочисленные множители
        up, down = _find_rational_approximation(target_Sps, original_Sps)
        
        if apply_filter and down <= 100 and up <= 100:
            # Используем resample_poly с фильтрацией
            resampled_signal = sp_signal.resample_poly(signal_array, up, down)
            method = f"resample_poly (up={up}, down={down})"
        else:
            # Используем универсальный resample
            resampled_signal = sp_signal.resample(signal_array, new_length)
            method = "resample (децимация)"
    
    # Вычисляем новую частоту дискретизации
    new_Fs = None
    if original_Fs is not None:
        new_Fs = original_Fs * resampling_ratio
    
    return {
        'signal': resampled_signal,
        'original_length': original_length,
        'new_length': len(resampled_signal),
        'original_Sps': original_Sps,
        'target_Sps': target_Sps,
        'resampling_ratio': resampling_ratio,
        'Fs': new_Fs,
        'num_symbols': num_symbols,
        'method': method
    }

def _find_rational_approximation(numerator, denominator, max_value=100):
    """
    Находит рациональное приближение для дроби numerator/denominator
    с целью использования в resample_poly
    
    Возвращает (up, down) такие, что up/down ≈ numerator/denominator
    """
    from fractions import Fraction
    
    # Используем Fraction для точного представления
    frac = Fraction(numerator).limit_denominator(max_value) / Fraction(denominator).limit_denominator(max_value)
    frac = frac.limit_denominator(max_value)
    
    return frac.numerator, frac.denominator

def apply_rrc_filter(input_signal, Sps, alpha=0.35, filter_span=10, Fs=None):
    """
    Применяет фильтр Root Raised Cosine (RRC) к сигналу
    
    Параметры:
    input_signal : array_like или dict
        Входной сигнал (массив numpy) или словарь из generate_mpsk_signal
    Sps : int
        Количество отсчетов на символ
    alpha : float, default=0.35
        Коэффициент скругления (roll-off factor), 0 ≤ α ≤ 1
        α=0: прямоугольный спектр (узкая полоса, долгий импульс)
        α=1: широкий спектр (широкая полоса, короткий импульс)
        Типичные значения: 0.2-0.5
    filter_span : int, default=10
        Длина фильтра в символах (чем больше, тем лучше, но медленнее)
    Fs : float, optional
        Частота дискретизации (если не указана, берется из словаря)
    
    Возвращает:
    dict : Словарь с результатами:
        'signal' : отфильтрованный сигнал
        'filter' : импульсная характеристика фильтра
        'original_length' : длина исходного сигнала
        'filtered_length' : длина отфильтрованного сигнала
        'alpha' : использованный коэффициент скругления
        'filter_span' : длина фильтра в символах
    """
    
    # Если на вход пришел словарь из generate_mpsk_signal
    if isinstance(input_signal, dict):
        signal_array = input_signal['signal']
        if Fs is None and 'Fs' in input_signal:
            Fs = input_signal['Fs']
        if Sps != input_signal.get('Sps'):
            print(f"Внимание: Sps ({Sps}) не совпадает с SPS в словаре ({input_signal.get('Sps')})")
    else:
        signal_array = np.array(input_signal)
    
    # Длина фильтра в отсчетах (должна быть нечетной для симметрии)
    filter_length = filter_span * Sps
    if filter_length % 2 == 0:
        filter_length += 1
    
    # Временная ось для фильтра (центрированная, в символьных периодах)
    t = np.arange(-filter_length//2, filter_length//2 + 1) / Sps
    
    # Формула Root Raised Cosine фильтра
    rrc_filter = np.zeros(len(t))
    
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            # Особый случай: t = 0
            rrc_filter[i] = (1 - alpha + 4*alpha/np.pi)
        elif alpha > 0 and abs(abs(ti) - 1/(4*alpha)) < 1e-10:
            # Особый случай: t = ±1/(4α)
            rrc_filter[i] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) + 
                (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )
        else:
            # Общий случай
            numerator = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            denominator = np.pi*ti*(1 - (4*alpha*ti)**2)
            rrc_filter[i] = numerator / denominator
    
    # Нормализация фильтра (сумма квадратов = 1, энергия = 1)
    rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter**2))
    
    # Применяем фильтр (свертка)
    # mode='full' дает полный результат, затем обрезаем до исходной длины
    filtered_full = np.convolve(signal_array, rrc_filter, mode='full')
    
    # Вычисляем задержку фильтра (group delay)
    delay = len(rrc_filter) // 2
    
    # Обрезаем результат, учитывая задержку, чтобы длина совпадала с исходным сигналом
    filtered_signal = filtered_full[delay:delay + len(signal_array)]
    
    print(f"\nПрименен RRC фильтр:")
    print(f"  Alpha (roll-off): {alpha}")
    print(f"  Длина фильтра: {filter_span} символов ({len(rrc_filter)} отсчетов)")
    print(f"  Длина сигнала: {len(signal_array)} → {len(filtered_signal)} отсчетов")
    
    return {
        'signal': filtered_signal,
        'filter': rrc_filter,
        'original_length': len(signal_array),
        'filtered_length': len(filtered_signal),
        'alpha': alpha,
        'filter_span': filter_span,
        'Sps': Sps,
        'Fs': Fs
    }

def apply_rc_filter(input_signal, Sps, alpha=0.35, filter_span=10, Fs=None):
    """
    Применяет фильтр Raised Cosine (RC) к сигналу
    
    Параметры:
    input_signal : array_like или dict
        Входной сигнал (массив numpy) или словарь из generate_mpsk_signal
    Sps : int
        Количество отсчетов на символ
    alpha : float, default=0.35
        Коэффициент скругления (roll-off factor), 0 ≤ α ≤ 1
        α=0: прямоугольный спектр (узкая полоса, долгий импульс)
        α=1: широкий спектр (широкая полоса, короткий импульс)
    filter_span : int, default=10
        Длина фильтра в символах
    Fs : float, optional
        Частота дискретизации
    
    Возвращает:
    dict : Словарь с результатами:
        'signal' : отфильтрованный сигнал
        'filter' : импульсная характеристика фильтра
        'original_length' : длина исходного сигнала
        'filtered_length' : длина отфильтрованного сигнала
        'alpha' : использованный коэффициент скругления
        'filter_span' : длина фильтра в символах
    
    Примечание:
    RC = RRC × RRC (в частотной области)
    Raised Cosine - это квадрат Root Raised Cosine
    """
    
    # Если на вход пришел словарь из generate_mpsk_signal
    if isinstance(input_signal, dict):
        signal_array = input_signal['signal']
        if Fs is None and 'Fs' in input_signal:
            Fs = input_signal['Fs']
        if Sps != input_signal.get('Sps'):
            print(f"Внимание: Sps ({Sps}) не совпадает с SPS в словаре ({input_signal.get('Sps')})")
    else:
        signal_array = np.array(input_signal)
    
    # Длина фильтра в отсчетах (должна быть нечетной для симметрии)
    filter_length = filter_span * Sps
    if filter_length % 2 == 0:
        filter_length += 1
    
    # Временная ось для фильтра (центрированная, в символьных периодах)
    # t представляет время, нормированное на длительность символа
    t = np.arange(-filter_length//2, filter_length//2 + 1) / Sps
    
    # Формула Raised Cosine фильтра
    rc_filter = np.zeros(len(t))
    
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            # Особый случай: t = 0
            # В центре фильтр имеет максимальное значение
            rc_filter[i] = 1.0
        elif alpha > 0 and abs(abs(ti) - 1/(2*alpha)) < 1e-10:
            # Особый случай: t = ±1/(2α)
            rc_filter[i] = (np.pi/4) * np.sinc(1/(2*alpha))
        else:
            # Общий случай
            # sinc(t) = sin(πt)/(πt), но numpy.sinc уже использует sinc(x) = sin(πx)/(πx)
            rc_filter[i] = np.sinc(ti) * np.cos(np.pi*alpha*ti) / (1 - (2*alpha*ti)**2)
    
    # КРИТИЧЕСКИ ВАЖНО: нормализация для сохранения энергии символов
    # Сумма всех отсчетов фильтра, приходящихся на один символьный период, должна быть 1
    # Для этого берем сумму каждого Sps-го отсчета, начиная с центра
    sample_sum = 0
    center_idx = len(rc_filter) // 2
    for k in range(-filter_span//2, filter_span//2 + 1):
        idx = center_idx + k * Sps
        if 0 <= idx < len(rc_filter):
            sample_sum += rc_filter[idx]
    
    if abs(sample_sum) > 1e-10:
        rc_filter = rc_filter / sample_sum
    
    # Применяем фильтр (свертка)
    # mode='full' дает полный результат, затем обрезаем до исходной длины
    filtered_full = np.convolve(signal_array, rc_filter, mode='full')
    
    # Вычисляем задержку фильтра (group delay)
    delay = len(rc_filter) // 2
    
    # Обрезаем результат, учитывая задержку, чтобы длина совпадала с исходным сигналом
    filtered_signal = filtered_full[delay:delay + len(signal_array)]
    
    print(f"\nПрименен RC фильтр:")
    print(f"  Alpha (roll-off): {alpha}")
    print(f"  Длина фильтра: {filter_span} символов ({len(rc_filter)} отсчетов)")
    print(f"  Длина сигнала: {len(signal_array)} → {len(filtered_signal)} отсчетов")
    
    return {
        'signal': filtered_signal,
        'filter': rc_filter,
        'original_length': len(signal_array),
        'filtered_length': len(filtered_signal),
        'alpha': alpha,
        'filter_span': filter_span,
        'Sps': Sps,
        'Fs': Fs
    }



# Визуализация результатов
if __name__ == "__main__":

    modulation_types = ['bpsk', 'qpsk', '8psk']
    signals = {}

    for mod_type in modulation_types:
        signals[mod_type] = generate_mpsk_signal(
            modulation_type=mod_type,
            Fs=100000,
            Fc=1000,
            Sps=100,
            num_symbols=100,
            use_pulse_shaping=False  # Прямоугольные импульсы для наглядной визуализации
        )
        print(f"\n{mod_type.upper()}:")
        print(f"Биты: {signals[mod_type]['bits']}")
        print(f"Количество символов: {len(signals[mod_type]['symbols'])}")
        print(f"Количество отсчетов: {len(signals[mod_type]['signal'])}")
    import matplotlib.pyplot as plt
    
    # Создаем одно окно с 3 рядами и 3 колонками (9 графиков)
    fig = plt.figure(figsize=(18, 12))
    
    for idx, mod_type in enumerate(modulation_types):
        signal_data = signals[mod_type]
        
        # Показываем только первые 2 символа для наглядности
        samples_to_show = 2 * signal_data['Sps']
        t_plot = signal_data['t'][:samples_to_show]
        
        # Левая колонка: Временные графики сигналов
        plt.subplot(3, 3, idx * 3 + 1)
        plt.plot(t_plot, signal_data['signal'][:samples_to_show], 'b-', linewidth=1.5)
        plt.title(f'{mod_type.upper()} сигнал (2 символа)')
        plt.grid(True, alpha=0.3)
        plt.ylabel('Амплитуда')
        if idx == len(modulation_types) - 1:
            plt.xlabel('Время, с')
        
        # Средняя колонка: Спектр сигнала (FFT)
        plt.subplot(3, 3, idx * 3 + 2)
        # Вычисляем FFT для визуализации спектра
        signal_fft = np.fft.fft(signal_data['signal'])
        freqs = np.fft.fftfreq(len(signal_data['signal']), 1/signal_data['Fs'])
        # Показываем только положительные частоты
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(signal_fft[:len(freqs)//2])
        plt.plot(positive_freqs, magnitude, 'g-', linewidth=1.5)
        plt.title(f'{mod_type.upper()} спектр')
        plt.grid(True, alpha=0.3)
        plt.ylabel('Магнитуда')
        plt.xlim(0, 5000)  # Показываем до 5 кГц
        if idx == len(modulation_types) - 1:
            plt.xlabel('Частота, Гц')
        
        # Правая колонка: Диаграммы созвездий
        plt.subplot(3, 3, idx * 3 + 3)
        if mod_type == 'bpsk':
            symbols_complex = signal_data['symbols'].astype(complex)
        else:
            symbols_complex = signal_data['symbols']
        
        plt.scatter(np.real(symbols_complex), np.imag(symbols_complex), 
                   s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
        plt.title(f'{mod_type.upper()} созвездие')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Re (I)')
        plt.ylabel('Im (Q)')
        plt.axis('equal')
        # Устанавливаем одинаковые пределы для всех созвездий
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
    
    plt.suptitle('M-PSK Модуляция: Сигнал, Спектр и Созвездие', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()