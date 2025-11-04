import numpy as np
import matplotlib.pyplot as plt
from gardner import gardner_ted


def generate_qpsk_symbols(n_symbols):
    """
    Генерирует случайные QPSK символы.
    
    Параметры:
    n_symbols - количество символов
    
    Возвращает:
    symbols - массив комплексных QPSK символов
    """
    # QPSK: 4 возможные точки созвездия
    qpsk_constellation = np.array([
        1 + 1j,   # 00
        -1 + 1j,  # 01
        1 - 1j,   # 10
        -1 - 1j   # 11
    ]) / np.sqrt(2)  # нормализация по энергии
    
    # Случайный выбор символов
    indices = np.random.randint(0, 4, size=n_symbols)
    symbols = qpsk_constellation[indices]
    
    return symbols


def upsample_signal(symbols, sps, pulse_shape='rect'):
    """
    Повышает частоту дискретизации сигнала (upsampling).
    
    Параметры:
    symbols - массив комплексных символов
    sps - samples per symbol (количество отсчетов на символ)
    pulse_shape - форма импульса ('rect' или 'rrc')
    
    Возвращает:
    signal - сигнал с повышенной частотой дискретизации
    """
    n_symbols = len(symbols)
    
    if pulse_shape == 'rect':
        # Прямоугольный импульс (простейший случай)
        signal = np.zeros(n_symbols * sps, dtype=np.complex128)
        for i, sym in enumerate(symbols):
            signal[i * sps:(i + 1) * sps] = sym
            
    elif pulse_shape == 'rrc':
        # Root Raised Cosine фильтр (более реалистичный)
        # Создаем RRC фильтр
        span = 10  # длина фильтра в символах
        alpha = 0.35  # roll-off factor
        
        # Временная ось для фильтра
        t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
        
        # Формула RRC
        h = np.zeros_like(t, dtype=np.float64)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = (1 + alpha * (4 / np.pi - 1))
            elif abs(abs(ti) - 1 / (4 * alpha)) < 1e-10:
                h[i] = alpha / np.sqrt(2) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                numerator = np.sin(np.pi * ti * (1 - alpha)) + \
                           4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
                denominator = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
                h[i] = numerator / denominator
        
        # Нормализация
        h = h / np.sqrt(np.sum(h ** 2))
        
        # Upsampling с нулями
        upsampled = np.zeros(n_symbols * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        
        # Применяем RRC фильтр
        signal = np.convolve(upsampled, h, mode='same')
    
    else:
        raise ValueError(f"Неизвестная форма импульса: {pulse_shape}")
    
    return signal


def add_timing_offset(signal, offset_samples, method='linear'):
    """
    Вносит временной сдвиг в сигнал (симуляция ошибки СТС).
    
    Параметры:
    signal - комплексный сигнал
    offset_samples - сдвиг во времени в отсчетах (может быть дробным)
    method - метод интерполяции ('linear', 'cubic', 'sinc')
    
    Возвращает:
    shifted_signal - сигнал со сдвигом
    """
    n = len(signal)
    
    if method == 'linear':
        # Линейная интерполяция
        shifted = np.zeros_like(signal)
        for i in range(n):
            src_idx = i - offset_samples
            
            if src_idx < 0 or src_idx >= n - 1:
                shifted[i] = 0
                continue
            
            idx_low = int(np.floor(src_idx))
            idx_high = idx_low + 1
            
            if idx_high >= n:
                shifted[i] = signal[idx_low]
            else:
                frac = src_idx - idx_low
                shifted[i] = signal[idx_low] * (1 - frac) + signal[idx_high] * frac
                
    elif method == 'cubic':
        # Кубическая интерполяция
        from scipy.interpolate import interp1d
        
        x_original = np.arange(n)
        x_shifted = np.arange(n) - offset_samples
        
        # Обрезаем края для корректной интерполяции
        valid_mask = (x_shifted >= 0) & (x_shifted < n - 1)
        
        # Разделяем на I и Q для интерполяции
        f_real = interp1d(x_original, np.real(signal), kind='cubic', 
                         bounds_error=False, fill_value=0)
        f_imag = interp1d(x_original, np.imag(signal), kind='cubic', 
                         bounds_error=False, fill_value=0)
        
        shifted = np.zeros_like(signal)
        shifted[valid_mask] = f_real(x_shifted[valid_mask]) + 1j * f_imag(x_shifted[valid_mask])
        
    elif method == 'sinc':
        # Sinc интерполяция (идеальная)
        shifted = np.zeros_like(signal)
        for i in range(n):
            src_idx = i - offset_samples
            
            # Берем окрестность для sinc интерполяции
            window_size = 32
            start = max(0, int(src_idx) - window_size // 2)
            end = min(n, int(src_idx) + window_size // 2)
            
            for k in range(start, end):
                if abs(src_idx - k) < 1e-10:
                    shifted[i] += signal[k]
                else:
                    sinc_val = np.sin(np.pi * (src_idx - k)) / (np.pi * (src_idx - k))
                    shifted[i] += signal[k] * sinc_val
    
    else:
        raise ValueError(f"Неизвестный метод интерполяции: {method}")
    
    return shifted


def add_frequency_offset(signal, f_offset, sample_rate=1.0):
    """
    Вносит частотный сдвиг в сигнал.
    
    Параметры:
    signal - комплексный сигнал
    f_offset - частотный сдвиг (нормализованный к частоте дискретизации)
    sample_rate - частота дискретизации (для справки, по умолчанию нормализована к 1.0)
    
    Возвращает:
    signal_with_offset - сигнал с частотным сдвигом
    """
    n = np.arange(len(signal))
    return signal * np.exp(1j * 2 * np.pi * f_offset * n)


def add_noise(signal, snr_db):
    """
    Добавляет белый гауссовский шум к сигналу.
    
    Параметры:
    signal - комплексный сигнал
    snr_db - отношение сигнал/шум в дБ
    
    Возвращает:
    noisy_signal - сигнал с шумом
    """
    # Вычисляем мощность сигнала
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Вычисляем требуемую мощность шума
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Генерируем комплексный гауссовский шум
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    
    return signal + noise


# =============== ТЕСТИРОВАНИЕ ===============

if __name__ == "__main__":
    # Параметры сигнала
    n_symbols = 1000
    sps = 4  # samples per symbol
    
    print(f"Генерация QPSK сигнала: {n_symbols} символов, {sps} отсчетов на символ")
    
    # 1. Генерируем QPSK символы
    symbols = generate_qpsk_symbols(n_symbols)
    print(f"Сгенерировано {len(symbols)} QPSK символов")
    
    # 2. Повышаем частоту дискретизации
    signal = upsample_signal(symbols, sps, pulse_shape='rrc')
    print(f"Длина сигнала после upsampling: {len(signal)} отсчетов")
    
    # 3. Вносим временной сдвиг (имитация ошибки СТС)
    timing_offset = 2  # сдвиг на 0.3 отсчета
    signal_with_offset = add_timing_offset(signal, timing_offset, method='linear')
    print(f"Внесен временной сдвиг: {timing_offset} отсчета")
    
    # 4. Добавляем небольшой шум
    snr_db = 30
    signal_noisy = add_noise(signal_with_offset, snr_db)
    print(f"Добавлен шум: SNR = {snr_db} дБ")
    
    # 5. Применяем Gardner TED для восстановления синхронизации
    print("\nПрименяем детектор Gardner TED...")
    recovered_symbols, errors, mu_history = gardner_ted(signal_noisy, sps)
    print(f"Восстановлено {len(recovered_symbols)} символов")
    
    # =============== ВИЗУАЛИЗАЦИЯ ===============
    
    # Созвездия
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 3, 1)
    plt.scatter(np.real(symbols[:200]), np.imag(symbols[:200]), s=20, alpha=0.6)
    plt.title("Исходные QPSK символы")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 3, 2)
    plt.scatter(np.real(signal[::sps][:200]), np.imag(signal[::sps][:200]), s=20, alpha=0.6)
    plt.title("После формирования импульсов")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 3, 3)
    plt.scatter(np.real(signal_with_offset[::sps][:200]), 
                np.imag(signal_with_offset[::sps][:200]), s=20, alpha=0.6)
    plt.title(f"После временного сдвига ({timing_offset} отсч.)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 3, 4)
    plt.scatter(np.real(signal_noisy[::sps][:200]), 
                np.imag(signal_noisy[::sps][:200]), s=20, alpha=0.6)
    plt.title(f"После добавления шума (SNR={snr_db} дБ)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 3, 5)
    plt.scatter(np.real(recovered_symbols[:200]), 
                np.imag(recovered_symbols[:200]), s=20, alpha=0.6)
    plt.title("После Gardner TED")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')
    
    # График ошибок Gardner
    plt.subplot(2, 3, 6)
    plt.plot(errors)
    plt.title("Ошибка синхронизации Gardner")
    plt.xlabel("Номер символа")
    plt.ylabel("Ошибка")
    plt.grid(True)
    
    # Дополнительные графики
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mu_history)
    plt.title(f"Дробный сдвиг μ (среднее: {np.mean(mu_history):.3f})")
    plt.xlabel("Номер символа")
    plt.ylabel("μ (отсчёты)")
    plt.grid(True)
    plt.axhline(y=-timing_offset, color='r', linestyle='--', 
                label=f'Целевое значение: {-timing_offset:.3f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Сигнал во временной области (первые 100 отсчетов)
    t = np.arange(100)
    plt.plot(t, np.real(signal[:100]), 'b-', alpha=0.7, label='I (без сдвига)')
    plt.plot(t, np.imag(signal[:100]), 'r-', alpha=0.7, label='Q (без сдвига)')
    plt.plot(t, np.real(signal_with_offset[:100]), 'b--', alpha=0.5, label='I (со сдвигом)')
    plt.plot(t, np.imag(signal_with_offset[:100]), 'r--', alpha=0.5, label='Q (со сдвигом)')
    plt.title("Сигнал во временной области")
    plt.xlabel("Отсчет")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print("="*50)
    print(f"Внесенный временной сдвиг: {timing_offset:.3f} отсчета")
    print(f"Оценка Gardner (среднее μ): {np.mean(mu_history):.3f} отсчета")
    print(f"Оценка Gardner (финальное μ): {mu_history[-1]:.3f} отсчета")
    print(f"Ошибка оценки: {abs(mu_history[-1] + timing_offset):.3f} отсчета")
    print("="*50)

