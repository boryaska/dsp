import numpy as np
import matplotlib.pyplot as plt


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


def interpolate_sample(x, idx_float):
    """
    Линейная интерполяция для получения значения в дробной позиции.
    """
    idx_low = int(np.floor(idx_float))
    idx_high = idx_low + 1
    
    if idx_high >= len(x):
        return x[idx_low]
    
    frac = idx_float - idx_low
    return x[idx_low] * (1 - frac) + x[idx_high] * frac


def gardner_ted(x, sps, delta_mu=0.002):
    """
    Gardner Timing Error Detector для IQ-сигнала.
    
    Параметры:
    x        - комплексный сигнал (numpy array)
    sps      - samples per symbol (целое >= 1)
    delta_mu - шаг корректировки (learning rate)
    
    Возвращает:
    y          - отсчёты, взятые с корректировкой времени (downsampled)
    errors     - массив ошибок синхронизации
    mu_history - история значений mu (дробная задержка)
    """
    # инициализация
    n_symbols = (len(x) - 2 * sps) // sps  # сколько полных символов (с запасом)
    y = []
    errors = []
    mu_history = []

    mu = 0.0  # дробная задержка (отсчёты)
    
    for k in range(n_symbols):
        # Вычисляем дробные индексы с учётом mu
        idx_k = k * sps + mu
        idx_next = (k + 1) * sps + mu
        idx_mid = k * sps + mu + sps / 2.0
        
        # Проверяем границы
        if idx_next + 1 >= len(x) or idx_mid + 1 >= len(x):
            break
        
        # Получаем интерполированные значения
        x_k = interpolate_sample(x, idx_k)
        x_k1 = interpolate_sample(x, idx_next)
        x_mid = interpolate_sample(x, idx_mid)

        # Ошибка Gardner (используем оба компонента I и Q)
        e = np.real(x_mid * np.conj(x_k1 - x_k))
        errors.append(e)
        mu_history.append(mu)

        # Коррекция задержки (интегрируем ошибку)
        mu += delta_mu * e

        # Ограничиваем mu в разумных пределах
        mu = np.clip(mu, -sps, sps)

        # Берем символ в текущей позиции
        y.append(x_k)

    return np.array(y), np.array(errors), np.array(mu_history)

if __name__ == "__main__":
    # Загрузка сигнала
    signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
    signal_iq = signal[::2] + 1j * signal[1::2]

    # f_rel = 0.01643158030233549
    # n = np.arange(signal_iq.size, dtype=np.float32)
    # signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)
    # print("Компенсация по оценённому f_rel выполнена.")

    # Применение RRC фильтра
    sps = 4  # samples per symbol
    span = 10  # длина фильтра в символах
    alpha = 0.35  # roll-off factor

    rrc = rrc_filter(sps, span, alpha)
    signal_filtered = np.convolve(signal_iq, rrc, mode='same')
    print(f"RRC фильтр применен: sps={sps}, span={span}, alpha={alpha}")
    print(f"Длина фильтра: {len(rrc)} отсчетов")

    # Загрузка преамбулы
    print(f"\nЗагрузка преамбулы из файла...")
    preamble_data = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
    # У преамбулы sps = 1 (один отсчет на символ - это уже символы, не oversampled)
    preamble_iq = preamble_data[::2] + 1j * preamble_data[1::2]
    print(f"Загружено {len(preamble_iq)} символов преамбулы (sps=1)")
    print(f"Первые 5 символов преамбулы: {preamble_iq[:5]}")

    # Создание массива сигналов с разными сдвигами по отсчетам
    print(f"\nСоздание массива сигналов с разными сдвигами...")
    sampled_signals = []
    conv_results = []

    for i in range(sps):
        # Берем каждый sps-й отсчет, начиная с i
        sampled = signal_filtered[i::sps]
        sampled_signals.append(sampled)
        print(f"  Сдвиг {i}: {len(sampled)} отсчетов")
        
        # Свертка с преамбулой
        conv = np.convolve(sampled, np.conj(preamble_iq[::-1]), mode='valid')
        conv_results.append(conv)
        max_pos = np.argmax(np.abs(conv))
        max_val = np.max(np.abs(conv))
        print(f"    Свертка: {len(conv)} отсчетов, макс = {max_val:.2f} в позиции {max_pos}")


    # Находим оптимальный сдвиг (с максимальной корреляцией)
    max_vals = [np.max(np.abs(conv_results[i])) for i in range(sps)]
    optimal_shift = np.argmax(max_vals)
    optimal_conv_pos = np.argmax(np.abs(conv_results[optimal_shift]))

    print(f"\nОптимальный сдвиг: {optimal_shift}")
    print(f"Максимум корреляции: {max_vals[optimal_shift]:.2f}")
    print(f"Позиция в downsampled сигнале: {optimal_conv_pos}")

    # Пересчитываем позицию обратно в координаты исходного signal_filtered
    # Позиция в signal_filtered = сдвиг + позиция_в_downsampled * sps
    cut_position = optimal_shift + optimal_conv_pos * sps
    print(f"Позиция начала преамбулы в signal_filtered: {cut_position}")

    # Обрезаем signal_filtered С начала преамбулы и далее (отбрасываем все до преамбулы)
    signal_cut = signal_filtered[cut_position:]
    print(f"Исходная длина signal_filtered: {len(signal_filtered)}")
    print(f"Длина после обрезки (с преамбулой): {len(signal_cut)}")

    # 4 графика результатов свертки для разных сдвигов
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for i in range(sps):
        row = i // 2
        col = i % 2
        
        # График амплитуды свертки
        conv_abs = np.abs(conv_results[i])
        axes[row, col].plot(conv_abs, linewidth=1, color='blue')
        
        # Отмечаем максимум
        max_pos = np.argmax(conv_abs)
        max_val = conv_abs[max_pos]
        axes[row, col].plot(max_pos, max_val, 'ro', markersize=10, 
                        label=f'Макс = {max_val:.2f}')
        
        # Подсвечиваем оптимальный сдвиг
        if i == optimal_shift:
            axes[row, col].set_facecolor('#f0fff0')
            title_weight = 'bold'
            title_color = 'darkgreen'
        else:
            title_weight = 'normal'
            title_color = 'black'
        
        axes[row, col].set_title(f'Результат свертки (сдвиг {i})', 
                                fontsize=12, fontweight=title_weight, color=title_color)
        axes[row, col].set_xlabel('Отсчет')
        axes[row, col].set_ylabel('Амплитуда')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].legend(fontsize=10)

    fig.suptitle(f'Свертка с преамбулой для разных фаз дискретизации\n(Оптимальный сдвиг: {optimal_shift})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Построение созвездия для обрезанного сигнала
    print(f"\nПостроение созвездия...")
    # Делаем downsampling (начинаем с 0, так как cut_position уже учитывает optimal_shift)
    # cut_position построена как: optimal_shift + optimal_conv_pos * sps
    # Поэтому фаза уже правильная
    signal_symbols = signal_cut[0::sps]
    print(f"Извлечено символов: {len(signal_symbols)}")

    # Созвездие
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))

    # Ограничиваем количество точек для наглядности
    n_symbols_show = min(2000, len(signal_symbols))
    ax2.scatter(signal_symbols[:n_symbols_show].real, 
            signal_symbols[:n_symbols_show].imag, 
            s=20, alpha=0.6, c='darkblue', edgecolors='black', linewidths=0.5)

    # Добавление идеальных точек QPSK
    ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    ax2.scatter(ideal_points.real, ideal_points.imag, 
            s=250, marker='x', c='red', linewidths=3, 
            label='Идеальные точки QPSK', zorder=5)

    ax2.set_title(f'Созвездие QPSK (обрезанный сигнал, сдвиг {optimal_shift})\n{n_symbols_show} символов', 
                fontsize=14, fontweight='bold')
    ax2.set_xlabel('I (In-phase)', fontsize=12)
    ax2.set_ylabel('Q (Quadrature)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend(fontsize=11)

    # Добавляем статистику
    stats_text = f'Всего символов: {len(signal_symbols)}\n'
    stats_text += f'Оптимальный сдвиг: {optimal_shift}\n'
    stats_text += f'Средняя амплитуда: {np.mean(np.abs(signal_symbols)):.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print(f"\nГотово! Сохранен сигнал с отсчета {cut_position} (начало преамбулы) и далее")
