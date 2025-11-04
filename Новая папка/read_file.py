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


# Загрузка сигнала
signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

f_rel = 0.01643158030233549
n = np.arange(signal_iq.size, dtype=np.float32)
signal_iq = signal_iq * np.exp(-1j * 2 * np.pi * f_rel * n)
print("Компенсация по оценённому f_rel выполнена.")

# Применение RRC фильтра
sps = 4  # samples per symbol
span = 10  # длина фильтра в символах
alpha = 0.35  # roll-off factor

rrc = rrc_filter(sps, span, alpha)
signal_filtered = np.convolve(signal_iq, rrc, mode='same')
print(f"RRC фильтр применен: sps={sps}, span={span}, alpha={alpha}")
print(f"Длина фильтра: {len(rrc)} отсчетов")

# Применение алгоритма Гарднера для синхронизации символов
print(f"\nПрименение алгоритма Гарднера...")
delta_mu = 0.001  # Коэффициент обучения (можно регулировать)
signal_gardner, gardner_errors, mu_history = gardner_ted(signal_filtered, sps, delta_mu)
print(f"Алгоритм Гарднера выполнен:")
print(f"  Извлечено символов: {len(signal_gardner)}")
print(f"  Средний сдвиг μ: {np.mean(mu_history):.4f} отсчётов")
print(f"  Финальный сдвиг μ: {mu_history[-1]:.4f} отсчётов")
print(f"  Стандартное отклонение ошибки: {np.std(gardner_errors):.6f}")

# Создание массива сигналов со сдвигом по отсчетам для подбора фазы дискретизации (для сравнения)
sampled_signals = []
timing_phases = []

print(f"\nСоздание массива сигналов с разными фазами дискретизации:")

# Определяем минимальную длину для всех фаз
min_length = len(signal_gardner[sps-1::sps])  # Последняя фаза имеет минимальную длину

for i in range(sps):
    # Сдвиг начальной точки на i отсчетов
    sampled = signal_gardner[i::sps]  # Берем каждый sps-й отсчет, начиная с i
    # Обрезаем до минимальной длины для однородности
    sampled = sampled[:min_length]
    sampled_signals.append(sampled)
    timing_phases.append(i)
    print(f"  Фаза {i}: {len(sampled)} символов (сдвиг {i} отсчетов)")

sampled_signals = np.array(sampled_signals, dtype=complex)
print(f"Создан массив размером: {sampled_signals.shape} (фазы x символы)")

# Вычисление метрик для каждой фазы дискретизации
metrics = []
print(f"\nВычисление метрик для выбора оптимальной фазы:")
for i in range(sps):
    # Метрика: среднеквадратичное значение амплитуды
    metric = np.mean(np.abs(sampled_signals[i]))
    metrics.append(metric)
    print(f"  Фаза {i}: средняя амплитуда = {metric:.4f}")

optimal_phase = np.argmax(metrics)
print(f"\nОптимальная фаза дискретизации: {optimal_phase}")
print(f"Значение метрики: {metrics[optimal_phase]:.4f}")

# Выбор оптимально дискретизированного сигнала
signal_sampled = sampled_signals[optimal_phase]
print(f"Выбран сигнал с фазой {optimal_phase}, длина: {len(signal_sampled)} символов")


# Построение графиков RRC фильтра
fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))

# Импульсная характеристика RRC
t_rrc = np.arange(len(rrc)) - len(rrc) // 2
axes1[0].plot(t_rrc / sps, rrc, linewidth=2, marker='o', markersize=3)
axes1[0].set_title('Импульсная характеристика RRC фильтра', fontsize=12, fontweight='bold')
axes1[0].set_xlabel('Время (символы)')
axes1[0].set_ylabel('Амплитуда')
axes1[0].grid(True, alpha=0.3)

# АЧХ RRC фильтра
rrc_freq = np.fft.fftshift(np.fft.fft(rrc, 1024))
rrc_freq_db = 20 * np.log10(np.abs(rrc_freq) + 1e-10)
freq_axis = np.fft.fftshift(np.fft.fftfreq(1024))
axes1[1].plot(freq_axis, rrc_freq_db, linewidth=1.5)
axes1[1].set_title('АЧХ RRC фильтра', fontsize=12, fontweight='bold')
axes1[1].set_xlabel('Нормализованная частота (f/fs)')
axes1[1].set_ylabel('Амплитуда (дБ)')
axes1[1].grid(True, alpha=0.3)
axes1[1].set_xlim([-0.5, 0.5])

plt.tight_layout()

# Графики алгоритма Гарднера
fig_gardner, axes_gardner = plt.subplots(2, 2, figsize=(14, 10))

# График ошибки синхронизации
axes_gardner[0, 0].plot(gardner_errors, linewidth=0.8, alpha=0.7)
axes_gardner[0, 0].set_title('Ошибка синхронизации Gardner TED', fontsize=12, fontweight='bold')
axes_gardner[0, 0].set_xlabel('Номер символа')
axes_gardner[0, 0].set_ylabel('Ошибка')
axes_gardner[0, 0].grid(True, alpha=0.3)
axes_gardner[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)

# График истории mu (дробный сдвиг)
axes_gardner[0, 1].plot(mu_history, linewidth=0.8, color='darkgreen')
axes_gardner[0, 1].set_title(f'Дробный сдвиг μ (среднее: {np.mean(mu_history):.4f})', 
                             fontsize=12, fontweight='bold')
axes_gardner[0, 1].set_xlabel('Номер символа')
axes_gardner[0, 1].set_ylabel('μ (отсчёты)')
axes_gardner[0, 1].grid(True, alpha=0.3)

# Гистограмма ошибок
axes_gardner[1, 0].hist(gardner_errors, bins=50, edgecolor='black', alpha=0.7)
axes_gardner[1, 0].set_title(f'Гистограмма ошибок (std={np.std(gardner_errors):.6f})', 
                             fontsize=12, fontweight='bold')
axes_gardner[1, 0].set_xlabel('Ошибка')
axes_gardner[1, 0].set_ylabel('Количество')
axes_gardner[1, 0].grid(True, alpha=0.3, axis='y')

# Созвездие после Gardner
n_gardner = min(1000, len(signal_gardner))
axes_gardner[1, 1].scatter(signal_gardner[:n_gardner].real, 
                           signal_gardner[:n_gardner].imag, 
                           s=10, alpha=0.6, c='purple')
axes_gardner[1, 1].set_title(f'Созвездие после Gardner ({n_gardner} символов)', 
                             fontsize=12, fontweight='bold')
axes_gardner[1, 1].set_xlabel('I (In-phase)')
axes_gardner[1, 1].set_ylabel('Q (Quadrature)')
axes_gardner[1, 1].grid(True, alpha=0.3)
axes_gardner[1, 1].axis('equal')

# Добавление идеальных точек QPSK
ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
axes_gardner[1, 1].scatter(ideal_points.real, ideal_points.imag, 
                           s=150, marker='x', c='red', linewidths=2, 
                           label='Идеальные точки', zorder=5)
axes_gardner[1, 1].legend(fontsize=9)

fig_gardner.suptitle('Анализ алгоритма Гарднера', fontsize=14, fontweight='bold')
plt.tight_layout()

# Создание временной оси
time = np.arange(len(signal_iq))

# Построение графиков сигнала до и после фильтрации
fig2, axes = plt.subplots(3, 2, figsize=(14, 10))

# До фильтрации - I компонента
axes[0, 0].plot(time, signal_iq.real, linewidth=0.5)
axes[0, 0].set_title('До фильтрации: I компонента', fontsize=11)
axes[0, 0].set_xlabel('Отсчеты')
axes[0, 0].set_ylabel('Амплитуда')
axes[0, 0].grid(True, alpha=0.3)

# После фильтрации - I компонента
axes[0, 1].plot(time, signal_filtered.real, linewidth=0.5, color='darkblue')
axes[0, 1].set_title('После RRC фильтрации: I компонента', fontsize=11)
axes[0, 1].set_xlabel('Отсчеты')
axes[0, 1].set_ylabel('Амплитуда')
axes[0, 1].grid(True, alpha=0.3)

# До фильтрации - Q компонента
axes[1, 0].plot(time, signal_iq.imag, linewidth=0.5, color='orange')
axes[1, 0].set_title('До фильтрации: Q компонента', fontsize=11)
axes[1, 0].set_xlabel('Отсчеты')
axes[1, 0].set_ylabel('Амплитуда')
axes[1, 0].grid(True, alpha=0.3)

# После фильтрации - Q компонента
axes[1, 1].plot(time, signal_filtered.imag, linewidth=0.5, color='darkorange')
axes[1, 1].set_title('После RRC фильтрации: Q компонента', fontsize=11)
axes[1, 1].set_xlabel('Отсчеты')
axes[1, 1].set_ylabel('Амплитуда')
axes[1, 1].grid(True, alpha=0.3)

# До фильтрации - Амплитуда
axes[2, 0].plot(time, np.abs(signal_iq), linewidth=0.5, color='green')
axes[2, 0].set_title('До фильтрации: Амплитуда', fontsize=11)
axes[2, 0].set_xlabel('Отсчеты')
axes[2, 0].set_ylabel('Амплитуда')
axes[2, 0].grid(True, alpha=0.3)

# После фильтрации - Амплитуда
axes[2, 1].plot(time, np.abs(signal_filtered), linewidth=0.5, color='darkgreen')
axes[2, 1].set_title('После RRC фильтрации: Амплитуда', fontsize=11)
axes[2, 1].set_xlabel('Отсчеты')
axes[2, 1].set_ylabel('Амплитуда')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Построение спектров до и после фильтрации
fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))

# Вычисление FFT для исходного сигнала
N = len(signal_iq)
spectrum = np.fft.fftshift(np.fft.fft(signal_iq))
spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-10)

# Вычисление FFT для отфильтрованного сигнала
spectrum_filtered = np.fft.fftshift(np.fft.fft(signal_filtered))
spectrum_filtered_db = 20 * np.log10(np.abs(spectrum_filtered) + 1e-10)

# Частотная ось (нормализованная частота)
freq = np.fft.fftshift(np.fft.fftfreq(N))

# График спектра до фильтрации
axes3[0].plot(freq, spectrum_db, linewidth=0.8)
axes3[0].set_title('Спектр сигнала ДО RRC фильтрации', fontsize=12, fontweight='bold')
axes3[0].set_xlabel('Нормализованная частота (f/fs)')
axes3[0].set_ylabel('Амплитуда (дБ)')
axes3[0].grid(True, alpha=0.3)
axes3[0].set_xlim([-0.5, 0.5])

# График спектра после фильтрации
axes3[1].plot(freq, spectrum_filtered_db, linewidth=0.8, color='darkblue')
axes3[1].set_title('Спектр сигнала ПОСЛЕ RRC фильтрации', fontsize=12, fontweight='bold')
axes3[1].set_xlabel('Нормализованная частота (f/fs)')
axes3[1].set_ylabel('Амплитуда (дБ)')
axes3[1].grid(True, alpha=0.3)
axes3[1].set_xlim([-0.5, 0.5])

plt.tight_layout()

# Визуализация разных фаз дискретизации
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# Созвездия для каждой фазы
for i in range(sps):
    row = i // 2
    col = i % 2
    
    # Берем ограниченное количество символов для наглядности
    n_symbols = min(500, len(sampled_signals[i]))
    
    axes4[row, col].scatter(sampled_signals[i][:n_symbols].real, 
                           sampled_signals[i][:n_symbols].imag, 
                           s=10, alpha=0.5)
    axes4[row, col].set_title(f'Фаза {i} (метрика: {metrics[i]:.4f})', 
                             fontsize=11, 
                             fontweight='bold' if i == optimal_phase else 'normal',
                             color='darkgreen' if i == optimal_phase else 'black')
    axes4[row, col].set_xlabel('I (In-phase)')
    axes4[row, col].set_ylabel('Q (Quadrature)')
    axes4[row, col].grid(True, alpha=0.3)
    axes4[row, col].axis('equal')
    
    if i == optimal_phase:
        axes4[row, col].set_facecolor('#f0fff0')  # Светло-зеленый фон для оптимальной фазы

fig4.suptitle(f'Сравнение созвездий для разных фаз дискретизации\n(Оптимальная фаза: {optimal_phase})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# График метрик для всех фаз
fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
ax5.bar(range(sps), metrics, color=['darkgreen' if i == optimal_phase else 'steelblue' for i in range(sps)])
ax5.set_xlabel('Фаза дискретизации', fontsize=12)
ax5.set_ylabel('Средняя амплитуда', fontsize=12)
ax5.set_title('Метрики для разных фаз дискретизации', fontsize=14, fontweight='bold')
ax5.set_xticks(range(sps))
ax5.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics):
    ax5.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Созвездие оптимально дискретизированного сигнала
fig6, ax6 = plt.subplots(1, 1, figsize=(10, 10))
n_symbols = min(2000, len(signal_sampled))
ax6.scatter(signal_sampled[:n_symbols].real, 
           signal_sampled[:n_symbols].imag, 
           s=15, alpha=0.6, c='darkblue')
ax6.set_title(f'Созвездие QPSK (фаза {optimal_phase}, {n_symbols} символов)', 
             fontsize=14, fontweight='bold')
ax6.set_xlabel('I (In-phase)', fontsize=12)
ax6.set_ylabel('Q (Quadrature)', fontsize=12)
ax6.grid(True, alpha=0.3)
ax6.axis('equal')

# Добавление идеальных точек QPSK для справки
ideal_points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
ax6.scatter(ideal_points.real, ideal_points.imag, 
           s=200, marker='x', c='red', linewidths=3, 
           label='Идеальные точки QPSK', zorder=5)
ax6.legend(fontsize=11)

plt.tight_layout()

# Финальное сравнение: Гарднер vs Простая дискретизация
fig_compare, axes_compare = plt.subplots(1, 2, figsize=(14, 6))

# Созвездие Гарднера
n_comp = min(1500, len(signal_gardner), len(signal_sampled))
axes_compare[0].scatter(signal_gardner[:n_comp].real, 
                        signal_gardner[:n_comp].imag, 
                        s=12, alpha=0.6, c='purple')
axes_compare[0].scatter(ideal_points.real, ideal_points.imag, 
                        s=150, marker='x', c='red', linewidths=2, 
                        label='Идеальные точки', zorder=5)
axes_compare[0].set_title(f'Метод 1: Алгоритм Гарднера\n(амплитуда={np.mean(np.abs(signal_gardner[:n_comp])):.4f})', 
                          fontsize=12, fontweight='bold')
axes_compare[0].set_xlabel('I (In-phase)', fontsize=11)
axes_compare[0].set_ylabel('Q (Quadrature)', fontsize=11)
axes_compare[0].grid(True, alpha=0.3)
axes_compare[0].axis('equal')
axes_compare[0].legend(fontsize=9)

# Созвездие простой дискретизации
axes_compare[1].scatter(signal_sampled[:n_comp].real, 
                        signal_sampled[:n_comp].imag, 
                        s=12, alpha=0.6, c='darkblue')
axes_compare[1].scatter(ideal_points.real, ideal_points.imag, 
                        s=150, marker='x', c='red', linewidths=2, 
                        label='Идеальные точки', zorder=5)
axes_compare[1].set_title(f'Метод 2: Простая дискретизация (фаза {optimal_phase})\n(амплитуда={metrics[optimal_phase]:.4f})', 
                          fontsize=12, fontweight='bold')
axes_compare[1].set_xlabel('I (In-phase)', fontsize=11)
axes_compare[1].set_ylabel('Q (Quadrature)', fontsize=11)
axes_compare[1].grid(True, alpha=0.3)
axes_compare[1].axis('equal')
axes_compare[1].legend(fontsize=9)

fig_compare.suptitle('Сравнение методов синхронизации символов', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()

print(f"\n{'='*70}")
print(f"ИТОГОВАЯ СТАТИСТИКА ОБРАБОТКИ СИГНАЛА")
print(f"{'='*70}")
print(f"Исходный сигнал:")
print(f"  Количество отсчетов: {len(signal_iq)}")
print(f"  Диапазон I: [{signal_iq.real.min():.3f}, {signal_iq.real.max():.3f}]")
print(f"  Диапазон Q: [{signal_iq.imag.min():.3f}, {signal_iq.imag.max():.3f}]")

print(f"\nПосле RRC фильтрации:")
print(f"  Диапазон I: [{signal_filtered.real.min():.3f}, {signal_filtered.real.max():.3f}]")
print(f"  Диапазон Q: [{signal_filtered.imag.min():.3f}, {signal_filtered.imag.max():.3f}]")

print(f"\n--- МЕТОД 1: Алгоритм Гарднера (адаптивная синхронизация) ---")
print(f"  Количество символов: {len(signal_gardner)}")
print(f"  Диапазон I: [{signal_gardner.real.min():.3f}, {signal_gardner.real.max():.3f}]")
print(f"  Диапазон Q: [{signal_gardner.imag.min():.3f}, {signal_gardner.imag.max():.3f}]")
print(f"  Средняя амплитуда: {np.mean(np.abs(signal_gardner)):.4f}")
print(f"  Среднее μ: {np.mean(mu_history):.4f} отсчётов")
print(f"  Финальное μ: {mu_history[-1]:.4f} отсчётов")

print(f"\n--- МЕТОД 2: Простая дискретизация (фаза {optimal_phase}) ---")
print(f"  Количество символов: {len(signal_sampled)}")
print(f"  Диапазон I: [{signal_sampled.real.min():.3f}, {signal_sampled.real.max():.3f}]")
print(f"  Диапазон Q: [{signal_sampled.imag.min():.3f}, {signal_sampled.imag.max():.3f}]")
print(f"  Средняя амплитуда: {metrics[optimal_phase]:.4f}")

print(f"\n{'='*70}")
print(f"ВЫВОД:")
print(f"  Алгоритм Гарднера обеспечивает адаптивную синхронизацию с учетом")
print(f"  временного дрейфа, в то время как простая дискретизация использует")
print(f"  фиксированную фазу. Gardner TED предпочтителен для реальных систем.")
print(f"{'='*70}")
