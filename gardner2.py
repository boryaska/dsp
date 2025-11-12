import numpy as np
import matplotlib.pyplot as plt
from read_file import rrc_filter


def gardner_timing_recovery(signal, sps, alpha=0.007, mu_initial=0.0):
    """
    Алгоритм Гарднера для восстановления тактовой синхронизации.
    
    Параметры:
    ----------
    signal : np.ndarray
        Входной сигнал (комплексный или действительный)
    sps : int
        Samples per symbol (количество отсчетов на символ)
    alpha : float, optional
        Коэффициент адаптации (шаг корректировки), по умолчанию 0.07
        Меньше значение = медленнее сходимость, но стабильнее
        Больше значение = быстрее сходимость, но может быть нестабильно
    mu_initial : float, optional
        Начальное значение дробной задержки, по умолчанию 0.0
    
    Возвращает:
    -----------
    recovered : np.ndarray
        Восстановленные символы
    timing_errors : np.ndarray
        История ошибок синхронизации
    mu_history : np.ndarray
        История значений mu (дробной задержки)
    
    Пример использования:
    --------------------
    >>> recovered, errors, mu_hist = gardner_timing_recovery(shifted_signal, sps=2, alpha=0.07)
    >>> plt.plot(recovered.real, recovered.imag, 'o')
    """
    
    def interpolate(sig, idx):
        """Линейная интерполация для дробного индекса"""
        idx_int = int(idx)
        frac = idx - idx_int
        
        # Проверка границ массива
        if idx_int < 0 or idx_int >= len(sig):
            return 0.0  # Возвращаем 0 для выхода за границы
        
        if idx_int + 1 < len(sig):
            return sig[idx_int] * (1 - frac) + sig[idx_int + 1] * frac
        else:
            return sig[idx_int]
    
    timing_errors = []
    recovered = []
    mu_history = []
    
    mu = mu_initial
    k = sps * 2  # начинаем с 2 символов запаса
    
    while k < len(signal) - sps:
        # Нормализация mu: если mu выходит за пределы [-sps, sps], корректируем k
        while mu >= sps:
            mu -= sps
            k += sps
        while mu < -sps:
            mu += sps
            k -= sps
        
        # Берем три отсчета с учетом дробной задержки mu
        idx_early = k - sps + mu
        idx_mid = k - sps//2 + mu
        idx_late = k + mu
        
        # Проверка границ для всех индексов
        if (idx_early < 0 or idx_late + 1 >= len(signal) or 
            k + mu < 0 or k + mu + 1 >= len(signal)):
            break
        
        x_early = interpolate(signal, idx_early)
        x_mid = interpolate(signal, idx_mid)
        x_late = interpolate(signal, idx_late)
        
        # Формула Гарднера: error = real(mid * conj(late - early))
        err = np.real(x_mid * np.conj(x_late - x_early))
        timing_errors.append(err)
        
        # Обновляем mu (дробную задержку)
        mu = mu - alpha * err
        mu_history.append(mu)
        
        # Берем отсчет на текущей позиции с коррекцией
        recovered.append(interpolate(signal, k + mu))
        
        # Переходим к следующему символу
        k += sps
    
    return np.array(recovered), np.array(timing_errors), np.array(mu_history)

if __name__ == "__main__":
    Nsym = 2000         # число символов
    sps = 8            # samples per symbol
    T = 1              # символный период (условно)
    mu = 0.0           # fractional timing offset (начальный)
    alpha = 0.07       # шаг корректировки тайминга (петли)
    np.random.seed(1)


    data = np.random.randint(0, 4, Nsym)
    symbols = np.exp(1j * 2 * np.pi * data / 4)
    # plt.plot(symbols.real, symbols.imag, 'o')



    samples = np.zeros(Nsym*sps, dtype=complex)
    for i in range(Nsym):
        samples[i*sps] = symbols[i]

    # plt.plot(np.abs(samples), 'o')    

    rrc = rrc_filter(sps, 10, 0.35)
    samples = np.convolve(samples, rrc, mode='same')
    # plt.plot(samples.real, samples.imag, 'o')

        


    # --- Вносим временной сдвиг (искусственно) ---
    delay = 0.095  
    t = np.arange(len(samples))
    shifted = np.interp(t, t - delay, samples)



    # --- Gardner timing recovery ---
    recovered, timing_errors, mu_history = gardner_timing_recovery(shifted, sps, alpha=alpha, mu_initial=mu)

    # --- Визуализация ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    recovered_arr = recovered  # уже является np.array после вызова функции

    # График сигнала со сдвигом
    ax = axes[0, 0]
    ax.plot(shifted.real, shifted.imag, 'o', alpha=0.3, markersize=4)
    ax.set_title("Сигнал со сдвигом", fontsize=12, fontweight='bold')
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # График восстановленных символов
    ax = axes[0, 1]
    ax.plot(recovered_arr.real, recovered_arr.imag, 'go', markersize=8, alpha=0.7)
    ax.set_title("Восстановленные символы", fontsize=12, fontweight='bold')
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # График ошибок синхронизации
    ax = axes[0, 2]
    ax.plot(timing_errors)
    ax.set_title("Ошибка синхронизации (Gardner TED)")
    ax.set_xlabel("Символ")
    ax.set_ylabel("Ошибка")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    # График истории mu
    ax = axes[1, 0]
    ax.plot(mu_history)
    ax.set_title("История mu (дробная задержка)")
    ax.set_xlabel("Символ")
    ax.set_ylabel("mu (отсчеты)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    # График исходных символов
    ax = axes[1, 1]
    ax.plot(symbols.real, symbols.imag, 'rx', markersize=8, alpha=0.7)
    ax.set_title("Исходные символы", fontsize=12, fontweight='bold')
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # График сравнения (наложение)
    ax = axes[1, 2]
    ax.plot(symbols.real, symbols.imag, 'rx', markersize=10, alpha=0.4, label='Исходные')
    ax.plot(recovered_arr.real, recovered_arr.imag, 'go', markersize=6, alpha=0.7, label='Восстановленные')
    ax.set_title("Сравнение (наложение)")
    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"Количество восстановленных символов: {len(recovered)}")
    print(f"Количество исходных символов: {len(symbols)}")
    print(f"Средняя ошибка тайминга: {np.mean(np.abs(timing_errors)):.4f}")
    print(f"Финальное значение mu: {mu_history[-1]:.4f}")