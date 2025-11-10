import numpy as np


def estimate_initial_frequency(signal, num_samples):
        """
        Грубая оценка частотного сдвига по первым num_samples символам
        
        Метод: Fourth-power method для QPSK
        -------------------------------------
        1. Возводим сигнал в 4-ю степень: z[n] = s[n]^4
        2. Для QPSK это полностью убирает модуляцию:
           s[n] = a[n]·exp(j·2π·Δf·n), где a[n] ∈ {±1±j}/√2
           s[n]^4 = a[n]^4·exp(j·8π·Δf·n)
           a[n]^4 = const для любого QPSK символа
        3. Вычисляем автокорреляцию с задержкой 1:
           R = Σ z[n]·conj(z[n-1]) = exp(j·8π·Δf)
        4. Частота: Δf = angle(R) / (8π)
        
        Преимущества:
        - Полностью убирает QPSK модуляцию
        - Точность не зависит от случайности символов
        - Стандартный метод в DSP
        
        Args:
            signal: массив комплексных символов
            num_samples: количество символов для анализа
            
        Returns:
            freq_estimate: оценка частотного сдвига
        """
        # Используем только доступные отсчеты
        n = min(num_samples, len(signal))
        if n < 2:
            return 0.0
        
        # Возводим сигнал в 4-ю степень для удаления QPSK модуляции
        signal_4th = signal[:n] ** 4
        
        # Вычисляем автокорреляцию с задержкой 1
        autocorr = 0.0 + 0.0j
        for i in range(1, n):
            autocorr += signal_4th[i] * np.conj(signal_4th[i-1])
        
        # Усредняем
        autocorr /= (n - 1)
        
        # Извлекаем фазовый сдвиг
        # s^4 создает 4x частотное умножение: phase = 8π·Δf
        phase_shift = np.angle(autocorr)
        
        # Частота = фазовый сдвиг / (8π)
        # Делим на 8π, т.к. s^4 умножает частоту на 4, 
        # а автокорреляция с задержкой 1 добавляет еще x2
        freq_estimate = phase_shift / (8 * np.pi)
        
        print(f"[Auto-Acquire Debug] phase_shift={phase_shift:.6f}, freq_estimate={freq_estimate:.6f}")
        
        return freq_estimate

if __name__ == "__main__":
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 4000) / np.sqrt(2)
    signal = symbols * np.exp(1j * 2 * np.pi * 0.005242 * np.arange(len(symbols)))
    freq_estimate = estimate_initial_frequency(signal, 4000)
    print(f"freq_estimate: {freq_estimate:.6f}")