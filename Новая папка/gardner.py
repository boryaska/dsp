import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("iq_int16.pcm")  


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


def gardner_ted(x, sps):
    """
    Gardner Timing Error Detector для IQ-сигнала.
    
    Параметры:
    x   - комплексный сигнал (numpy array)
    sps - samples per symbol (целое > 1)
    
    Возвращает:
    y       - отсчёты, взятые с корректировкой времени (downsampled)
    errors  - массив ошибок синхронизации
    mu_history - история значений mu (дробная задержка)
    """

    # инициализация
    n_symbols = (len(x) - 2 * sps) // sps  # сколько полных символов (с запасом)
    y = []
    errors = []
    mu_history = []

    mu = 0.0  # дробная задержка (отсчёты)
    delta_mu = 0.001  # шаг корректировки (learning rate) - уменьшил для стабильности
    
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

        # Ошибка Gardner
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


def read_iq_int16(path: Path, normalize: bool = True) -> np.ndarray:
    """Читает int16 IQ-файл и возвращает комплексный массив."""
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size < 2:
        raise ValueError("Файл должен содержать хотя бы одну пару I/Q отсчётов.")
    if raw.size % 2:
        raw = raw[:-1]
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    iq = i + 1j * q
    if normalize:
        peak = np.max(np.abs(iq))
        if peak > 0:
            iq = iq / peak
    return iq.astype(np.complex64)


try:
    iq_samples = read_iq_int16(DATA_PATH)
    print(f"Загружено {iq_samples.size} комплексных отсчётов из {DATA_PATH.name}")
    print(f"Пример первых 5 отсчётов: {iq_samples[:5]}")
except FileNotFoundError:
    iq_samples = np.array([], dtype=np.complex64)
    print(f"Файл {DATA_PATH} не найден. Укажите корректный путь в переменной DATA_PATH.")
except Exception as exc:
    iq_samples = np.array([], dtype=np.complex64)
    print(f"Ошибка при чтении: {exc}")

f_rel = -0.00026

if iq_samples.size and not np.isclose(f_rel, 0.0):
    n = np.arange(iq_samples.size, dtype=np.float32)
    iq_baseband = iq_samples * np.exp(-1j * 2 * np.pi * f_rel * n)
    print("Компенсация по оценённому f_rel выполнена.")
else:
    iq_baseband = iq_samples.copy()
    print("Компенсация не выполнялась (f_rel≈0 или нет данных).")

if iq_baseband.size:
    sps_file = 2
    y_file, errors_file, mu_history = gardner_ted(iq_baseband, sps_file)

    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    plt.scatter(np.real(iq_samples), np.imag(iq_samples), s=5)
    plt.title("Созвездие RAW IQ")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1,3,2)
    plt.scatter(np.real(iq_baseband), np.imag(iq_baseband), s=5)
    plt.title("Созвездие после компенсации f_rel")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(1,3,3)
    plt.scatter(np.real(y_file), np.imag(y_file), s=5)
    plt.title("Созвездие после Gardner TED")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis('equal')

    # График ошибок и сдвига
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(errors_file)
    plt.title("Ошибка синхронизации Gardner")
    plt.xlabel("Номер символа")
    plt.ylabel("Ошибка")
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(mu_history)
    plt.title(f"Дробный сдвиг μ (среднее: {np.mean(mu_history):.3f})")
    plt.xlabel("Номер символа")
    plt.ylabel("μ (отсчёты)")
    plt.grid(True)
    
    print(f"Средний сдвиг μ: {np.mean(mu_history):.3f} отсчётов")
    print(f"Финальный сдвиг μ: {mu_history[-1]:.3f} отсчётов")
else:
    y_file = np.array([], dtype=np.complex64)
    errors_file = np.array([], dtype=np.float32)
    mu_history = np.array([], dtype=np.float32)


plt.show()