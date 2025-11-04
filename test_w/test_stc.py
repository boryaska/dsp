

"""
Синтез QPSK сигнала с RRC (alpha=0.3), sps=5, частотной отстройкой 0.007 и фазой 35°.
Результат сохраняется в qpsk_signal.pcm как float32 I/Q interleaved.
"""

import numpy as np
from scipy import signal
import numpy.fft as fft
from typing import Optional, Tuple


def rrc_filter(alpha: float, sps: int, span_symbols: int) -> np.ndarray:
    """Импульсная характеристика RRC длиной span_symbols*sps+1 (нечётное число тапов)."""
    t = np.arange(-span_symbols/2, span_symbols/2 + 1/sps, 1/sps)
    h = np.zeros_like(t)
    eps = 1e-12

    zero_idx = np.abs(t) < eps
    h[zero_idx] = 1 - alpha + 4*alpha/np.pi

    special_idx = np.zeros_like(t, dtype=bool)
    if alpha > 0:
        special_idx = np.isclose(np.abs(t), 1/(4*alpha), atol=1e-12)
        if np.any(special_idx):
            h[special_idx] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )

    normal_idx = ~(zero_idx | special_idx)
    if np.any(normal_idx):
        tn = t[normal_idx]
        h[normal_idx] = (
            np.sin(np.pi*tn*(1-alpha)) + 4*alpha*tn*np.cos(np.pi*tn*(1+alpha))
        ) / (
            np.pi*tn*(1 - (4*alpha*tn)**2)
        )

    h = h / np.sqrt(np.sum(h**2))
    return h


def qpsk_modulate(
    num_symbols: int,
    sps: int = 5,
    alpha: float = 0.3,
    freq_offset: float = 0.007,
    phase_deg: float = 35.0,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Генерация комплексной огибающей QPSK с формирующим RRC и частотным сдвигом."""
    rng = np.random.default_rng(random_seed)

    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
    sym_idx = rng.integers(0, 4, size=num_symbols)
    symbols = constellation[sym_idx]

    h_rrc = rrc_filter(alpha, sps, span_symbols=10)

    shaped = signal.upfirdn(h_rrc, symbols, up=sps)
    gd = (len(h_rrc) - 1) // 2
    start = gd
    stop = start + num_symbols * sps
    shaped = shaped[start:stop]

    n = np.arange(len(shaped), dtype=np.float64)
    carrier = np.exp(1j * (2*np.pi*freq_offset*n + np.deg2rad(phase_deg)))
    y = shaped * carrier
    return y.astype(np.complex64)


def save_iq_pcm(x: np.ndarray, filename: str) -> None:
    """Сохранение I/Q в RAW PCM float32 (I,Q,I,Q,...)"""
    iq = np.empty(2 * len(x), dtype=np.float32)
    iq[0::2] = x.real.astype(np.float32, copy=False)
    iq[1::2] = x.imag.astype(np.float32, copy=False)
    with open(filename, "wb") as f:
        iq.tofile(f)


def compute_amplitude_spectrum(x: np.ndarray, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Амплитудно-частотный спектр: возвращает (f_norm, |X(f)| в dB)."""
    if window == "hann":
        w = np.hanning(len(x))
    elif window == "hamming":
        w = np.hamming(len(x))
    else:
        w = np.ones(len(x))

    X = fft.fftshift(fft.fft(x * w))
    f = np.linspace(-0.5, 0.5, len(X), endpoint=False)
    Sdb = 20.0 * np.log10(np.maximum(np.abs(X), 1e-12))
    return f, Sdb


def plot_spectrum(f: np.ndarray, Sdb: np.ndarray, title: str = "Амплитудно-частотный спектр") -> None:
    """Отображение спектра, если доступен matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib недоступен: пропускаю отображение графика")
        return

    plt.figure(figsize=(7, 3))
    plt.plot(f, Sdb)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Нормированная частота, циклы/отсчёт")
    plt.ylabel("Амплитуда, dB")
    plt.tight_layout()
    plt.show()


def test_point_1():
    print("Пункт 1: синтез QPSK")

    params = dict(
        num_symbols=1000,
        sps=5,
        alpha=0.3,
        freq_offset=0.007,
        phase_deg=35.0,
        random_seed=42,
    )

    y = qpsk_modulate(**params)

    outfile = "qpsk_signal.pcm"
    save_iq_pcm(y, outfile)

    print(f"L={len(y)} сэмплов, Pavg={np.mean(np.abs(y)**2):.4f}, файл: {outfile}")

    # Пункт 2: расчёт АЧС и (опционально) отображение
    f, Sdb = compute_amplitude_spectrum(y, window="hann")
    print(f"Спектр рассчитан: {len(f)} частотных отсчётов")

    # Пункт 3: отображение АЧС
    plot_spectrum(f, Sdb, title="Спектр синтезированного QPSK сигнала")
    return y


if __name__ == "__main__":
    signal = test_point_1()
