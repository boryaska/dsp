"""Демонстрация коррекции дробного смещения выборок с помощью лагранжевого ресэмплинга.

Сценарий:
1. Генерируем QPSK-символы и формируем импульсный поезд (дельта-функции).
2. Пропускаем импульсы через RRC-фильтр — получаем базовый комплексный сигнал без ошибок.
3. Вносим дробное смещение выборок (timing offset) в 0.35 отсчёта — имитируем ошибку тактовой синхронизации.
4. Восстанавливаем исходный сигнал с помощью лагранжевого ресэмплинга и сравниваем созвездия, спектры и
   временные представления на всех этапах.

Файл можно запускать как самостоятельный скрипт:

    python hw1/test_lagrange_resampling.py

Все графики и численные показатели выводятся в интерактивном окне matplotlib и в консоль.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Добавляем корневую директорию в sys.path, чтобы корректно импортировать syntheses.py при запуске из корня репо.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hw1.synthesys import apply_rrc_filter, generate_mpsk_signal  # noqa: E402


def lagrange_resample(signal: np.ndarray, positions: np.ndarray, half_len: int = 4) -> np.ndarray:
    """Интерполирует значения ``signal`` в произвольных позициях ``positions`` при помощи полинома Лагранжа."""

    signal = np.asarray(signal)
    positions = np.asarray(positions, dtype=float)

    offsets_int = np.arange(-half_len, half_len + 1, dtype=int)
    offsets = offsets_int.astype(np.float64)
    num_offsets = len(offsets)

    # Предварительно рассчитываем барицентрические веса для равномерной сетки ``offsets``.
    weights = np.ones(num_offsets, dtype=np.float64)
    for i in range(num_offsets):
        for j in range(num_offsets):
            if i == j:
                continue
            weights[i] *= 1.0 / (offsets[i] - offsets[j])

    interpolated = np.zeros(len(positions), dtype=signal.dtype)

    for idx, pos in enumerate(positions):
        base_index = int(np.floor(pos))
        mu = pos - base_index

        indices = base_index + offsets_int
        indices = np.clip(indices, 0, len(signal) - 1)
        samples = signal[indices]

        diffs = mu - offsets
        exact_idx = np.where(np.abs(diffs) < 1e-12)[0]
        if exact_idx.size:
            interpolated[idx] = samples[exact_idx[0]]
            continue

        numer = np.sum(weights * samples / diffs)
        denom = np.sum(weights / diffs)
        interpolated[idx] = numer / denom

    return interpolated


def apply_fractional_shift(signal: np.ndarray, shift: float, half_len: int = 4) -> np.ndarray:
    """Смещает сигнал на дробное количество отсчётов (положительное значение = задержка)."""

    positions = np.arange(len(signal), dtype=float) + shift
    return lagrange_resample(signal, positions, half_len=half_len)


def sample_at_symbol_centers(
    signal: np.ndarray,
    positions: np.ndarray,
    use_lagrange: bool = False,
    half_len: int = 4,
) -> np.ndarray:
    """Берёт значения сигнала в точках выборки символов.

    Если ``use_lagrange=True``, то значения вычисляются лагранжевой интерполяцией (актуально при дробных позициях).
    В противном случае позиции приводятся к ближайшему целому индексу (имитация некорректного сэмплирования).
    """

    if use_lagrange:
        return lagrange_resample(signal, positions, half_len=half_len)

    indices = np.clip(np.round(positions).astype(int), 0, len(signal) - 1)
    return signal[indices]


def compute_spectrum(signal: np.ndarray, Fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает спектр (модуль в dB) и соответствующие частоты."""

    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1.0 / Fs))
    magnitude = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    return freqs, magnitude


def rms_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Среднеквадратическая ошибка между двумя наборами комплексных символов."""

    return float(np.sqrt(np.mean(np.abs(reference - estimate) ** 2)))


def main() -> None:
    np.random.seed(42)

    modulation_type = "qpsk"
    Fs = 100_000
    Sps = 100
    Fc = 1_000
    num_symbols = 200
    timing_offset = 0.35  # дробное смещение в отсчётах (>1, сильная ошибка синхронизации)

    print("=" * 80)
    print("Лагранжев ресэмплинг для коррекции дробного смещения выборок")
    print("=" * 80)
    print(f"Тип модуляции: {modulation_type.upper()}, Sps={Sps}, Fs={Fs} Гц, смещение = {timing_offset:.2f} отсчёта")

    # 1. Генерируем импульсный поезд символов (делта-функции)
    base_signal = generate_mpsk_signal(
        modulation_type=modulation_type,
        Fs=Fs,
        Fc=Fc,
        Sps=Sps,
        num_symbols=num_symbols,
        use_pulse_shaping=True,
    )

    # Нам нужен только импульсный поезд (symbols_expanded) и оригинальные символы для сравнения
    symbol_impulses = base_signal["symbols_expanded"]

    # 2. Применяем RRC фильтр (имитация передатчика). Сигнал остаётся в базовой полосе (baseband).
    rrc_result = apply_rrc_filter(symbol_impulses, Sps=Sps, alpha=0.35, filter_span=10, Fs=Fs)
    baseband_clean = rrc_result["signal"]
    filter_delay = len(rrc_result["filter"]) // 2

    # Выбираем символы, не затронутые краевыми эффектами фильтра
    guard_symbols = 12
    valid_symbol_indices = np.arange(guard_symbols, num_symbols - guard_symbols)
    symbol_positions = filter_delay + valid_symbol_indices * Sps

    clean_constellation = baseband_clean[symbol_positions]

    # 3. Вносим дробное смещение (timing offset): «съезжаем» по времени на величину timing_offset
    misaligned_signal = apply_fractional_shift(baseband_clean, shift=timing_offset, half_len=6)

    # Имитируем неправильное сэмплирование: берём ближайший отсчёт без компенсации
    misaligned_constellation = sample_at_symbol_centers(
        misaligned_signal, symbol_positions.astype(float), use_lagrange=False
    )

    # 4. Корректируем смещение при помощи лагранжевого ресэмплинга (добавляем обратное смещение)
    corrected_signal = apply_fractional_shift(misaligned_signal, shift=-timing_offset, half_len=6)
    corrected_constellation = sample_at_symbol_centers(
        corrected_signal, symbol_positions.astype(float), use_lagrange=False
    )

    # Оценка ошибок
    err_misaligned = rms_error(clean_constellation, misaligned_constellation)
    err_corrected = rms_error(clean_constellation, corrected_constellation)

    print(f"Ошибка (misaligned vs reference): {err_misaligned:.4e}")
    print(f"Ошибка (corrected vs reference):  {err_corrected:.4e}")
    print()

    # Сервисы для визуализации
    time_axis = np.arange(len(baseband_clean)) / Fs

    def prepare_slice(center_symbol: int, num_view_symbols: int = 6) -> slice:
        start = max(center_symbol * Sps + filter_delay - Sps, 0)
        end = min((center_symbol + num_view_symbols) * Sps + filter_delay + Sps, len(baseband_clean))
        return slice(start, end)

    slice_indices = prepare_slice(valid_symbol_indices[len(valid_symbol_indices) // 2])

    # Для временных графиков будем отмечать используемые точки сэмплирования
    sample_times = time_axis[symbol_positions]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        "Коррекция дробного смещения выборок (Lagrange Resampling)",
        fontsize=16,
        fontweight="bold",
    )

    # --- Ряд 1: Исходный сигнал ---
    ax_time = axes[0, 0]
    ax_time.plot(time_axis[slice_indices], np.real(baseband_clean[slice_indices]), label="Re", color="tab:blue")
    ax_time.plot(time_axis[slice_indices], np.imag(baseband_clean[slice_indices]), label="Im", color="tab:orange", alpha=0.7)
    ax_time.scatter(sample_times, np.real(clean_constellation), color="black", s=25, zorder=5, label="Выборки символов")
    ax_time.set_title("Исходный baseband (без смещения)")
    ax_time.set_xlabel("Время, с")
    ax_time.set_ylabel("Амплитуда")
    ax_time.grid(alpha=0.3)
    ax_time.legend(loc="upper right")

    ax_spec = axes[0, 1]
    freqs, magnitude = compute_spectrum(baseband_clean, Fs)
    ax_spec.plot(freqs, magnitude, color="tab:green")
    ax_spec.set_title("Спектр (исходный)")
    ax_spec.set_xlabel("Частота, Гц")
    ax_spec.set_ylabel("|S(f)|, dB")
    ax_spec.grid(alpha=0.3)
    ax_spec.set_xlim(-6_000, 6_000)

    ax_const = axes[0, 2]
    ax_const.scatter(clean_constellation.real, clean_constellation.imag, s=20, color="tab:blue", alpha=0.7)
    ax_const.set_title("Созвездие (эталон)")
    ax_const.set_xlabel("Re (I)")
    ax_const.set_ylabel("Im (Q)")
    ax_const.axhline(0, color="grey", linewidth=0.5)
    ax_const.axvline(0, color="grey", linewidth=0.5)
    ax_const.set_aspect("equal", adjustable="box")
    ax_const.set_xlim(-1.5, 1.5)
    ax_const.set_ylim(-1.5, 1.5)

    # --- Ряд 2: Смещённый сигнал ---
    ax_time = axes[1, 0]
    ax_time.plot(time_axis[slice_indices], np.real(misaligned_signal[slice_indices]), color="tab:blue")
    ax_time.plot(time_axis[slice_indices], np.imag(misaligned_signal[slice_indices]), color="tab:orange", alpha=0.7)
    ax_time.scatter(sample_times, np.real(misaligned_constellation), color="red", s=25, zorder=5, label="Выборки (ошибка)")
    ax_time.set_title("Baseband со смещением +0.35 отсчета")
    ax_time.set_xlabel("Время, с")
    ax_time.set_ylabel("Амплитуда")
    ax_time.grid(alpha=0.3)
    ax_time.legend(loc="upper right")

    ax_spec = axes[1, 1]
    freqs, magnitude = compute_spectrum(misaligned_signal, Fs)
    ax_spec.plot(freqs, magnitude, color="tab:green")
    ax_spec.set_title("Спектр (после смещения)")
    ax_spec.set_xlabel("Частота, Гц")
    ax_spec.set_ylabel("|S(f)|, dB")
    ax_spec.grid(alpha=0.3)
    ax_spec.set_xlim(-6_000, 6_000)

    ax_const = axes[1, 2]
    ax_const.scatter(misaligned_constellation.real, misaligned_constellation.imag, s=20, color="red", alpha=0.7)
    ax_const.set_title("Созвездие (без компенсации)")
    ax_const.set_xlabel("Re (I)")
    ax_const.set_ylabel("Im (Q)")
    ax_const.axhline(0, color="grey", linewidth=0.5)
    ax_const.axvline(0, color="grey", linewidth=0.5)
    ax_const.set_aspect("equal", adjustable="box")
    ax_const.set_xlim(-1.5, 1.5)
    ax_const.set_ylim(-1.5, 1.5)

    # --- Ряд 3: Скорректированный сигнал ---
    ax_time = axes[2, 0]
    ax_time.plot(time_axis[slice_indices], np.real(corrected_signal[slice_indices]), color="tab:blue")
    ax_time.plot(time_axis[slice_indices], np.imag(corrected_signal[slice_indices]), color="tab:orange", alpha=0.7)
    ax_time.scatter(sample_times, np.real(corrected_constellation), color="lime", s=25, zorder=5, label="Выборки (после Лагранжа)")
    ax_time.set_title("Baseband после компенсации")
    ax_time.set_xlabel("Время, с")
    ax_time.set_ylabel("Амплитуда")
    ax_time.grid(alpha=0.3)
    ax_time.legend(loc="upper right")

    ax_spec = axes[2, 1]
    freqs, magnitude = compute_spectrum(corrected_signal, Fs)
    ax_spec.plot(freqs, magnitude, color="tab:green")
    ax_spec.set_title("Спектр (после компенсации)")
    ax_spec.set_xlabel("Частота, Гц")
    ax_spec.set_ylabel("|S(f)|, dB")
    ax_spec.grid(alpha=0.3)
    ax_spec.set_xlim(-6_000, 6_000)

    ax_const = axes[2, 2]
    ax_const.scatter(corrected_constellation.real, corrected_constellation.imag, s=20, color="lime", alpha=0.7)
    ax_const.set_title("Созвездие (после Лагранжа)")
    ax_const.set_xlabel("Re (I)")
    ax_const.set_ylabel("Im (Q)")
    ax_const.axhline(0, color="grey", linewidth=0.5)
    ax_const.axvline(0, color="grey", linewidth=0.5)
    ax_const.set_aspect("equal", adjustable="box")
    ax_const.set_xlim(-1.5, 1.5)
    ax_const.set_ylim(-1.5, 1.5)

    for ax in axes.flat:
        ax.tick_params(labelsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


if __name__ == "__main__":
    main()

