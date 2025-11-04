#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QPSK + RRC (roll-off 0.3) синтез квадратурных отсчётов
с нормированной частотной отстройкой и начальной фазой.
Результат сохраняется в .pcm (RAW float32) в формате I/Q (interleaved).

Состав:
1) Генерация случайных QPSK-символов (Gray-кодирование, мощность = 1).
2) Апсемплинг до требуемых "отсчётов на символ" (sps = 5).
3) Формирование импульса RRC-фильтром (α = 0.3).
4) Внесение нормированной частотной отстройки f_off = 0.007 [циклов/отсчет]
   и начальной фазы φ0 = 35°.
5) Сохранение в "qpsk_signal.pcm" как float32 I,Q,I,Q,...

Примечания:
- "Нормированная частота" 0.007 означает 0.007 циклов на ОДИН дискрет (cycles/sample).
  Если бы у нас был реальный частотный масштаб с частотой дискретизации Fs (Гц),
  физический сдвиг по частоте был бы f_Hz = f_off * Fs.
- RRC нормирован по энергии (unit energy), чтобы в среднем мощность сигнала после
  формирования совпадала с мощностью входной последовательности символов.
- Кол-во символов, seed и др. параметры можно менять в блоке "Параметры".
"""

import numpy as np
import math
from typing import Tuple

# --------------------------- Параметры синтеза --------------------------------

# Количество QPSK-символов (длина полезной последовательности до фильтрации)
NUM_SYMBOLS: int = 1000

# Отсчётов на символ (Samples Per Symbol). Влияет на "скорость" дискретизации:
# чем больше SPS, тем больше дискретов на один символ и шире сигнал в сэмплах.
SPS: int = 5  # согласно ТЗ

# RRC (Root Raised Cosine) коэффициент скругления (roll-off).
# α=0 -> узкая полоса, длинный импульс; α=1 -> шире полоса, короче "хвост".
RRC_ALPHA: float = 0.3  # согласно ТЗ

# Длина RRC-импульсной характеристики в символах (суммарный "размах" по времени).
# Итоговое число тапов = SPAN_SYM * SPS + 1 (делаем нечётное для симметрии).
# Чем больше SPAN_SYM, тем точнее аппроксимация теоретического RRC, но длиннее фильтр.
SPAN_SYM: int = 10

# Нормированная частотная отстройка (cycles/sample). 0.007 => на каждый отсчёт
# фаза несущей увеличивается на 2π*0.007 рад. Это эквивалентно умножению
# комплексной огибающей на exp(j*2π*f_off*n).
FREQ_OFFSET_NORM: float = 0.007  # согласно ТЗ

# Начальная фаза несущей (в градусах). Конвертируется в радианы.
PHASE0_DEG: float = 35.0  # согласно ТЗ

# Имя выходного файла .pcm (RAW float32, I/Q interleaved)
OUT_PCMC_FILENAME: str = "qpsk_signal.pcm"

# Флаг построения простых проверочных графиков (созвездие, спектр) — опционально.
# Для чистого синтеза можно выключить, чтобы не тянуть matplotlib.
PLOT: bool = False

# Фиксируем генератор случайных чисел для воспроизводимости (можно убрать)
RNG_SEED: int = 42


# --------------------------- Вспомогательные функции --------------------------

def qpsk_gray_map(symbol_indices: np.ndarray) -> np.ndarray:
    """
    Отображение индексов {0,1,2,3} на точки созвездия QPSK по Gray-коду.
    Мы используем созвездие со средней мощностью = 1 (нормировка на sqrt(2)):

        биты  ->  индекс -> точка
        00    ->    0    -> ( +1 + j*+1 ) / sqrt(2)
        01    ->    1    -> ( -1 + j*+1 ) / sqrt(2)
        11    ->    2    -> ( -1 + j*-1 ) / sqrt(2)
        10    ->    3    -> ( +1 + j*-1 ) / sqrt(2)

    Такой порядок индексов соответствует серому коду (Gray), где соседние точки
    отличаются одним битом — это снижает вероятность битовых ошибок при решении.
    """
    mapping = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex128) / np.sqrt(2.0)
    return mapping[symbol_indices]


def rrc_taps(beta: float, sps: int, span_sym: int) -> np.ndarray:
    """
    Генерация импульсной характеристики RRC (Root Raised Cosine).

    Параметры:
      beta     (α) : roll-off (0..1). 0 => узкая полоса (длинный импульс), 1 => широкая.
      sps           : отсчётов на символ.
      span_sym      : суммарная длина в "символах" по времени (обычно 8..12).

    Возвращает:
      h : numpy-вектор длины (span_sym * sps + 1), симметричный, нечётной длины.

    Формула использует непрерывный прототип с t в единицах длительности символа Ts (=1),
    затем дискретизирует с шагом 1/sps. Особые точки t=0 и t=±1/(4β) обрабатываются
    через предельные значения (устойчивый к численной потере точности вариант).
    """
    assert sps > 0 and isinstance(sps, int), "SPS должен быть положительным целым"
    assert span_sym > 0 and isinstance(span_sym, int), "SPAN_SYM должен быть положительным целым"
    assert 0.0 <= beta <= 1.0, "β (roll-off) должен быть в диапазоне [0, 1]"

    # Временная сетка в единицах Ts (Ts=1): от -span/2 до +span/2 с шагом 1/sps
    # Благодаря "+ 1/sps" правая граница включается, получая нечётное число тапов.
    t = np.arange(-span_sym/2, span_sym/2 + 1/sps, 1/sps, dtype=np.float64)

    h = np.zeros_like(t, dtype=np.float64)
    pi = math.pi
    eps = 1e-12  # численный порог для сравнения с "особой" точкой

    # Обрабатываем общий случай векторно
    # h(t) = [ sin(pi t (1 - β)) + 4 β t cos(pi t (1 + β)) ] / [ pi t (1 - (4 β t)^2) ]
    # Особые точки:
    # 1) t = 0: h(0) = 1 - β + 4β/π
    # 2) t = ±1/(4β): h = (β/√2)*[ (1+2/π)sin(π/(4β)) + (1 - 2/π)cos(π/(4β)) ]
    if beta == 0.0:
        # Предельный случай: Raised Cosine при β→0 вырождается в Sinc.
        # Здесь вернём нормированный sinc (по энергии нормируем позже).
        # h(t) = sin(pi t) / (pi t)
        # Обработаем t=0 отдельно
        h = np.sinc(t)  # np.sinc(x) = sin(pi x) / (pi x)
    else:
        # Общая формула
        denom = (pi * t * (1 - (4 * beta * t) ** 2))
        num = (np.sin(pi * t * (1 - beta)) + 4 * beta * t * np.cos(pi * t * (1 + beta)))
        mask_regular = (np.abs(t) > eps) & (np.abs(np.abs(t) - 1/(4*beta)) > eps)
        h[mask_regular] = num[mask_regular] / denom[mask_regular]

        # t = 0
        mask_zero = (np.abs(t) <= eps)
        h[mask_zero] = 1.0 - beta + (4 * beta / pi)

        # t = ±1/(4β)
        t_special = 1.0 / (4.0 * beta)
        mask_special = np.isclose(np.abs(t), t_special, atol=1e-9)
        if np.any(mask_special):
            h[mask_special] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / pi) * np.sin(pi / (4 * beta)) +
                (1 - 2 / pi) * np.cos(pi / (4 * beta))
            )

    # Нормировка по энергии:
    # После фильтрации средняя мощность примерно сохраняется относительно входных символов.
    h_energy = np.sqrt(np.sum(h ** 2))
    if h_energy > 0:
        h = h / h_energy

    return h


def upsample_by_zeros(x: np.ndarray, sps: int) -> np.ndarray:
    """
    Апсемплинг методом "вставки нулей": между соседними символами добавляем (sps-1) нулей.
    Это готовит последовательность к фильтрации передаточным фильтром (RRC).
    """
    y = np.zeros(len(x) * sps, dtype=x.dtype)
    y[::sps] = x
    return y


def apply_frequency_offset(x: np.ndarray, f_off: float, phi0_deg: float) -> np.ndarray:
    """
    Внесение нормированной частотной отстройки и начальной фазы.

    x         : комплексная огибающая (после RRC), dtype=complex
    f_off     : нормированная частота в cycles/sample (например, 0.007).
                На каждом дискрете фаза растёт на 2π*f_off.
    phi0_deg  : начальная фаза в градусах.

    Возвращает:
      y = x * exp( j * ( 2π f_off n + φ0 ) )
    """
    n = np.arange(len(x), dtype=np.float64)
    phi0 = np.deg2rad(phi0_deg)
    carrier = np.exp(1j * (2 * math.pi * f_off * n + phi0))
    return x * carrier


def save_as_pcm_float32_iq(x: np.ndarray, filename: str) -> None:
    """
    Сохранение комплексной последовательности в RAW .pcm (float32, I/Q interleaved).
    Формат: [I0, Q0, I1, Q1, ...], без заголовка, little-endian.

    Важно: большинство SDR-инструментов ожидают именно такое чередование.
    """
    assert np.iscomplexobj(x), "Ожидался комплексный массив"
    iq = np.empty(2 * len(x), dtype=np.float32)
    iq[0::2] = x.real.astype(np.float32, copy=False)
    iq[1::2] = x.imag.astype(np.float32, copy=False)
    with open(filename, "wb") as f:
        iq.tofile(f)


# --------------------------- Основной сценарий --------------------------------

def main() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Выполняет все этапы синтеза и возвращает ключевые стадии:
    (symbols, upsampled, shaped, signal)

    symbols  : комплексные QPSK-точки до апсемплинга
    upsampled: нулевставленная последовательность (комплексная)
    shaped   : после RRC-фильтра (комплексная огибающая)
    signal   : после внесения частотного сдвига и начальной фазы (комплексная)
    """
    # 0) Инициализация ГСЧ (для воспроизводимости)
    rng = np.random.default_rng(RNG_SEED)

    # 1) Генерация случайных QPSK-символов
    # Каждый символ кодирует 2 бита -> 4 возможных состояния (0..3).
    # "Значение символов: случайное" согласно ТЗ.
    sym_indices = rng.integers(low=0, high=4, size=NUM_SYMBOLS, endpoint=False)
    symbols = qpsk_gray_map(sym_indices)  # мощность ≈ 1

    # 2) Апсемплинг вставкой нулей: теперь длина в сэмплах = NUM_SYMBOLS * SPS
    upsampled = upsample_by_zeros(symbols, SPS)

    # 3) RRC-фильтр: генерируем импульсную характеристику и фильтруем
    h_rrc = rrc_taps(RRC_ALPHA, SPS, SPAN_SYM)
    # Свёртка 'same' даёт выход той же длины, что и upsampled, с групповым запаздыванием
    # ~ (len(h)-1)/2 сэмплов. На концах есть переходные процессы — это нормально.
    shaped = np.convolve(upsampled, h_rrc, mode="same")

    # 4) Внесём нормированную частотную отстройку и начальную фазу
    signal = apply_frequency_offset(shaped, FREQ_OFFSET_NORM, PHASE0_DEG)

    # 5) Сохраняем в .pcm как float32 I/Q interleaved
    save_as_pcm_float32_iq(signal.astype(np.complex64, copy=False), OUT_PCMC_FILENAME)

    # Небольшая сводка по амплитудам — просто для контроля
    pwr_avg = np.mean(np.abs(signal) ** 2)
    print(f"Синтез завершён.")
    print(f"  Символов:            {NUM_SYMBOLS}")
    print(f"  SPS (отсчётов/симв): {SPS}")
    print(f"  RRC alpha:           {RRC_ALPHA}")
    print(f"  RRC taps:            {len(h_rrc)}  (span={SPAN_SYM} символов)")
    print(f"  f_off (norm):        {FREQ_OFFSET_NORM} циклов/отсчёт")
    print(f"  phi0:                {PHASE0_DEG}°")
    print(f"  Средняя мощность:    {pwr_avg:.4f}")
    print(f"  Файл:                {OUT_PCMC_FILENAME}  (RAW float32 I/Q)")

    return symbols, upsampled, shaped, signal


# --------------------------- Визуализация (необязательно) ---------------------

def _maybe_plot(symbols: np.ndarray, shaped: np.ndarray, signal: np.ndarray) -> None:
    """
    Простая визуализация: созвездие (по отсчётам вблизи моментов съёма),
    временная форма и грубый спектр. Включается флагом PLOT.
    """
    import matplotlib.pyplot as plt
    import numpy.fft as fft

    # Созвездие: возьмём отсчёты на решётке символов (компенсируем грубо групповую задержку)
    # Групповая задержка RRC ~ (len(h)-1)/2. Точного h здесь нет, но разумная демонстрация:
    # Мы не знаем h здесь, поэтому оценим задержку через SPS*SPAN_SYM/2 (точнее: (taps-1)/2).
    # Для красоты возьмём центральную часть, чтобы избежать краёв.
    taps = SPAN_SYM * SPS + 1
    gd = (taps - 1) // 2
    start = gd
    stop = len(shaped) - gd
    sym_samples = shaped[start:stop:SPS]
    sig_samples = signal[start:stop:SPS]

    plt.figure()
    plt.scatter(np.real(sig_samples), np.imag(sig_samples), alpha=0.5, s=12)
    plt.title("Созвездие QPSK (после RRC и частотного сдвига)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.axis("equal")
    plt.grid(True)

    # Временная огибающая (модуль)
    plt.figure()
    plt.plot(np.abs(signal))
    plt.title("Модуль комплексной огибающей |s[n]|")
    plt.xlabel("n")
    plt.ylabel("|s[n]|")
    plt.grid(True)

    # Простой спектр (|FFT| в dB)
    S = fft.fftshift(fft.fft(signal * np.hanning(len(signal))))
    f = np.linspace(-0.5, 0.5, len(S), endpoint=False)  # нормированная частота (циклов/отсчёт)
    Sdb = 20 * np.log10(np.maximum(np.abs(S), 1e-12))
    plt.figure()
    plt.plot(f, Sdb)
    plt.title("Амплитудный спектр (норм. частота, циклы/отсчёт)")
    plt.xlabel("Нормированная частота")
    plt.ylabel("|S(f)|, dB")
    plt.grid(True)

    plt.show()


# --------------------------- Точка входа --------------------------------------

if __name__ == "__main__":
    symbols_, upsampled_, shaped_, signal_ = main()
    if PLOT:
        _maybe_plot(symbols_, shaped_, signal_)