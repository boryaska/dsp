#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Синтез QPSK сигнала строго по ТЗ
Параметры: QPSK, 5 отсчетов/символ, случайные символы, 
частотная отстройка 0.007, фаза 35°, RRC α=0.3
"""

import numpy as np
from scipy import signal


def rrc_filter(alpha: float, sps: int, span_symbols: int = 10) -> np.ndarray:
    """RRC фильтр"""
    t = np.arange(-span_symbols/2, span_symbols/2 + 1/sps, 1/sps)
    h = np.zeros_like(t)
    eps = 1e-12

    # t = 0
    zero_idx = np.abs(t) < eps
    h[zero_idx] = 1 - alpha + 4*alpha/np.pi

    # t = ±1/(4α)
    if alpha > 0:
        special_idx = np.isclose(np.abs(t), 1/(4*alpha), atol=1e-12)
        if np.any(special_idx):
            h[special_idx] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )

    # Остальные точки
    normal_idx = ~(zero_idx | (special_idx if alpha > 0 else np.zeros_like(t, dtype=bool)))
    if np.any(normal_idx):
        tn = t[normal_idx]
        h[normal_idx] = (
            np.sin(np.pi*tn*(1-alpha)) + 4*alpha*tn*np.cos(np.pi*tn*(1+alpha))
        ) / (
            np.pi*tn*(1 - (4*alpha*tn)**2)
        )

    # Нормировка
    h = h / np.sqrt(np.sum(h**2))
    return h


def qpsk_modulate(num_symbols: int = 1000, seed: int = 42) -> np.ndarray:
    """Синтез QPSK сигнала по ТЗ"""
    # Параметры по ТЗ
    sps = 5                    # отсчетов на символ
    alpha = 0.3               # RRC roll-off
    freq_offset = 0.007       # нормированная частотная отстройка
    phase_deg = 35.0          # начальная фаза несущей

    # Генерация случайных символов
    rng = np.random.default_rng(seed)
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)  # QPSK
    sym_idx = rng.integers(0, 4, size=num_symbols)
    symbols = constellation[sym_idx]

    # RRC формирующий фильтр
    h_rrc = rrc_filter(alpha, sps)

    # Формирование импульсов
    shaped = signal.upfirdn(h_rrc, symbols, up=sps)
    
    # Компенсация групповой задержки
    gd = (len(h_rrc) - 1) // 2
    shaped = shaped[gd:gd + num_symbols * sps]

    # Модуляция на несущую с отстройкой и фазой
    n = np.arange(len(shaped))
    carrier = np.exp(1j * (2*np.pi*freq_offset*n + np.deg2rad(phase_deg)))
    
    return shaped * carrier


def save_iq_pcm(signal: np.ndarray, filename: str = "qpsk_signal.pcm"):
    """Сохранение I/Q в PCM float32"""
    iq = np.empty(2 * len(signal), dtype=np.float32)
    iq[0::2] = signal.real
    iq[1::2] = signal.imag
    with open(filename, "wb") as f:
        iq.tofile(f)


# Основной код
if __name__ == "__main__":
    # Синтез QPSK сигнала
    qpsk_signal = qpsk_modulate(num_symbols=1000, seed=42)
    
    # Сохранение в файл
    save_iq_pcm(qpsk_signal)
    
    # Статистика
    print(f"Синтезирован QPSK сигнал:")
    print(f"  Длина: {len(qpsk_signal)} отсчетов")
    print(f"  Средняя мощность: {np.mean(np.abs(qpsk_signal)**2):.4f}")
    print(f"  Сохранен в: qpsk_signal.pcm")
