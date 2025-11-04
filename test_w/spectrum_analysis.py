#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Расчёт и отображение амплитудно-частотного спектра сигнала
Пункт 2 ТЗ
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_spectrum(signal: np.ndarray, window: str = "hann") -> tuple:
    """
    Расчёт амплитудно-частотного спектра
    
    Args:
        signal: комплексный сигнал
        window: тип окна ("hann", "hamming", "blackman" или None)
    
    Returns:
        f_norm: нормированные частоты [-0.5, 0.5]
        spectrum_db: амплитудный спектр в дБ
    """
    # Выбор оконной функции
    if window == "hann":
        w = np.hanning(len(signal))
    elif window == "hamming":
        w = np.hamming(len(signal))
    elif window == "blackman":
        w = np.blackman(len(signal))
    else:
        w = np.ones(len(signal))
    
    # Применение окна и БПФ
    windowed_signal = signal * w
    spectrum = np.fft.fftshift(np.fft.fft(windowed_signal))
    
    # Амплитудный спектр в дБ
    magnitude = np.abs(spectrum)
    spectrum_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    
    # Нормированные частоты
    f_norm = np.linspace(-0.5, 0.5, len(spectrum), endpoint=False)
    
    return f_norm, spectrum_db


def plot_spectrum(f_norm: np.ndarray, spectrum_db: np.ndarray, 
                 title: str = "Амплитудно-частотный спектр", 
                 save_fig: str = None):
    """
    Отображение амплитудно-частотного спектра
    
    Args:
        f_norm: нормированные частоты
        spectrum_db: амплитудный спектр в дБ
        title: заголовок графика
        save_fig: имя файла для сохранения (опционально)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(f_norm, spectrum_db, linewidth=1.0)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Нормированная частота (cycles/sample)')
    plt.ylabel('Амплитуда (дБ)')
    plt.title(title)
    plt.xlim(-0.5, 0.5)
    
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_fig}")
    
    plt.tight_layout()
    plt.show()


def load_iq_signal(filename: str, dtype=np.complex64) -> np.ndarray:
    """
    Загрузка I/Q сигнала из файла
    
    Args:
        filename: имя файла
        dtype: тип данных (complex64 для float32 I/Q, int16 для PCM)
    
    Returns:
        signal: комплексный сигнал
    """
    if dtype == np.complex64:
        # float32 I/Q interleaved
        raw = np.fromfile(filename, dtype=np.float32)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        signal = raw[::2] + 1j * raw[1::2]
    else:
        # int16 I/Q interleaved
        raw = np.fromfile(filename, dtype=np.int16)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        signal = (raw[::2] + 1j * raw[1::2]) / 32768.0
    
    return signal.astype(np.complex64)


def analyze_signal_spectrum(filename: str, window: str = "hann", 
                          title: str = None, save_fig: str = None):
    """
    Полный анализ спектра сигнала из файла
    
    Args:
        filename: имя файла с сигналом
        window: тип окна
        title: заголовок графика
        save_fig: имя файла для сохранения графика
    """
    try:
        # Попробуем загрузить как float32 I/Q
        signal = load_iq_signal(filename, dtype=np.complex64)
        print(f"Загружен сигнал: {filename} (float32 I/Q)")
    except:
        try:
            # Попробуем загрузить как int16 I/Q
            signal = load_iq_signal(filename, dtype=np.int16)
            print(f"Загружен сигнал: {filename} (int16 I/Q)")
        except Exception as e:
            print(f"Ошибка загрузки файла {filename}: {e}")
            return None, None
    
    print(f"Длина сигнала: {len(signal)} отсчётов")
    print(f"Средняя мощность: {np.mean(np.abs(signal)**2):.6f}")
    
    # Расчёт спектра
    f_norm, spectrum_db = compute_spectrum(signal, window)
    
    # Отображение
    if title is None:
        title = f"Амплитудно-частотный спектр: {filename}"
    
    plot_spectrum(f_norm, spectrum_db, title, save_fig)
    
    return f_norm, spectrum_db


# Основной код для демонстрации
if __name__ == "__main__":
    # Анализ синтезированного QPSK сигнала
    print("=== Анализ амплитудно-частотного спектра ===")
    
    # Попробуем проанализировать файл qpsk_signal.pcm
    f_norm, spectrum_db = analyze_signal_spectrum(
        "qpsk_signal.pcm", 
        window="hann",
        title="Спектр синтезированного QPSK сигнала",
        save_fig="qpsk_spectrum.png"
    )
    
    if f_norm is not None:
        # Дополнительная статистика
        peak_idx = np.argmax(spectrum_db)
        peak_freq = f_norm[peak_idx]
        peak_power = spectrum_db[peak_idx]
        
        print(f"\nСтатистика спектра:")
        print(f"  Пиковая частота: {peak_freq:+.6f}")
        print(f"  Пиковая мощность: {peak_power:.2f} дБ")
        print(f"  Динамический диапазон: {peak_power - np.min(spectrum_db):.2f} дБ")
