#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Обработка полосового сигнала из файла samples.pcm
Пункт 4 ТЗ: чтение I/Q int16, оценка границ, перенос на 0, фильтрация, анализ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def read_iq_int16(filename: str) -> np.ndarray:
    """Чтение I/Q данных из файла (int16 interleaved)"""
    raw_data = np.fromfile(filename, dtype=np.int16)
    if len(raw_data) % 2 != 0:
        raw_data = raw_data[:-1]
    
    # I, Q, I, Q, ... -> комплексный сигнал
    complex_data = raw_data[::2] + 1j * raw_data[1::2]
    # Нормализация
    complex_data = complex_data / np.max(np.abs(complex_data))
    
    return complex_data


def average_spectrum(data: np.ndarray, n_fft: int = 1024, overlap_ratio: float = 0.5):
    """Усреднение спектра методом Уэлча"""
    window = np.hamming(n_fft)
    noverlap = int(round(n_fft * overlap_ratio))
    step = n_fft - noverlap
    nsegments = int(np.floor((len(data) - noverlap) / step))
    
    psd_sum = np.zeros(n_fft)
    
    for i in range(nsegments):
        start_idx = i * step
        end_idx = start_idx + n_fft
        
        if end_idx > len(data):
            break
        
        segment = data[start_idx:end_idx]
        windowed_segment = segment * window
        spectrum = np.fft.fft(windowed_segment, n_fft)
        psd = np.abs(spectrum)**2
        psd_sum += psd
    
    psd_avg = psd_sum / nsegments
    psd_avg_db = 10 * np.log10(psd_avg + np.finfo(float).eps)
    psd_avg_db = np.fft.fftshift(psd_avg_db)
    
    f_rel = np.linspace(-0.5, 0.5, n_fft)
    return f_rel, psd_avg_db


def estimate_signal_bounds(f_rel: np.ndarray, psd_db: np.ndarray, noise_threshold_db: float = 3):
    """Оценка частотных границ полосового сигнала"""
    # Уровень шума (медиана спектра)
    noise_floor = np.median(psd_db)
    threshold = noise_floor + noise_threshold_db
    
    # Поиск границ сигнала
    signal_start = -1
    signal_end = -1
    
    # Слева направо - первое превышение порога
    for i in range(len(psd_db)):
        if psd_db[i] > threshold:
            signal_start = i
            break
    
    # Справа налево - последнее превышение порога
    for i in range(len(psd_db) - 1, -1, -1):
        if psd_db[i] > threshold:
            signal_end = i
            break
    
    if signal_start == -1 or signal_end == -1:
        raise ValueError("Сигнал не обнаружен! Попробуйте уменьшить порог.")
    
    f_start = f_rel[signal_start]
    f_end = f_rel[signal_end]
    center_freq = (f_start + f_end) / 2
    bandwidth = abs(f_end - f_start)
    
    return {
        'f_start': f_start,
        'f_end': f_end,
        'center_freq': center_freq,
        'bandwidth': bandwidth,
        'noise_floor': noise_floor,
        'threshold': threshold
    }


def freq_shift_to_baseband(data: np.ndarray, center_freq: float) -> np.ndarray:
    """Первичный частотный перенос на нулевую частоту"""
    n = np.arange(len(data))
    # Перенос центральной частоты в ноль
    shifted = data * np.exp(-1j * 2 * np.pi * center_freq * n)
    return shifted


def lowpass_filter(data: np.ndarray, cutoff_freq: float, numtaps: int = 101) -> np.ndarray:
    """Низкочастотная фильтрация"""
    # Проектирование FIR НЧ фильтра
    h = signal.firwin(numtaps, cutoff_freq, window='hamming')
    # Фильтрация
    filtered = np.convolve(data, h, mode='same')
    return filtered


def plot_spectrum(f_rel: np.ndarray, psd_db: np.ndarray, title: str = "Спектр"):
    """Отображение амплитудно-частотного спектра"""
    plt.figure(figsize=(10, 6))
    plt.plot(f_rel, psd_db, linewidth=1.0)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Нормированная частота')
    plt.ylabel('Мощность (дБ)')
    plt.title(title)
    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.show()


def analyze_signal(filename: str = 'samples.pcm'):
    """Полный анализ сигнала согласно ТЗ"""
    print(f"=== Анализ файла: {filename} ===")
    
    # 1. Чтение I/Q данных (int16 interleaved)
    print("1. Чтение I/Q данных...")
    try:
        data = read_iq_int16(filename)
        print(f"   Прочитано {len(data)} комплексных отсчетов")
    except FileNotFoundError:
        print(f"   Ошибка: файл {filename} не найден")
        return
    
    # 2. Усреднение спектра и оценка границ
    print("2. Оценка частотных границ полосового сигнала...")
    f_rel, psd_db = average_spectrum(data)
    bounds = estimate_signal_bounds(f_rel, psd_db)
    
    print(f"   Центральная частота: {bounds['center_freq']:.4f}")
    print(f"   Полоса сигнала: {bounds['bandwidth']:.4f}")
    print(f"   Границы: [{bounds['f_start']:.4f}, {bounds['f_end']:.4f}]")
    
    # 3. Отображение исходного спектра
    print("3. Отображение исходного спектра...")
    plot_spectrum(f_rel, psd_db, "Исходный спектр сигнала")
    
    # 4. Первичный частотный перенос на нулевую частоту
    print("4. Частотный перенос на нулевую частоту...")
    data_baseband = freq_shift_to_baseband(data, bounds['center_freq'])
    
    # Спектр после переноса
    f_bb, psd_bb = average_spectrum(data_baseband)
    plot_spectrum(f_bb, psd_bb, "Спектр после переноса на нулевую частоту")
    
    # 5. Частотная фильтрация
    print("5. Низкочастотная фильтрация...")
    # Полоса фильтра чуть шире половины полосы сигнала
    cutoff = min(0.4, bounds['bandwidth'] * 0.6)
    data_filtered = lowpass_filter(data_baseband, cutoff)
    
    # Спектр после фильтрации
    f_filt, psd_filt = average_spectrum(data_filtered)
    plot_spectrum(f_filt, psd_filt, "Спектр после НЧ фильтрации")
    
    # 6. Оценка символьной скорости
    print("6. Оценка символьной скорости...")
    # Приблизительная оценка: Rs ≈ BW / (1 + α), где α ≈ 0.3 для RRC
    symbol_rate = bounds['bandwidth'] / 1.3
    print(f"   Символьная скорость: {symbol_rate:.4f}")
    
    # Итоговые результаты
    print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    print(f"Центральная частота: {bounds['center_freq']:+.4f}")
    print(f"Полоса сигнала: {bounds['bandwidth']:.4f}")
    print(f"Символьная скорость: {symbol_rate:.4f}")
    print(f"Уровень шума: {bounds['noise_floor']:.2f} дБ")
    
    return {
        'center_freq': bounds['center_freq'],
        'bandwidth': bounds['bandwidth'],
        'symbol_rate': symbol_rate,
        'data_filtered': data_filtered
    }


if __name__ == "__main__":
    analyze_signal('samples.pcm')



