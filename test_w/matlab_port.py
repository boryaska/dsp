#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для анализа комплексного PCM файла int16 с усреднением спектра
Простой алгоритм: усредняем спектр, находим первое превышение порога шума
Портирован с MATLAB
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ================== НАСТРОЙКИ ==================
FILENAME = 'samples.pcm'      # Имя PCM файла
NOISE_THRESHOLD_DB = 3        # Порог выше шума для обнаружения сигнала (дБ)
N_FFT = 1024                  # Размер БПФ для усреднения
OVERLAP_RATIO = 0.5           # Коэффициент перекрытия окон (0-1)


def average_spectrum(data, n_fft, overlap_ratio):
    """
    Усреднение спектра методом Уэлча
    
    Args:
        data: комплексный сигнал
        n_fft: размер БПФ
        overlap_ratio: коэффициент перекрытия (0-1)
    
    Returns:
        psd_avg_db: усредненный спектр в дБ
        f_rel: относительные частоты
    """
    # Создаем окно Хэмминга
    window = np.hamming(n_fft)
    
    # Вычисляем количество перекрытия
    noverlap = int(round(n_fft * overlap_ratio))
    step = n_fft - noverlap
    
    # Вычисляем количество сегментов
    nsegments = int(np.floor((len(data) - noverlap) / step))
    
    print(f'Усреднение {nsegments} сегментов...')
    
    # Инициализируем суммарный спектр
    psd_sum = np.zeros(n_fft)
    
    # Обрабатываем каждый сегмент
    for i in range(nsegments):
        start_idx = i * step
        end_idx = start_idx + n_fft
        
        if end_idx > len(data):
            break
        
        # Извлекаем сегмент данных
        segment = data[start_idx:end_idx]
        
        # Применяем окно
        windowed_segment = segment * window
        
        # Вычисляем БПФ и спектральную плотность мощности
        spectrum = np.fft.fft(windowed_segment, n_fft)
        psd = np.abs(spectrum)**2
        
        # Суммируем
        psd_sum += psd
    
    # Усредняем
    psd_avg = psd_sum / nsegments
    
    # Преобразуем в дБ и делаем fftshift
    psd_avg_db = 10 * np.log10(psd_avg + np.finfo(float).eps)
    psd_avg_db = np.fft.fftshift(psd_avg_db)
    
    # Относительные частоты
    f_rel = np.linspace(-0.5, 0.5, n_fft)
    
    return psd_avg_db, f_rel


def main():
    print(f'=== Анализ PCM файла: {FILENAME} ===')
    
    # ================== ЧТЕНИЕ ФАЙЛА ==================
    print('Чтение файла...')
    
    try:
        # Чтение комплексных данных int16
        raw_data = np.fromfile(FILENAME, dtype=np.int16)
    except FileNotFoundError:
        print(f'Ошибка: Не удалось открыть файл: {FILENAME}')
        return
    
    # Преобразование в комплексный сигнал
    if len(raw_data) % 2 != 0:
        raw_data = raw_data[:-1]
    
    complex_data = raw_data[::2] + 1j * raw_data[1::2]
    N = len(complex_data)
    print(f'Прочитано {N} комплексных отсчетов')
    
    # Нормализация данных
    complex_data = complex_data / np.max(np.abs(complex_data))
    
    # ================== УСРЕДНЕНИЕ СПЕКТРА ==================
    print('Усреднение спектра...')
    psd_avg_db, f_rel = average_spectrum(complex_data, N_FFT, OVERLAP_RATIO)
    
    # Оценка уровня шума (медиана всего спектра)
    noise_floor = np.median(psd_avg_db)
    print(f'Уровень шума: {noise_floor:.2f} дБ')
    
    # Порог для обнаружения сигнала
    threshold = noise_floor + NOISE_THRESHOLD_DB
    print(f'Порог обнаружения: {threshold:.2f} дБ')
    
    # ================== ПОИСК ГРАНИЦ СИГНАЛА ==================
    signal_start = -1
    signal_end = -1
    
    # Идем слева направо, ищем первое превышение порога
    for i in range(N_FFT):
        if psd_avg_db[i] > threshold:
            signal_start = i
            break
    
    # Идем справа налево, ищем последнее превышение порога
    for i in range(N_FFT - 1, -1, -1):
        if psd_avg_db[i] > threshold:
            signal_end = i
            break
    
    # Проверяем, найден ли сигнал
    if signal_start == -1 or signal_end == -1:
        print('Ошибка: Сигнал не обнаружен! Попробуйте уменьшить порог.')
        return
    
    print(f'Начало полосы: индекс {signal_start} ({f_rel[signal_start]:.4f})')
    print(f'Конец полосы: индекс {signal_end} ({f_rel[signal_end]:.4f})')
    
    # Вычисляем центральную частоту (среднее между началом и концом)
    center_freq = (f_rel[signal_start] + f_rel[signal_end]) / 2
    print(f'Центральная частота: {center_freq:.4f}')
    
    # Вычисляем полосу сигнала
    bandwidth = abs(f_rel[signal_end] - f_rel[signal_start])
    print(f'Полоса сигнала: {bandwidth:.4f}')
    
    # Оцениваем символьную скорость (примерно равна полосе сигнала)
    symbol_rate = bandwidth
    print(f'Символьная скорость: {symbol_rate:.4f}')
    
    # Максимальная мощность в полосе сигнала
    signal_power = np.max(psd_avg_db[signal_start:signal_end+1])
    snr = signal_power - noise_floor
    print(f'Макс. мощность сигнала: {signal_power:.2f} дБ')
    print(f'ОСШ: {snr:.2f} дБ')
    
    # ================== ВИЗУАЛИЗАЦИЯ ==================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Спектр с выделенной полосой сигнала
    ax1.plot(f_rel, psd_avg_db, 'b', linewidth=1, label='Спектр')
    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Порог')
    ax1.plot(f_rel[signal_start:signal_end+1], psd_avg_db[signal_start:signal_end+1], 
             'g', linewidth=2, label='Полоса сигнала')
    ax1.axvline(x=center_freq, color='m', linestyle='--', linewidth=2, label='Центр. частота')
    ax1.plot(f_rel[signal_start], psd_avg_db[signal_start], 'ko', 
             markersize=8, linewidth=2, label='Начало')
    ax1.plot(f_rel[signal_end], psd_avg_db[signal_end], 'ks', 
             markersize=8, linewidth=2, label='Конец')
    
    ax1.set_xlabel('Относительная частота')
    ax1.set_ylabel('Мощность (дБ)')
    ax1.set_title('Усредненный спектр сигнала с обнаруженной полосой')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim([-0.5, 0.5])
    
    # Увеличенный вид полосы сигнала
    ax2.plot(f_rel, psd_avg_db, 'b', linewidth=1)
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    ax2.plot(f_rel[signal_start:signal_end+1], psd_avg_db[signal_start:signal_end+1], 
             'g', linewidth=2)
    ax2.axvline(x=center_freq, color='m', linestyle='--', linewidth=2)
    ax2.plot(f_rel[signal_start], psd_avg_db[signal_start], 'ko', markersize=8, linewidth=2)
    ax2.plot(f_rel[signal_end], psd_avg_db[signal_end], 'ks', markersize=8, linewidth=2)
    
    ax2.set_xlabel('Относительная частота')
    ax2.set_ylabel('Мощность (дБ)')
    ax2.set_title('Увеличенный вид полосы сигнала')
    ax2.grid(True)
    
    # Устанавливаем границы для увеличенного вида
    margin = 0.02
    ax2.set_xlim([f_rel[signal_start] - margin, f_rel[signal_end] + margin])
    
    plt.tight_layout()
    plt.show()
    
    # ================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==================
    base_name = os.path.splitext(FILENAME)[0]
    results_filename = f'{base_name}_results.txt'
    
    try:
        with open(results_filename, 'w') as f:
            f.write(f'Результаты анализа файла: {FILENAME}\n')
            f.write(f'Дата анализа: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Размер БПФ: {N_FFT}\n')
            f.write(f'Перекрытие окон: {OVERLAP_RATIO:.2f}\n')
            f.write(f'Уровень шума: {noise_floor:.2f} дБ\n')
            f.write(f'Порог обнаружения: {threshold:.2f} дБ\n')
            f.write(f'Начало полосы: {f_rel[signal_start]:.4f}\n')
            f.write(f'Конец полосы: {f_rel[signal_end]:.4f}\n')
            f.write(f'Центральная частота: {center_freq:.4f}\n')
            f.write(f'Полоса сигнала: {bandwidth:.4f}\n')
            f.write(f'Символьная скорость: {symbol_rate:.4f}\n')
            f.write(f'Макс. мощность сигнала: {signal_power:.2f} дБ\n')
            f.write(f'ОСШ: {snr:.2f} дБ\n')
        
        print(f'\nРезультаты сохранены в файл: {results_filename}')
        
    except Exception as e:
        print(f'Ошибка при сохранении результатов: {e}')


if __name__ == "__main__":
    main()
