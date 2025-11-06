"""
Функции для анализа и построения спектров сигналов
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


def compute_spectrum(signal: np.ndarray,
                     sample_rate: float = 1.0,
                     nfft: Optional[int] = None,
                     window: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисление спектра сигнала с помощью FFT.
    
    Параметры:
    ----------
    signal : np.ndarray
        Входной сигнал (может быть комплексным или вещественным)
    sample_rate : float
        Частота дискретизации (по умолчанию 1.0 - нормализованная)
    nfft : int, optional
        Размер FFT (если None, используется длина сигнала)
    window : str, optional
        Оконная функция: 'hann', 'hamming', 'blackman', None
        
    Возвращает:
    -----------
    freq : np.ndarray
        Массив частот
    spectrum : np.ndarray
        Комплексный спектр
    """
    
    # Применение оконной функции
    signal_windowed = signal.copy()
    if window is not None:
        if window == 'hann':
            win = np.hanning(len(signal))
        elif window == 'hamming':
            win = np.hamming(len(signal))
        elif window == 'blackman':
            win = np.blackman(len(signal))
        else:
            win = np.ones(len(signal))
        signal_windowed = signal * win
    
    # Вычисление FFT
    if nfft is None:
        nfft = len(signal)
    
    spectrum = np.fft.fftshift(np.fft.fft(signal_windowed, n=nfft))
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0/sample_rate))
    
    return freq, spectrum


def plot_spectrum(signal: np.ndarray,
                  sample_rate: float = 1.0,
                  title: str = 'Спектр сигнала',
                  figsize: Tuple[int, int] = (14, 6),
                  scale: str = 'db',
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  nfft: Optional[int] = None,
                  window: Optional[str] = None,
                  show: bool = True,
                  save_path: Optional[str] = None,
                  ax: Optional[plt.Axes] = None):
    """
    Построение спектра сигнала с помощью FFT.
    
    Параметры:
    ----------
    signal : np.ndarray
        Входной сигнал (может быть комплексным или вещественным)
    sample_rate : float
        Частота дискретизации (по умолчанию 1.0 - нормализованная)
    title : str
        Заголовок графика
    figsize : tuple
        Размер фигуры
    scale : str
        'db' - логарифмическая шкала (дБ), 'linear' - линейная, 'power' - мощность
    xlim : tuple, optional
        Пределы по оси X (частота)
    ylim : tuple, optional
        Пределы по оси Y (мощность)
    nfft : int, optional
        Размер FFT (если None, используется длина сигнала)
    window : str, optional
        Оконная функция: 'hann', 'hamming', 'blackman', None
    show : bool
        Показывать ли график
    save_path : str, optional
        Путь для сохранения
    ax : plt.Axes, optional
        Существующая ось для построения (если None, создается новая)
        
    Возвращает:
    -----------
    fig, ax, freq, spectrum : фигура, ось, частоты, спектр
    
    Примеры:
    --------
    # Простое использование
    plot_spectrum(signal_iq, sample_rate=1.0, title='Спектр QPSK')
    
    # С окном и ограничением частот
    plot_spectrum(signal_iq, sample_rate=4.0, window='hann', 
                  xlim=(-0.5, 0.5), scale='db')
    
    # Несколько спектров на одном графике
    fig, ax = plt.subplots()
    plot_spectrum(signal1, ax=ax, show=False, title='Сравнение спектров')
    plot_spectrum(signal2, ax=ax, show=True)
    """
    
    # Вычисление спектра
    freq, spectrum = compute_spectrum(signal, sample_rate, nfft, window)
    
    # Масштабирование
    spectrum_mag = np.abs(spectrum)
    
    if scale == 'db':
        # Избегаем log(0)
        spectrum_mag = 20 * np.log10(spectrum_mag + 1e-10)
        ylabel = 'Мощность (дБ)'
    elif scale == 'power':
        spectrum_mag = spectrum_mag ** 2
        ylabel = 'Мощность'
    else:  # linear
        ylabel = 'Амплитуда'
    
    # Построение
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.plot(freq, spectrum_mag, linewidth=1, alpha=0.8)
    
    xlabel = 'Частота (Гц)' if sample_rate != 1.0 else 'Нормализованная частота'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График спектра сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax, freq, spectrum


def plot_psd(signal: np.ndarray,
             sample_rate: float = 1.0,
             title: str = 'Спектральная плотность мощности',
             figsize: Tuple[int, int] = (14, 6),
             nfft: int = 1024,
             noverlap: Optional[int] = None,
             window: str = 'hann',
             show: bool = True,
             save_path: Optional[str] = None):
    """
    Построение спектральной плотности мощности (PSD) методом Уэлча.
    
    Параметры:
    ----------
    signal : np.ndarray
        Входной сигнал
    sample_rate : float
        Частота дискретизации
    title : str
        Заголовок графика
    figsize : tuple
        Размер фигуры
    nfft : int
        Размер FFT для каждого сегмента
    noverlap : int, optional
        Количество перекрывающихся отсчетов (по умолчанию nfft//2)
    window : str
        Оконная функция
    show : bool
        Показывать ли график
    save_path : str, optional
        Путь для сохранения
        
    Возвращает:
    -----------
    fig, ax, freq, psd : фигура, ось, частоты, PSD
    """
    
    if noverlap is None:
        noverlap = nfft // 2
    
    # Вычисление PSD методом Уэлча
    from matplotlib.mlab import psd as mlab_psd
    
    psd, freq = mlab_psd(signal, NFFT=nfft, Fs=sample_rate, 
                         window=np.hanning(nfft), noverlap=noverlap)
    
    # Для комплексных сигналов центрируем частоты
    if np.iscomplexobj(signal):
        freq = np.fft.fftshift(freq)
        psd = np.fft.fftshift(psd)
    
    # Построение
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq, 10 * np.log10(psd + 1e-10), linewidth=1)
    
    xlabel = 'Частота (Гц)' if sample_rate != 1.0 else 'Нормализованная частота'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Мощность/Частота (дБ/Гц)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График PSD сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax, freq, psd


def plot_spectrogram(signal: np.ndarray,
                     sample_rate: float = 1.0,
                     title: str = 'Спектрограмма',
                     figsize: Tuple[int, int] = (14, 8),
                     nfft: int = 256,
                     noverlap: Optional[int] = None,
                     window: str = 'hann',
                     cmap: str = 'viridis',
                     show: bool = True,
                     save_path: Optional[str] = None):
    """
    Построение спектрограммы сигнала.
    
    Параметры:
    ----------
    signal : np.ndarray
        Входной сигнал
    sample_rate : float
        Частота дискретизации
    title : str
        Заголовок графика
    figsize : tuple
        Размер фигуры
    nfft : int
        Размер FFT
    noverlap : int, optional
        Количество перекрывающихся отсчетов
    window : str
        Оконная функция
    cmap : str
        Цветовая карта ('viridis', 'jet', 'hot', 'cool', и т.д.)
    show : bool
        Показывать ли график
    save_path : str, optional
        Путь для сохранения
        
    Возвращает:
    -----------
    fig, ax, spec, freqs, t : фигура, ось, спектрограмма, частоты, время
    """
    
    if noverlap is None:
        noverlap = nfft // 2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    spec, freqs, t, im = ax.specgram(signal, NFFT=nfft, Fs=sample_rate,
                                      noverlap=noverlap, cmap=cmap,
                                      scale='dB')
    
    xlabel = 'Время (с)' if sample_rate != 1.0 else 'Время (отсчеты)'
    ylabel = 'Частота (Гц)' if sample_rate != 1.0 else 'Нормализованная частота'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Мощность (дБ)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Спектрограмма сохранена в: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax, spec, freqs, t


def compare_spectra(signals: list,
                    labels: list,
                    sample_rate: float = 1.0,
                    title: str = 'Сравнение спектров',
                    figsize: Tuple[int, int] = (14, 6),
                    scale: str = 'db',
                    window: Optional[str] = 'hann',
                    xlim: Optional[Tuple[float, float]] = None,
                    show: bool = True,
                    save_path: Optional[str] = None):
    """
    Сравнение спектров нескольких сигналов на одном графике.
    
    Параметры:
    ----------
    signals : list of np.ndarray
        Список сигналов для сравнения
    labels : list of str
        Подписи для каждого сигнала
    sample_rate : float
        Частота дискретизации
    title : str
        Заголовок графика
    figsize : tuple
        Размер фигуры
    scale : str
        Масштаб ('db', 'linear', 'power')
    window : str, optional
        Оконная функция
    xlim : tuple, optional
        Пределы по оси X
    show : bool
        Показывать ли график
    save_path : str, optional
        Путь для сохранения
        
    Возвращает:
    -----------
    fig, ax : фигура и ось
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for signal, label in zip(signals, labels):
        freq, spectrum = compute_spectrum(signal, sample_rate, window=window)
        
        spectrum_mag = np.abs(spectrum)
        
        if scale == 'db':
            spectrum_mag = 20 * np.log10(spectrum_mag + 1e-10)
            ylabel = 'Мощность (дБ)'
        elif scale == 'power':
            spectrum_mag = spectrum_mag ** 2
            ylabel = 'Мощность'
        else:
            ylabel = 'Амплитуда'
        
        ax.plot(freq, spectrum_mag, linewidth=1.5, alpha=0.7, label=label)
    
    xlabel = 'Частота (Гц)' if sample_rate != 1.0 else 'Нормализованная частота'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    if xlim:
        ax.set_xlim(xlim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сравнения сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


# Пример использования
if __name__ == "__main__":
    # Генерация тестового сигнала
    t = np.linspace(0, 1, 1000, endpoint=False)
    sample_rate = 1000  # Гц
    
    # QPSK-подобный сигнал
    signal = np.exp(1j * 2 * np.pi * 50 * t)  # Несущая 50 Гц
    signal += 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))  # Шум
    
    print("Примеры использования функций спектрального анализа:\n")
    
    # 1. Простой спектр
    print("1. Построение спектра")
    plot_spectrum(signal, sample_rate=sample_rate, 
                  title='Спектр QPSK сигнала',
                  window='hann',
                  show=True)
    
    # 2. Спектральная плотность мощности
    print("\n2. PSD (Spectral Power Density)")
    plot_psd(signal, sample_rate=sample_rate,
             title='PSD методом Уэлча',
             show=True)
    
    # 3. Спектрограмма
    print("\n3. Спектрограмма")
    plot_spectrogram(signal, sample_rate=sample_rate,
                     title='Спектрограмма сигнала',
                     show=True)
    
    # 4. Сравнение спектров
    print("\n4. Сравнение спектров")
    signal2 = np.exp(1j * 2 * np.pi * 100 * t)  # Другая несущая
    signal2 += 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    
    compare_spectra([signal, signal2],
                    labels=['Сигнал 1 (50 Гц)', 'Сигнал 2 (100 Гц)'],
                    sample_rate=sample_rate,
                    title='Сравнение двух сигналов',
                    show=True)

