"""
Универсальные функции для построения графиков при анализе сигналов
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional


def plot_signal(data: Union[np.ndarray, List[np.ndarray]], 
                titles: Union[str, List[str]] = None,
                xlabel: str = 'Отсчет',
                ylabel: Union[str, List[str]] = 'Амплитуда',
                plot_type: str = 'line',
                figsize: Tuple[int, int] = (14, 6),
                save_path: Optional[str] = None,
                show: bool = True,
                grid: bool = True,
                markers: Optional[List[dict]] = None,
                layout: Optional[Tuple[int, int]] = None,
                **kwargs):
    """
    Универсальная функция для построения графиков сигналов.
    
    Параметры:
    ----------
    data : np.ndarray или list of np.ndarray
        Данные для отображения. Может быть:
        - одномерный массив (один график)
        - список массивов (несколько графиков)
        - комплексный массив (будет разделен на I/Q или модуль/фазу)
    
    titles : str или list of str, optional
        Заголовок(и) графика(ов)
    
    xlabel : str
        Подпись оси X
    
    ylabel : str или list of str
        Подпись(и) оси Y
    
    plot_type : str
        Тип графика: 'line', 'scatter', 'stem', 'abs' (модуль), 
        'angle' (фаза), 'iq' (I/Q), 'constellation' (созвездие)
    
    figsize : tuple
        Размер фигуры (ширина, высота)
    
    save_path : str, optional
        Путь для сохранения графика
    
    show : bool
        Показывать ли график
    
    grid : bool
        Отображать ли сетку
    
    markers : list of dict, optional
        Список маркеров для отметки на графике.
        Каждый словарь должен содержать: {'x': позиция, 'label': метка, 'color': цвет}
    
    layout : tuple of (rows, cols), optional
        Расположение подграфиков. Если None, определяется автоматически
    
    **kwargs : дополнительные параметры для matplotlib
    
    Возвращает:
    -----------
    fig, axes : matplotlib figure и axes
    
    Примеры использования:
    ---------------------
    # Простой график одного сигнала
    plot_signal(signal, titles='Мой сигнал')
    
    # Несколько графиков
    plot_signal([signal1, signal2], titles=['Сигнал 1', 'Сигнал 2'])
    
    # График модуля комплексного сигнала
    plot_signal(complex_signal, plot_type='abs', titles='Модуль сигнала')
    
    # Созвездие
    plot_signal(symbols, plot_type='constellation', titles='QPSK созвездие')
    
    # С маркерами
    markers = [{'x': 100, 'label': 'Преамбула', 'color': 'red'}]
    plot_signal(signal, markers=markers)
    """
    
    # Преобразуем данные в список, если это одиночный массив
    if isinstance(data, np.ndarray):
        data = [data]
    
    # Определяем количество графиков
    n_plots = len(data)
    
    # Обработка для специальных типов графиков
    if plot_type == 'iq' and n_plots == 1:
        # Разделяем комплексный сигнал на I и Q
        data = [data[0].real, data[0].imag]
        n_plots = 2
        if titles is None or isinstance(titles, str):
            base_title = titles if isinstance(titles, str) else 'Сигнал'
            titles = [f'{base_title} - I (Real)', f'{base_title} - Q (Imag)']
        if isinstance(ylabel, str):
            ylabel = ['I', 'Q']
    
    elif plot_type in ['abs', 'angle'] and n_plots == 1:
        # Для комплексных сигналов вычисляем модуль или фазу
        if np.iscomplexobj(data[0]):
            if plot_type == 'abs':
                data = [np.abs(data[0])]
                if isinstance(ylabel, str):
                    ylabel = 'Модуль'
            else:  # angle
                data = [np.angle(data[0])]
                if isinstance(ylabel, str):
                    ylabel = 'Фаза (радианы)'
    
    elif plot_type == 'constellation':
        # Для созвездия создаем один график I vs Q
        if n_plots == 1 and np.iscomplexobj(data[0]):
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(data[0].real, data[0].imag, alpha=kwargs.get('alpha', 0.6), 
                      s=kwargs.get('s', 20), c=kwargs.get('c', 'blue'))
            ax.set_xlabel('I (In-phase)', fontsize=12)
            ax.set_ylabel('Q (Quadrature)', fontsize=12)
            ax.set_title(titles if titles else 'Созвездие', fontsize=14)
            ax.axis('equal')
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            if grid:
                ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"График сохранен в: {save_path}")
            if show:
                plt.show()
            
            return fig, ax
    
    # Определяем расположение подграфиков
    if layout is None:
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 1, 2
        elif n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        elif n_plots <= 9:
            rows, cols = 3, 3
        else:
            rows = int(np.ceil(np.sqrt(n_plots)))
            cols = int(np.ceil(n_plots / rows))
    else:
        rows, cols = layout
    
    # Создаем фигуру
    if n_plots == 1:
        fig, axes = plt.subplots(figsize=figsize)
        axes = [axes]  # Делаем список для единообразия
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Обработка заголовков и подписей
    if titles is None:
        titles = [f'График {i+1}' for i in range(n_plots)]
    elif isinstance(titles, str):
        titles = [titles]
    
    if isinstance(ylabel, str):
        ylabel = [ylabel] * n_plots
    
    # Строим графики
    for idx, (ax, signal_data) in enumerate(zip(axes[:n_plots], data)):
        # Создаем ось X
        x = np.arange(len(signal_data))
        
        # Выбираем тип графика
        if plot_type == 'line':
            ax.plot(x, signal_data, linewidth=kwargs.get('linewidth', 1), 
                   alpha=kwargs.get('alpha', 0.8), color=kwargs.get('color', None))
        elif plot_type == 'scatter':
            ax.scatter(x, signal_data, s=kwargs.get('s', 10), 
                      alpha=kwargs.get('alpha', 0.6), c=kwargs.get('c', 'blue'))
        elif plot_type == 'stem':
            ax.stem(x, signal_data, linefmt=kwargs.get('linefmt', 'b-'),
                   markerfmt=kwargs.get('markerfmt', 'bo'),
                   basefmt=kwargs.get('basefmt', 'r-'))
        elif plot_type in ['abs', 'angle']:
            ax.plot(x, signal_data, linewidth=kwargs.get('linewidth', 1), 
                   alpha=kwargs.get('alpha', 0.8))
        
        # Добавляем маркеры, если есть
        if markers and idx == 0:  # Маркеры только на первом графике
            for marker in markers:
                ax.axvline(marker.get('x', 0), 
                          color=marker.get('color', 'red'),
                          linestyle=marker.get('linestyle', '--'),
                          linewidth=marker.get('linewidth', 2),
                          label=marker.get('label', ''))
                if marker.get('label'):
                    ax.legend()
        
        # Настройка осей и заголовков
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel[idx] if idx < len(ylabel) else 'Амплитуда', fontsize=11)
        ax.set_title(titles[idx] if idx < len(titles) else f'График {idx+1}', 
                    fontsize=13, fontweight='bold')
        
        if grid:
            ax.grid(True, alpha=0.3)
    
    # Скрываем лишние подграфики
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Сохранение и отображение
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен в: {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_correlation_results(conv_results: List[np.ndarray],
                            conv_max: List[int],
                            offset: int,
                            sps: int,
                            save_path: Optional[str] = None,
                            show: bool = True,
                            window: int = 100):
    """
    Специализированная функция для построения результатов корреляции
    из функции find_preamble_offset.
    
    Параметры:
    ----------
    conv_results : list of np.ndarray
        Результаты корреляции для каждой фазы
    conv_max : list of int
        Позиции максимумов для каждой фазы
    offset : int
        Найденное смещение (медиана максимумов)
    sps : int
        Samples per symbol
    save_path : str, optional
        Путь для сохранения графика
    show : bool
        Показывать ли график
    window : int
        Размер окна вокруг максимума для увеличенного графика
    """
    n_phases = len(conv_results)
    
    # Первая фигура: отдельные графики для каждой фазы
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Дифференциальная корреляция для каждой фазы семплирования', 
                 fontsize=16, fontweight='bold')
    
    for idx in range(n_phases):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        corr_abs = np.abs(conv_results[idx])
        ax.plot(corr_abs, linewidth=1, color='blue', alpha=0.7)
        
        max_pos = conv_max[idx]
        max_val = corr_abs[max_pos]
        ax.axvline(max_pos, color='red', linestyle='--', linewidth=2,
                  label=f'Максимум: {max_pos} (знач: {max_val:.2f})')
        ax.scatter([max_pos], [max_val], color='red', s=100, zorder=5)
        
        ax.axvline(offset, color='green', linestyle=':', linewidth=2,
                  label=f'Медиана: {offset}', alpha=0.7)
        
        ax.set_title(f'Фаза {idx} (начало с отсчета {idx}, шаг {sps})',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Позиция (символы)', fontsize=11)
        ax.set_ylabel('|Корреляция|', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        x_min = max(0, offset - window)
        x_max = min(len(corr_abs), offset + window)
        ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    
    if save_path:
        base_path = save_path.rsplit('.', 1)
        path1 = f"{base_path[0]}_phases.{base_path[1]}" if len(base_path) > 1 else f"{save_path}_phases.png"
        plt.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"График фаз сохранен в: {path1}")
    
    if show:
        plt.show()
    
    # Вторая фигура: сравнение всех фаз
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Весь диапазон
    for idx in range(n_phases):
        corr_abs = np.abs(conv_results[idx])
        ax1.plot(corr_abs, linewidth=1, alpha=0.7, label=f'Фаза {idx}')
        ax1.scatter([conv_max[idx]], [corr_abs[conv_max[idx]]], s=100, zorder=5)
    
    ax1.axvline(offset, color='black', linestyle='--', linewidth=2.5,
               label=f'Offset={offset}')
    ax1.set_xlabel('Позиция (символы)', fontsize=12)
    ax1.set_ylabel('|Корреляция|', fontsize=12)
    ax1.set_title('Сравнение всех фаз (полный диапазон)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Увеличенная область
    x_min = max(0, offset - window)
    x_max = min(len(conv_results[0]), offset + window)
    
    for idx in range(n_phases):
        corr_abs = np.abs(conv_results[idx])
        ax2.plot(range(x_min, x_max), corr_abs[x_min:x_max], 
                linewidth=1.5, alpha=0.8, label=f'Фаза {idx}')
        if x_min <= conv_max[idx] < x_max:
            ax2.scatter([conv_max[idx]], [corr_abs[conv_max[idx]]], s=100, zorder=5)
    
    ax2.axvline(offset, color='black', linestyle='--', linewidth=2.5,
               label=f'Offset={offset}')
    ax2.set_xlabel('Позиция (символы)', fontsize=12)
    ax2.set_ylabel('|Корреляция|', fontsize=12)
    ax2.set_title(f'Увеличенная область (±{window} символов)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        base_path = save_path.rsplit('.', 1)
        path2 = f"{base_path[0]}_comparison.{base_path[1]}" if len(base_path) > 1 else f"{save_path}_comparison.png"
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"График сравнения сохранен в: {path2}")
    
    if show:
        plt.show()
    
    return fig1, fig2


# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    t = np.linspace(0, 2*np.pi, 1000)
    signal1 = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(len(t))
    signal2 = np.cos(2*np.pi*3*t)
    complex_signal = np.exp(1j*2*np.pi*t) * (1 + 0.1*np.random.randn(len(t)))
    
    # Примеры использования
    print("Пример 1: Простой график")
    plot_signal(signal1, titles='Синусоида с шумом', show=False)
    
    print("\nПример 2: Два графика")
    plot_signal([signal1, signal2], 
                titles=['Сигнал 1', 'Сигнал 2'], 
                show=False)
    
    print("\nПример 3: Модуль комплексного сигнала")
    plot_signal(complex_signal, plot_type='abs', 
                titles='Модуль комплексного сигнала', 
                show=False)
    
    print("\nПример 4: I/Q компоненты")
    plot_signal(complex_signal, plot_type='iq', 
                titles='Комплексный сигнал', 
                show=False)
    
    print("\nПример 5: Созвездие")
    symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) * (1 + 0.1*np.random.randn(100, 1))
    symbols = symbols.flatten()
    plot_signal(symbols, plot_type='constellation', 
                titles='QPSK созвездие',
                show=True)


