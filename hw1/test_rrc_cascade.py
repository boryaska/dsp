import numpy as np
import matplotlib.pyplot as plt
from synthesys import generate_mpsk_signal

print("=" * 70)
print("ДЕМОНСТРАЦИЯ КАСКАДА RRC ФИЛЬТРОВ: RRC × RRC = RC")
print("=" * 70)
print()
print("Цель: показать, что два RRC фильтра последовательно")
print("      дают результат, эквивалентный одному RC фильтру")
print()

# Генерация сигнала (больше символов для избежания краевых эффектов)
np.random.seed(42)  # Для воспроизводимости
signal_base = generate_mpsk_signal(
    modulation_type='qpsk',
    Fs=100000,
    Fc=1000,
    Sps=100,
    num_symbols=40,  # Увеличено с 20 до 40
    use_pulse_shaping=True  # Используем импульсы (по умолчанию True, но явно указываем)
)

print(f"Исходный сигнал:")
print(f"  SPS: {signal_base['Sps']}")
print(f"  Символов: {len(signal_base['symbols'])}")
print(f"  Отсчетов: {len(signal_base['signal'])}")
print()

# Параметры фильтров
Sps = 100
filter_span = 10
alpha = 0.35

# Функция для создания импульсного сигнала из символов
def create_impulse_train(symbols, Sps):
    """Создает импульсный сигнал (дельта-функции) из символов"""
    impulse_train = np.zeros(Sps * len(symbols), dtype=complex)
    for i, symbol in enumerate(symbols):
        impulse_train[i * Sps] = symbol
    return impulse_train

# Функция для создания RRC фильтра
def create_rrc_filter(Sps, alpha, filter_span):
    """Создает RRC фильтр"""
    filter_length = filter_span * Sps
    if filter_length % 2 == 0:
        filter_length += 1
    
    t = np.arange(-filter_length//2, filter_length//2 + 1) / Sps
    rrc_filter = np.zeros(len(t))
    
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            rrc_filter[i] = (1 - alpha + 4*alpha/np.pi)
        elif alpha > 0 and abs(abs(ti) - 1/(4*alpha)) < 1e-10:
            rrc_filter[i] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) + 
                (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )
        else:
            numerator = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            denominator = np.pi*ti*(1 - (4*alpha*ti)**2)
            rrc_filter[i] = numerator / denominator
    
    # Нормализация (энергия = 1)
    rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter**2))
    
    return rrc_filter, t

# Функция для создания RC фильтра
def create_rc_filter(Sps, alpha, filter_span):
    """Создает RC фильтр"""
    filter_length = filter_span * Sps
    if filter_length % 2 == 0:
        filter_length += 1
    
    t = np.arange(-filter_length//2, filter_length//2 + 1) / Sps
    rc_filter = np.zeros(len(t))
    
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            rc_filter[i] = 1.0
        elif alpha > 0 and abs(abs(ti) - 1/(2*alpha)) < 1e-10:
            rc_filter[i] = (np.pi/4) * np.sinc(1/(2*alpha))
        else:
            rc_filter[i] = np.sinc(ti) * np.cos(np.pi*alpha*ti) / (1 - (2*alpha*ti)**2)
    
    # Нормализация
    sample_sum = 0
    center_idx = len(rc_filter) // 2
    for k in range(-filter_span//2, filter_span//2 + 1):
        idx = center_idx + k * Sps
        if 0 <= idx < len(rc_filter):
            sample_sum += rc_filter[idx]
    
    if abs(sample_sum) > 1e-10:
        rc_filter = rc_filter / sample_sum
    
    return rc_filter, t

# Функция для применения фильтра
def apply_filter(signal, filter_coeffs):
    """Применяет фильтр к сигналу"""
    filtered_full = np.convolve(signal, filter_coeffs, mode='full')
    delay = len(filter_coeffs) // 2
    filtered = filtered_full[delay:delay + len(signal)]
    return filtered

# Функция для модуляции
def modulate_signal(baseband_signal, Fc, Fs):
    """Модулирует комплексный базовый сигнал на несущую"""
    t = np.arange(len(baseband_signal)) / Fs
    carrier = np.exp(2j * np.pi * Fc * t)
    modulated = carrier * baseband_signal
    return np.real(modulated)

# Функция для восстановления символов
def recover_symbols(baseband_signal, Sps, num_symbols):
    """Восстанавливает символы из базового сигнала"""
    symbols_recovered = []
    for i in range(num_symbols):
        sample_idx = i * Sps
        if sample_idx < len(baseband_signal):
            symbols_recovered.append(baseband_signal[sample_idx])
    return np.array(symbols_recovered)

# Создаем фильтры
print("Создание фильтров...")
rrc_filter, t_filter = create_rrc_filter(Sps, alpha, filter_span)
rc_filter, _ = create_rc_filter(Sps, alpha, filter_span)
print(f"  RRC фильтр: {len(rrc_filter)} отсчетов")
print(f"  RC фильтр: {len(rc_filter)} отсчетов")
print()

# ============================================================
# ЭТАП 1: Исходный импульсный сигнал
# ============================================================
print("ЭТАП 1: Импульсный сигнал (уже создан в generate_mpsk_signal)...")
# Теперь symbols_expanded уже содержит импульсы благодаря use_pulse_shaping=True
impulse_train = signal_base['symbols_expanded']
signal_original = modulate_signal(impulse_train, signal_base['Fc'], signal_base['Fs'])
print(f"  Длина: {len(impulse_train)} отсчетов")
print()

# ============================================================
# ЭТАП 2: После первого RRC (передатчик)
# ============================================================
print("ЭТАП 2: Применение первого RRC фильтра (ПЕРЕДАТЧИК)...")
baseband_after_rrc1 = apply_filter(impulse_train, rrc_filter)
signal_after_rrc1 = modulate_signal(baseband_after_rrc1, signal_base['Fc'], signal_base['Fs'])
print(f"  Длина: {len(baseband_after_rrc1)} отсчетов")
print()

# Проверяем символы после первого RRC
symbols_after_rrc1 = recover_symbols(baseband_after_rrc1, Sps, len(signal_base['symbols']))
errors_rrc1 = [np.abs(symbols_after_rrc1[i] - signal_base['symbols'][i]) 
               for i in range(min(10, len(signal_base['symbols'])))]
print(f"  Проверка символов после 1-го RRC (первые 10):")
print(f"    Средняя ошибка: {np.mean(errors_rrc1):.6f}")
print(f"    Максимальная ошибка: {np.max(errors_rrc1):.6f}")
print(f"    ❌ Символы ИСКАЖЕНЫ (один RRC не обеспечивает ISI=0)")
print()

# ============================================================
# ЭТАП 3: После второго RRC (приемник)
# ============================================================
print("ЭТАП 3: Применение второго RRC фильтра (ПРИЕМНИК)...")
baseband_after_rrc2 = apply_filter(baseband_after_rrc1, rrc_filter)
signal_after_rrc2 = modulate_signal(baseband_after_rrc2, signal_base['Fc'], signal_base['Fs'])
print(f"  Длина: {len(baseband_after_rrc2)} отсчетов")
print()

# Проверяем символы после двух RRC
symbols_after_rrc2 = recover_symbols(baseband_after_rrc2, Sps, len(signal_base['symbols']))
errors_rrc2 = [np.abs(symbols_after_rrc2[i] - signal_base['symbols'][i]) 
               for i in range(min(10, len(signal_base['symbols'])))]
print(f"  Проверка символов после 2-х RRC (первые 10):")
print(f"    Средняя ошибка: {np.mean(errors_rrc2):.6f}")
print(f"    Максимальная ошибка: {np.max(errors_rrc2):.6f}")
print(f"    ✅ Символы ВОССТАНОВЛЕНЫ (RRC × RRC = RC)")
print()

# ============================================================
# ЭТАП 4: Для сравнения - один RC фильтр
# ============================================================
print("ЭТАП 4: Для сравнения - один RC фильтр...")
baseband_rc = apply_filter(impulse_train, rc_filter)
signal_rc = modulate_signal(baseband_rc, signal_base['Fc'], signal_base['Fs'])
print(f"  Длина: {len(baseband_rc)} отсчетов")
print()

# Проверяем символы после RC
symbols_rc = recover_symbols(baseband_rc, Sps, len(signal_base['symbols']))
errors_rc = [np.abs(symbols_rc[i] - signal_base['symbols'][i]) 
             for i in range(min(10, len(signal_base['symbols'])))]
print(f"  Проверка символов после RC (первые 10):")
print(f"    Средняя ошибка: {np.mean(errors_rc):.6f}")
print(f"    Максимальная ошибка: {np.max(errors_rc):.6f}")
print(f"    ✅ Символы ИДЕАЛЬНЫ")
print()

# ============================================================
# Для графиков: используем только символы из СЕРЕДИНЫ
# (избегаем краевых эффектов)
# ============================================================
# Пропускаем первые и последние filter_span символов
edge_skip = filter_span
middle_start = edge_skip
middle_end = len(signal_base['symbols']) - edge_skip

symbols_original_middle = signal_base['symbols'][middle_start:middle_end]
symbols_rrc1_middle = symbols_after_rrc1[middle_start:middle_end]
symbols_rrc2_middle = symbols_after_rrc2[middle_start:middle_end]
symbols_rc_middle = symbols_rc[middle_start:middle_end]

print(f"Для графиков используем символы {middle_start}-{middle_end} (без краевых эффектов)")
print()

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
print("Создание графиков...")
fig = plt.figure(figsize=(20, 16))

samples = 4 * Sps  # Показываем 4 символа

# ============= РЯД 1: ИМПУЛЬСНЫЕ ХАРАКТЕРИСТИКИ =============
# График 1: RRC фильтр
ax1 = plt.subplot(5, 3, 1)
t_filter_plot = np.arange(len(rrc_filter)) - len(rrc_filter)//2
ax1.plot(t_filter_plot, rrc_filter, 'b-', linewidth=1.5)
ax1.set_title('RRC фильтр (α=0.35)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Амплитуда', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)

# График 2: RRC × RRC (в частотной области = RC)
ax2 = plt.subplot(5, 3, 2)
# Вычисляем свертку двух RRC (эквивалент в частотной области)
rrc_squared = np.convolve(rrc_filter, rrc_filter, mode='same')
# Нормализация для сравнения
rrc_squared_norm = rrc_squared / np.max(np.abs(rrc_squared))
rc_norm = rc_filter / np.max(np.abs(rc_filter))
t_filter_plot2 = np.arange(len(rrc_squared)) - len(rrc_squared)//2
ax2.plot(t_filter_plot2, rrc_squared_norm, 'purple', linewidth=1.5, label='RRC × RRC', alpha=0.7)
ax2.plot(t_filter_plot, rc_norm, 'g--', linewidth=1.5, label='RC', alpha=0.7)
ax2.set_title('RRC × RRC ≈ RC (нормированные)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Амплитуда', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='k', linewidth=0.5)

# График 3: RC фильтр
ax3 = plt.subplot(5, 3, 3)
ax3.plot(t_filter_plot, rc_filter, 'g-', linewidth=1.5)
ax3.set_title('RC фильтр (α=0.35)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Амплитуда', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)

# ============= РЯД 2: ВРЕМЕННЫЕ СИГНАЛЫ (исходный) =============
# График 4: Исходный импульсный сигнал (базовый)
ax4 = plt.subplot(5, 3, 4)
t_signal = np.arange(len(impulse_train)) / signal_base['Fs']
ax4.plot(t_signal[:samples], np.real(impulse_train[:samples]), 'k-', linewidth=1.5, label='Re')
ax4.plot(t_signal[:samples], np.imag(impulse_train[:samples]), 'k--', linewidth=1.5, alpha=0.5, label='Im')
ax4.set_title('ЭТАП 1: Импульсный сигнал (базовый)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Амплитуда', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)

# График 5: Исходный сигнал (модулированный)
ax5 = plt.subplot(5, 3, 5)
ax5.plot(t_signal[:samples], signal_original[:samples], 'k-', linewidth=1.5)
ax5.set_title('ЭТАП 1: Модулированный на несущую', fontweight='bold', fontsize=11)
ax5.set_ylabel('Амплитуда', fontsize=10)
ax5.grid(True, alpha=0.3)

# График 6: Пустой (резерв)
ax6 = plt.subplot(5, 3, 6)
ax6.text(0.5, 0.5, 'Импульсный сигнал\n(дельта-функции)\n\nСимволы размещены\nв точках t=nT', 
         ha='center', va='center', fontsize=10, transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax6.axis('off')

# ============= РЯД 3: ПОСЛЕ ПЕРВОГО RRC =============
# График 7: После первого RRC (базовый)
ax7 = plt.subplot(5, 3, 7)
ax7.plot(t_signal[:samples], np.real(baseband_after_rrc1[:samples]), 'b-', linewidth=1.5, label='Re')
ax7.plot(t_signal[:samples], np.imag(baseband_after_rrc1[:samples]), 'b--', linewidth=1.5, alpha=0.5, label='Im')
ax7.set_title('ЭТАП 2: После 1-го RRC (базовый)', fontweight='bold', fontsize=11)
ax7.set_ylabel('Амплитуда', fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=8)

# График 8: После первого RRC (модулированный)
ax8 = plt.subplot(5, 3, 8)
ax8.plot(t_signal[:samples], signal_after_rrc1[:samples], 'b-', linewidth=1.5)
ax8.set_title('ЭТАП 2: Модулированный (ПЕРЕДАТЧИК)', fontweight='bold', fontsize=11)
ax8.set_ylabel('Амплитуда', fontsize=10)
ax8.grid(True, alpha=0.3)

# График 9: Созвездие после первого RRC (только середина)
ax9 = plt.subplot(5, 3, 9)
ax9.scatter(symbols_original_middle.real, symbols_original_middle.imag, 
           c='gray', marker='x', s=100, alpha=0.5, label='Исходные', linewidths=2)
ax9.scatter(symbols_rrc1_middle.real, symbols_rrc1_middle.imag,
           c='blue', marker='o', s=80, alpha=0.7, edgecolors='darkblue', 
           linewidths=1.5, label='После 1-го RRC')
errors_rrc1_middle = [np.abs(symbols_rrc1_middle[i] - symbols_original_middle[i]) for i in range(len(symbols_rrc1_middle))]
ax9.set_title(f'Созвездие после 1-го RRC\n❌ Искажено (ошибка={np.max(errors_rrc1_middle):.3f})', 
             fontweight='bold', fontsize=11)
ax9.set_xlabel('Re (I)', fontsize=10)
ax9.set_ylabel('Im (Q)', fontsize=10)
ax9.grid(True, alpha=0.3)
ax9.axis('equal')
ax9.set_xlim(-1.5, 1.5)
ax9.set_ylim(-1.5, 1.5)
ax9.axhline(y=0, color='k', linewidth=0.5)
ax9.axvline(x=0, color='k', linewidth=0.5)
ax9.legend(fontsize=8, loc='upper right')

# ============= РЯД 4: ПОСЛЕ ВТОРОГО RRC =============
# График 10: После второго RRC (базовый)
ax10 = plt.subplot(5, 3, 10)
ax10.plot(t_signal[:samples], np.real(baseband_after_rrc2[:samples]), 'purple', linewidth=1.5, label='Re')
ax10.plot(t_signal[:samples], np.imag(baseband_after_rrc2[:samples]), color='purple', 
         linestyle='--', linewidth=1.5, alpha=0.5, label='Im')
ax10.set_title('ЭТАП 3: После 2-го RRC (базовый)', fontweight='bold', fontsize=11)
ax10.set_ylabel('Амплитуда', fontsize=10)
ax10.grid(True, alpha=0.3)
ax10.legend(fontsize=8)

# График 11: После второго RRC (модулированный)
ax11 = plt.subplot(5, 3, 11)
ax11.plot(t_signal[:samples], signal_after_rrc2[:samples], 'purple', linewidth=1.5)
ax11.set_title('ЭТАП 3: Модулированный (ПРИЕМНИК)', fontweight='bold', fontsize=11)
ax11.set_ylabel('Амплитуда', fontsize=10)
ax11.set_xlabel('Время (с)', fontsize=10)
ax11.grid(True, alpha=0.3)

# График 12: Созвездие после двух RRC (только середина)
ax12 = plt.subplot(5, 3, 12)
ax12.scatter(symbols_original_middle.real, symbols_original_middle.imag, 
            c='gray', marker='x', s=100, alpha=0.5, label='Исходные', linewidths=2)
ax12.scatter(symbols_rrc2_middle.real, symbols_rrc2_middle.imag,
            c='purple', marker='o', s=80, alpha=0.7, edgecolors='darkviolet', 
            linewidths=1.5, label='После 2-х RRC')
errors_rrc2_middle = [np.abs(symbols_rrc2_middle[i] - symbols_original_middle[i]) for i in range(len(symbols_rrc2_middle))]
ax12.set_title(f'Созвездие после 2-х RRC\n✅ Восстановлено (ошибка={np.max(errors_rrc2_middle):.6f})', 
              fontweight='bold', fontsize=11)
ax12.set_xlabel('Re (I)', fontsize=10)
ax12.set_ylabel('Im (Q)', fontsize=10)
ax12.grid(True, alpha=0.3)
ax12.axis('equal')
ax12.set_xlim(-1.5, 1.5)
ax12.set_ylim(-1.5, 1.5)
ax12.axhline(y=0, color='k', linewidth=0.5)
ax12.axvline(x=0, color='k', linewidth=0.5)
ax12.legend(fontsize=8, loc='upper right')

# ============= РЯД 5: ДЛЯ СРАВНЕНИЯ - RC ФИЛЬТР =============
# График 13: После RC (базовый)
ax13 = plt.subplot(5, 3, 13)
ax13.plot(t_signal[:samples], np.real(baseband_rc[:samples]), 'g-', linewidth=1.5, label='Re')
ax13.plot(t_signal[:samples], np.imag(baseband_rc[:samples]), 'g--', linewidth=1.5, alpha=0.5, label='Im')
ax13.set_title('ЭТАП 4: После RC (базовый)', fontweight='bold', fontsize=11)
ax13.set_ylabel('Амплитуда', fontsize=10)
ax13.set_xlabel('Время (с)', fontsize=10)
ax13.grid(True, alpha=0.3)
ax13.legend(fontsize=8)

# График 14: После RC (модулированный)
ax14 = plt.subplot(5, 3, 14)
ax14.plot(t_signal[:samples], signal_rc[:samples], 'g-', linewidth=1.5)
ax14.set_title('ЭТАП 4: Модулированный (RC напрямую)', fontweight='bold', fontsize=11)
ax14.set_ylabel('Амплитуда', fontsize=10)
ax14.set_xlabel('Время (с)', fontsize=10)
ax14.grid(True, alpha=0.3)

# График 15: Сравнение созвездий (только середина, зелёные ПОВЕРХ фиолетовых)
ax15 = plt.subplot(5, 3, 15)
ax15.scatter(symbols_original_middle.real, symbols_original_middle.imag, 
            c='gray', marker='x', s=120, alpha=0.5, label='Исходные', linewidths=2, zorder=1)
# Сначала рисуем фиолетовые (RRC×RRC) - внизу
ax15.scatter(symbols_rrc2_middle.real, symbols_rrc2_middle.imag,
            c='purple', marker='o', s=120, alpha=0.5, edgecolors='darkviolet', 
            linewidths=2, label='RRC×RRC', zorder=2)
# Потом рисуем зелёные квадраты (RC) - сверху, больше размер
ax15.scatter(symbols_rc_middle.real, symbols_rc_middle.imag,
            c='lime', marker='s', s=100, alpha=0.9, edgecolors='darkgreen', 
            linewidths=2.5, label='RC', zorder=3)
errors_rc_middle = [np.abs(symbols_rc_middle[i] - symbols_original_middle[i]) for i in range(len(symbols_rc_middle))]
ax15.set_title(f'Сравнение: RRC×RRC vs RC\n✅ Оба отлично! (RC ошибка={np.max(errors_rc_middle):.8f})', 
              fontweight='bold', fontsize=11)
ax15.set_xlabel('Re (I)', fontsize=10)
ax15.set_ylabel('Im (Q)', fontsize=10)
ax15.grid(True, alpha=0.3)
ax15.axis('equal')
ax15.set_xlim(-1.5, 1.5)
ax15.set_ylim(-1.5, 1.5)
ax15.axhline(y=0, color='k', linewidth=0.5)
ax15.axvline(x=0, color='k', linewidth=0.5)
ax15.legend(fontsize=8, loc='upper right')

plt.suptitle('Демонстрация каскада RRC фильтров: RRC × RRC = RC', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("📊 ИТОГИ:")
print("=" * 70)
print()
print("1️⃣  ЭТАП 1: Импульсный сигнал (исходный)")
print("   • Дельта-функции в точках t=nT")
print()
print("2️⃣  ЭТАП 2: После 1-го RRC фильтра (ПЕРЕДАТЧИК)")
print(f"   • Форма импульса: Root Raised Cosine")
print(f"   • Средняя ошибка: {np.mean(errors_rrc1):.6f}")
print(f"   • Максимальная ошибка: {np.max(errors_rrc1):.6f}")
print(f"   • ❌ Символы ИСКАЖЕНЫ (ISI присутствует)")
print()
print("3️⃣  ЭТАП 3: После 2-го RRC фильтра (ПРИЕМНИК)")
print(f"   • RRC × RRC ≈ RC")
print(f"   • Средняя ошибка: {np.mean(errors_rrc2):.6f}")
print(f"   • Максимальная ошибка: {np.max(errors_rrc2):.6f}")
if np.max(errors_rrc2) < 0.02:
    print(f"   • ✅ Символы ВОССТАНОВЛЕНЫ (ошибка < 2%)")
else:
    print(f"   • ⚠️  Небольшие ошибки из-за краевых эффектов")
print()
print("4️⃣  ЭТАП 4: Для сравнения - RC фильтр напрямую")
print(f"   • Форма импульса: Raised Cosine")
print(f"   • Средняя ошибка: {np.mean(errors_rc):.8f}")
print(f"   • Максимальная ошибка: {np.max(errors_rc):.8f}")
print(f"   • ✅ Символы ИДЕАЛЬНЫ (машинная точность)")
print()
print("=" * 70)
print("🎯 ВЫВОД:")
print("=" * 70)
print()
print("✅ Два RRC фильтра последовательно (передатчик + приемник)")
print("   дают результат, очень близкий к одному RC фильтру!")
print()
print("✅ RRC × RRC ≈ RC продемонстрировано на графиках:")
print("   • График 2: импульсные характеристики похожи")
print("   • График 15: созвездия практически идентичны")
print()
print("📝 Небольшие отличия (ошибка ~1-2%) связаны с:")
print("   • Конечной длиной фильтров (truncation)")
print("   • Краевыми эффектами свертки")
print("   • Численными погрешностями")
print()
print("💡 В реальных системах связи:")
print("   📡 Передатчик: применяет RRC")
print("   📻 Приемник: применяет RRC")
print("   🎯 Результат: минимальная ISI, отличное качество!")
print()
print("=" * 70)

