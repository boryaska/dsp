import numpy as np
import matplotlib.pyplot as plt
from synthesys import generate_mpsk_signal, apply_rrc_filter, apply_rc_filter

print("=" * 70)
print("ДЕМОНСТРАЦИЯ RRC И RC ФИЛЬТРОВ")
print("=" * 70)

# Генерация двух независимых сигналов (используем одинаковые параметры для fair comparison)
np.random.seed(42)  # Для воспроизводимости
signal_base = generate_mpsk_signal(
    modulation_type='qpsk',
    Fs=100000,
    Fc=1000,
    Sps=100,
    num_symbols=20
)

print(f"\nИсходный сигнал:")
print(f"  SPS: {signal_base['Sps']}")
print(f"  Символов: {len(signal_base['symbols'])}")
print(f"  Отсчетов: {len(signal_base['signal'])}")

# Функция для создания импульсного сигнала из символов
def create_impulse_train(symbols, Sps):
    """
    Создает импульсный сигнал (дельта-функции) из символов
    Вместо np.repeat(symbols, Sps), размещаем символы как дельта-функции
    """
    impulse_train = np.zeros(Sps * len(symbols), dtype=complex)
    for i, symbol in enumerate(symbols):
        impulse_train[i * Sps] = symbol
    return impulse_train

# Функция для применения фильтра к импульсному сигналу и последующей модуляции
def apply_filter_to_baseband(symbols, Sps, filter_coeffs, Fc, Fs):
    """
    Применяет фильтр к импульсному сигналу (дельта-функции), затем модулирует
    """
    # Создаем импульсный сигнал
    impulse_train = create_impulse_train(symbols, Sps)
    
    # Применяем фильтр
    filtered_full = np.convolve(impulse_train, filter_coeffs, mode='full')
    delay = len(filter_coeffs) // 2
    filtered_baseband = filtered_full[delay:delay + len(impulse_train)]
    
    # Модулируем на несущую
    t = np.arange(len(filtered_baseband)) / Fs
    carrier = np.exp(2j * np.pi * Fc * t)
    modulated = carrier * filtered_baseband
    
    return np.real(modulated), filtered_baseband

# Создаем фильтры вручную (такие же, как в функциях apply_rrc_filter и apply_rc_filter)
Sps = 100
filter_span = 10
alpha = 0.35

# RRC фильтр
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
rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter**2))

# RC фильтр
rc_filter = np.zeros(len(t))
for i, ti in enumerate(t):
    if abs(ti) < 1e-10:
        rc_filter[i] = 1.0
    elif alpha > 0 and abs(abs(ti) - 1/(2*alpha)) < 1e-10:
        rc_filter[i] = (np.pi/4) * np.sinc(1/(2*alpha))
    else:
        rc_filter[i] = np.sinc(ti) * np.cos(np.pi*alpha*ti) / (1 - (2*alpha*ti)**2)
sample_sum = 0
center_idx = len(rc_filter) // 2
for k in range(-filter_span//2, filter_span//2 + 1):
    idx = center_idx + k * Sps
    if 0 <= idx < len(rc_filter):
        sample_sum += rc_filter[idx]
if abs(sample_sum) > 1e-10:
    rc_filter = rc_filter / sample_sum

# Применяем фильтры к символам (создаем импульсный сигнал внутри функции)
rrc_signal, rrc_baseband = apply_filter_to_baseband(
    signal_base['symbols'], Sps, rrc_filter, signal_base['Fc'], signal_base['Fs']
)
rc_signal, rc_baseband = apply_filter_to_baseband(
    signal_base['symbols'], Sps, rc_filter, signal_base['Fc'], signal_base['Fs']
)

# Создаем словари для совместимости
rrc_filtered = {
    'signal': rrc_signal,
    'filter': rrc_filter,
    'baseband': rrc_baseband
}
rc_filtered = {
    'signal': rc_signal,
    'filter': rc_filter,
    'baseband': rc_baseband
}

signal_for_rrc = signal_base
signal_for_rc = signal_base

# Функция для восстановления символов из комплексного базового сигнала
def recover_symbols_from_baseband(baseband_signal, Sps, num_symbols):
    """
    Восстанавливает символы из комплексного базового сигнала
    Благодаря свойству Найквиста RC фильтра, достаточно взять отсчет в центре символа
    """
    symbols_recovered = []
    for i in range(num_symbols):
        # Берем отсчет в центре символа (в точке nT)
        sample_idx = i * Sps
        if sample_idx < len(baseband_signal):
            symbols_recovered.append(baseband_signal[sample_idx])
    
    return np.array(symbols_recovered)

# Восстанавливаем символы из отфильтрованных базовых сигналов
symbols_rrc_recovered = recover_symbols_from_baseband(
    rrc_filtered['baseband'], 100, 20
)

symbols_rc_recovered = recover_symbols_from_baseband(
    rc_filtered['baseband'], 100, 20
)

# Создаем визуализацию
fig = plt.figure(figsize=(20, 14))

# ============= РЯД 1: ИМПУЛЬСНЫЕ ХАРАКТЕРИСТИКИ ФИЛЬТРОВ =============
# График 1: Импульсная характеристика RRC
ax1 = plt.subplot(4, 3, 1)
t_filter_rrc = np.arange(len(rrc_filtered['filter'])) - len(rrc_filtered['filter'])//2
ax1.plot(t_filter_rrc, rrc_filtered['filter'], 'b-', linewidth=1.5)
ax1.set_title('RRC фильтр (α=0.35)', fontweight='bold')
ax1.set_ylabel('Амплитуда')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)

# График 2: Импульсная характеристика RC
ax2 = plt.subplot(4, 3, 2)
t_filter_rc = np.arange(len(rc_filtered['filter'])) - len(rc_filtered['filter'])//2
ax2.plot(t_filter_rc, rc_filtered['filter'], 'g-', linewidth=1.5)
ax2.set_title('RC фильтр (α=0.35)', fontweight='bold')
ax2.set_ylabel('Амплитуда')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)

# График 3: Сравнение фильтров
ax3 = plt.subplot(4, 3, 3)
ax3.plot(t_filter_rrc, rrc_filtered['filter'], 'b-', linewidth=1.5, label='RRC', alpha=0.7)
ax3.plot(t_filter_rc, rc_filtered['filter'], 'g-', linewidth=1.5, label='RC', alpha=0.7)
ax3.set_title('Сравнение RRC и RC', fontweight='bold')
ax3.set_ylabel('Амплитуда')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.axhline(y=0, color='k', linewidth=0.5)

# ============= РЯД 2: ВРЕМЕННЫЕ СИГНАЛЫ =============
samples = 4 * signal_for_rrc['Sps']

# График 4: Исходный сигнал для RRC
ax4 = plt.subplot(4, 3, 4)
t_signal_rrc = signal_for_rrc['t'][:samples]
ax4.plot(t_signal_rrc, signal_for_rrc['signal'][:samples], 'k-', linewidth=1.5)
ax4.set_title('Исходный сигнал (для RRC)', fontweight='bold')
ax4.set_ylabel('Амплитуда')
ax4.grid(True, alpha=0.3)

# График 5: После RRC фильтра
ax5 = plt.subplot(4, 3, 5)
t_rrc = np.arange(len(rrc_filtered['signal'])) / signal_for_rrc['Fs']
ax5.plot(t_rrc[:samples], rrc_filtered['signal'][:samples], 'b-', linewidth=1.5)
ax5.set_title('После RRC фильтра', fontweight='bold')
ax5.set_ylabel('Амплитуда')
ax5.grid(True, alpha=0.3)

# График 6: После RC фильтра
ax6 = plt.subplot(4, 3, 6)
t_rc = np.arange(len(rc_filtered['signal'])) / signal_for_rc['Fs']
ax6.plot(t_rc[:samples], rc_filtered['signal'][:samples], 'g-', linewidth=1.5)
ax6.set_title('После RC фильтра', fontweight='bold')
ax6.set_ylabel('Амплитуда')
ax6.grid(True, alpha=0.3)

# ============= РЯД 3: СПЕКТРЫ =============
def plot_spectrum_filter(ax, signal_data, Fs, title, color):
    fft_data = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(len(signal_data), 1/Fs)
    positive_mask = freqs >= 0
    freqs_pos = freqs[positive_mask]
    magnitude = np.abs(fft_data[positive_mask])
    
    ax.plot(freqs_pos, magnitude, color=color, linewidth=1.5)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_ylabel('Магнитуда')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5000)

# График 7: Спектр исходного RRC
ax7 = plt.subplot(4, 3, 7)
plot_spectrum_filter(ax7, signal_for_rrc['signal'], signal_for_rrc['Fs'], 
                    'Спектр до RRC', 'black')

# График 8: Спектр после RRC
ax8 = plt.subplot(4, 3, 8)
plot_spectrum_filter(ax8, rrc_filtered['signal'], signal_for_rrc['Fs'], 
                    'Спектр после RRC', 'blue')

# График 9: Спектр после RC
ax9 = plt.subplot(4, 3, 9)
plot_spectrum_filter(ax9, rc_filtered['signal'], signal_for_rc['Fs'], 
                    'Спектр после RC', 'green')

# ============= РЯД 4: СОЗВЕЗДИЯ =============
# График 10: Созвездие до RRC
ax10 = plt.subplot(4, 3, 10)
symbols_orig_rrc = signal_for_rrc['symbols']
ax10.scatter(np.real(symbols_orig_rrc), np.imag(symbols_orig_rrc), 
           s=100, alpha=0.7, edgecolors='black', linewidth=1.5, c='lightgray')
ax10.set_title('Созвездие до RRC', fontweight='bold', fontsize=10)
ax10.set_xlabel('Re (I)')
ax10.set_ylabel('Im (Q)')
ax10.grid(True, alpha=0.3)
ax10.axis('equal')
ax10.set_xlim(-1.5, 1.5)
ax10.set_ylim(-1.5, 1.5)
ax10.axhline(y=0, color='k', linewidth=0.5)
ax10.axvline(x=0, color='k', linewidth=0.5)

# График 11: Созвездие после RRC
ax11 = plt.subplot(4, 3, 11)
ax11.scatter(np.real(symbols_rrc_recovered), np.imag(symbols_rrc_recovered), 
           s=100, alpha=0.7, edgecolors='blue', linewidth=2, c='lightblue', label='После RRC')
ax11.scatter(np.real(symbols_orig_rrc), np.imag(symbols_orig_rrc), 
           s=50, alpha=0.4, c='gray', marker='x', linewidth=1.5, label='До фильтра')
ax11.set_title('Созвездие после RRC\n(искажено, нужен второй RRC)', fontweight='bold', fontsize=10)
ax11.set_xlabel('Re (I)')
ax11.set_ylabel('Im (Q)')
ax11.grid(True, alpha=0.3)
ax11.axis('equal')
ax11.set_xlim(-1.5, 1.5)
ax11.set_ylim(-1.5, 1.5)
ax11.axhline(y=0, color='k', linewidth=0.5)
ax11.axvline(x=0, color='k', linewidth=0.5)
ax11.legend(fontsize=7)

# График 12: Созвездие после RC
ax12 = plt.subplot(4, 3, 12)
symbols_orig_rc = signal_for_rc['symbols']
ax12.scatter(np.real(symbols_rc_recovered), np.imag(symbols_rc_recovered), 
           s=100, alpha=0.7, edgecolors='green', linewidth=2, c='lightgreen', label='После RC')
ax12.scatter(np.real(symbols_orig_rc), np.imag(symbols_orig_rc), 
           s=50, alpha=0.4, c='gray', marker='x', linewidth=1.5, label='До фильтра')
ax12.set_title('Созвездие после RC\n✓ Символы совпадают!', fontweight='bold', fontsize=10)
ax12.set_xlabel('Re (I)')
ax12.set_ylabel('Im (Q)')
ax12.grid(True, alpha=0.3)
ax12.axis('equal')
ax12.set_xlim(-1.5, 1.5)
ax12.set_ylim(-1.5, 1.5)
ax12.axhline(y=0, color='k', linewidth=0.5)
ax12.axvline(x=0, color='k', linewidth=0.5)
ax12.legend(fontsize=7)

plt.suptitle('Демонстрация RRC и RC фильтров для формирования спектра', 
            fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("📊 ЧТО ПОКАЗЫВАЮТ ГРАФИКИ:")
print("=" * 70)
print()
print("1️⃣  ИМПУЛЬСНЫЕ ХАРАКТЕРИСТИКИ (ряд 1):")
print("   • RRC: имеет более длинные хвосты")
print("   • RC: более компактный импульс")
print("   • RC = RRC × RRC (в частотной области)")
print()
print("2️⃣  ВРЕМЕННЫЕ СИГНАЛЫ (ряд 2):")
print("   • Исходный: резкие переходы между символами")
print("   • После RRC: частично сглажен")
print("   • После RC: полностью сглажен, идеальная форма")
print()
print("3️⃣  СПЕКТРЫ (ряд 3):")
print("   • Исходный: широкий спектр с боковыми лепестками")
print("   • После RRC: спектр ограничен")
print("   • После RC: наиболее компактный спектр")
print()
print("4️⃣  СОЗВЕЗДИЯ (ряд 4) - КЛЮЧЕВОЙ МОМЕНТ:")
print("   • График 10: исходные символы (до фильтров)")
print("   • График 11: после RRC → символы ИСКАЖЕНЫ ❌")
print("     - RRC один не обеспечивает нулевую ISI")
print("     - Нужен второй RRC фильтр в приемнике")
print("   • График 12: после RC → символы СОВПАДАЮТ ✓")
print("     - RC обеспечивает нулевую ISI!")
print("     - Свойство Найквиста выполнено")
print()
print("=" * 70)
print("🎯 ВАЖНОЕ ОТЛИЧИЕ RRC И RC:")
print("=" * 70)
print()
print("RRC (Root Raised Cosine):")
print("   ❌ ОДИН RRC фильтр НЕ сохраняет символы")
print("   ✓  Используется в ПЕРЕДАТЧИКЕ и ПРИЕМНИКЕ")
print("   ✓  RRC(TX) × RRC(RX) = RC → нулевая ISI")
print("   ✓  Распределяет фильтрацию между TX и RX")
print()
print("RC (Raised Cosine):")
print("   ✓  ОДИН RC фильтр сохраняет символы")
print("   ✓  Идеальный фильтр Найквиста")
print("   ✓  Нулевая ISI в точках семплирования")
print("   ❌  Вся фильтрация в одном месте (неоптимально)")
print()
print("💡 ПРАКТИЧЕСКИЙ ВЫВОД:")
print("   В реальных системах:")
print("   • Передатчик: применяет RRC")
print("   • Канал связи: добавляет искажения и шум")
print("   • Приемник: применяет RRC")
print("   • Результат: RRC × RRC = RC → идеальное восстановление!")
print()
print("   Почему так?")
print("   • Оптимальное соотношение сигнал/шум")
print("   • Распределение нагрузки между TX и RX")
print("   • Соответствие стандартам связи")
print()
print("=" * 70)
print("✓ Демонстрация завершена!")
print("=" * 70)

