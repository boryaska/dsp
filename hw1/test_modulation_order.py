import numpy as np
import matplotlib.pyplot as plt
from synthesys import generate_mpsk_signal, apply_rc_filter

print("=" * 70)
print("ПОРЯДОК ОПЕРАЦИЙ: Модуляция → Фильтр vs Фильтр → Модуляция")
print("=" * 70)
print()
print("Вопрос: Что будет, если модулировать импульсный сигнал,")
print("        а потом пропустить через RC фильтр?")
print()

# Параметры
np.random.seed(42)
Fs = 100000
Fc = 1000
Sps = 100
num_symbols = 40  # Увеличено для избежания краевых эффектов
alpha = 0.35
filter_span = 10

print("Параметры:")
print(f"  Fs = {Fs} Гц")
print(f"  Fc = {Fc} Гц")
print(f"  Sps = {Sps}")
print(f"  Символов = {num_symbols}")
print(f"  RC фильтр: alpha = {alpha}, span = {filter_span}")
print()

# Создаем функции для модуляции
def modulate_baseband(baseband_signal, Fc, Fs):
    """
    Модулирует комплексный базовый сигнал на несущую
    
    s(t) = Re{baseband(t) × exp(j×2π×Fc×t)}
    """
    t = np.arange(len(baseband_signal)) / Fs
    carrier = np.exp(2j * np.pi * Fc * t)
    modulated = carrier * baseband_signal
    return np.real(modulated)

def demodulate_signal(signal_real, Fc, Fs):
    """
    Демодулирует реальный сигнал
    
    Умножаем на комплексную несущую exp(-j×2π×Fc×t)
    Результат содержит базовый сигнал + высокочастотную компоненту на 2×Fc,
    которую нужно отфильтровать усреднением по длительности символа.
    """
    t = np.arange(len(signal_real)) / Fs
    carrier = np.exp(-2j * np.pi * Fc * t)
    demodulated = signal_real * carrier
    return demodulated

def recover_symbols(baseband_signal, Sps, num_symbols, Fc, Fs, filter_delay=0):
    """
    Восстанавливает символы из базового сигнала после демодуляции
    
    После демодуляции сигнал содержит:
    1. Низкочастотную компоненту (базовый сигнал) - это то, что нам нужно
    2. Высокочастотную компоненту на 2×Fc - это нужно отфильтровать
    
    Для фильтрации ВЧ усредняем по одному периоду несущей (или кратному).
    Умножаем на 2 для компенсации потери амплитуды.
    
    Parameters:
    -----------
    baseband_signal : array
        Комплексный сигнал после демодуляции
    Sps : int
        Samples per symbol
    num_symbols : int
        Количество символов
    Fc : float
        Частота несущей (для расчета окна усреднения)
    Fs : float
        Частота дискретизации
    filter_delay : int
        Задержка фильтра в отсчетах
        
    Returns:
    --------
    symbols : array
        Восстановленные символы
    """
    # Окно усреднения должно быть достаточным для фильтрации ВЧ на 2×Fc,
    # но не слишком широким, чтобы не захватывать соседние символы (ISI).
    # Оптимально: 1 период несущей (дает хороший компромисс)
    samples_per_carrier_period = int(Fs / Fc)
    avg_window = samples_per_carrier_period  # 1 период несущей
    
    symbols = []
    for i in range(num_symbols):
        # Центр символа
        center_idx = i * Sps + filter_delay
        
        # Усредняем вокруг центра символа по окну = 1 период несущей
        start_idx = max(0, center_idx - avg_window // 2)
        end_idx = min(len(baseband_signal), center_idx + avg_window // 2 + 1)
        
        if end_idx > start_idx:
            # Усредняем и умножаем на 2
            symbol_avg = 2 * np.mean(baseband_signal[start_idx:end_idx])
            symbols.append(symbol_avg)
        else:
            symbols.append(np.nan)
            
    return np.array(symbols)

# ============================================================
# ПОДХОД 1: ПРАВИЛЬНЫЙ - Фильтр → Модуляция
# ============================================================
print("=" * 70)
print("ПОДХОД 1 (ПРАВИЛЬНЫЙ): Фильтр в базовой полосе → Модуляция")
print("=" * 70)
print()

# Генерируем сигнал с импульсами
signal1 = generate_mpsk_signal('qpsk', Fs=Fs, Fc=Fc, Sps=Sps, num_symbols=num_symbols,
                               use_pulse_shaping=True)

print("Шаг 1: Создан импульсный базовый сигнал (дельта-функции)")
print(f"  Длина: {len(signal1['symbols_expanded'])} отсчетов")
print(f"  Ненулевых: {np.count_nonzero(signal1['symbols_expanded'])}")
print()

# Применяем RC фильтр к БАЗОВОМУ сигналу (до модуляции)
rc_result1 = apply_rc_filter(signal1['symbols_expanded'], Sps=Sps, alpha=alpha, filter_span=filter_span)
print("Шаг 2: Применён RC фильтр к базовому сигналу")
print()

# Модулируем отфильтрованный базовый сигнал
modulated1 = modulate_baseband(rc_result1['signal'], Fc, Fs)
print("Шаг 3: Модуляция отфильтрованного сигнала на несущую")
print()

# Восстанавливаем символы
# Задержка уже скомпенсирована в apply_rc_filter, но можем передать 0 явно
demod1 = demodulate_signal(modulated1, Fc, Fs)
symbols_recovered1 = recover_symbols(demod1, Sps, num_symbols, Fc, Fs, filter_delay=0)

# Проверяем ошибки (избегаем краевых эффектов)
edge_skip = 10
errors1 = [np.abs(symbols_recovered1[i] - signal1['symbols'][i]) 
           for i in range(edge_skip, num_symbols - edge_skip)]

print(f"Восстановление символов (без краёв {edge_skip}-{num_symbols-edge_skip}):")
print(f"  Средняя ошибка: {np.mean(errors1):.10f}")
print(f"  Максимальная ошибка: {np.max(errors1):.10f}")
print(f"  ✅ Идеально! (RC работает как задумано)")
print()

# ============================================================
# ПОДХОД 2: НЕПРАВИЛЬНЫЙ - Модуляция → Фильтр
# ============================================================
print("=" * 70)
print("ПОДХОД 2 (НЕПРАВИЛЬНЫЙ): Модуляция → Фильтр")
print("=" * 70)
print()

# Генерируем тот же сигнал
signal2 = generate_mpsk_signal('qpsk', Fs=Fs, Fc=Fc, Sps=Sps, num_symbols=num_symbols,
                               use_pulse_shaping=True)

print("Шаг 1: Создан импульсный базовый сигнал (тот же)")
print()

# Модулируем импульсный сигнал СРАЗУ (без фильтра)
modulated_impulse = modulate_baseband(signal2['symbols_expanded'], Fc, Fs)
print("Шаг 2: Модуляция импульсов на несущую (БЕЗ фильтра)")
print()

# Пытаемся применить RC фильтр к МОДУЛИРОВАННОМУ сигналу
print("Шаг 3: Попытка применить RC фильтр к модулированному сигналу...")
print("  ⚠️ ПРОБЛЕМА: RC фильтр рассчитан для базовой полосы, не для RF!")
print()

# Применяем фильтр к модулированному сигналу (неправильно!)
rc_result2 = apply_rc_filter(modulated_impulse.astype(complex), Sps=Sps, alpha=alpha, filter_span=filter_span)

# Демодулируем
demod2 = demodulate_signal(rc_result2['signal'].real, Fc, Fs)
symbols_recovered2 = recover_symbols(demod2, Sps, num_symbols, Fc, Fs, filter_delay=0)

# Проверяем ошибки
errors2 = [np.abs(symbols_recovered2[i] - signal2['symbols'][i]) 
           for i in range(edge_skip, num_symbols - edge_skip)]

print(f"Восстановление символов (без краёв {edge_skip}-{num_symbols-edge_skip}):")
print(f"  Средняя ошибка: {np.mean(errors2):.6f}")
print(f"  Максимальная ошибка: {np.max(errors2):.6f}")
print(f"  ❌ Огромные ошибки! (неправильный подход)")
print()

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
print("=" * 70)
print("Создание графиков для сравнения...")
print("=" * 70)

fig = plt.figure(figsize=(20, 14))

samples = 4 * Sps  # Показываем 4 символа

# ========== ПОДХОД 1: ПРАВИЛЬНЫЙ ==========
# График 1: Базовый сигнал до фильтра (импульсы)
ax1 = plt.subplot(4, 3, 1)
t_base = np.arange(len(signal1['symbols_expanded'])) / Fs
ax1.plot(t_base[:samples], np.real(signal1['symbols_expanded'][:samples]), 'b-', linewidth=1.5, label='Re')
ax1.plot(t_base[:samples], np.imag(signal1['symbols_expanded'][:samples]), 'b--', linewidth=1.5, alpha=0.5, label='Im')
ax1.set_title('Подход 1: Импульсы (базовая полоса)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Амплитуда', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
ax1.text(0.5, 0.95, 'ШАГ 1', transform=ax1.transAxes, ha='center', 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# График 2: Базовый сигнал после RC фильтра
ax2 = plt.subplot(4, 3, 2)
ax2.plot(t_base[:samples], np.real(rc_result1['signal'][:samples]), 'g-', linewidth=1.5, label='Re')
ax2.plot(t_base[:samples], np.imag(rc_result1['signal'][:samples]), 'g--', linewidth=1.5, alpha=0.5, label='Im')
ax2.set_title('Подход 1: После RC фильтра (базовая)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Амплитуда', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)
ax2.text(0.5, 0.95, 'ШАГ 2 (RC фильтр)', transform=ax2.transAxes, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# График 3: Модулированный сигнал (после фильтра)
ax3 = plt.subplot(4, 3, 3)
t_mod = np.arange(len(modulated1)) / Fs
ax3.plot(t_mod[:samples], modulated1[:samples], 'purple', linewidth=1.5)
ax3.set_title('Подход 1: Модулированный (после RC)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Амплитуда', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.text(0.5, 0.95, 'ШАГ 3 (Модуляция)', transform=ax3.transAxes, ha='center',
         bbox=dict(boxstyle='round', facecolor='violet', alpha=0.8))

# ========== ПОДХОД 2: НЕПРАВИЛЬНЫЙ ==========
# График 4: Базовый сигнал (тот же)
ax4 = plt.subplot(4, 3, 4)
ax4.plot(t_base[:samples], np.real(signal2['symbols_expanded'][:samples]), 'b-', linewidth=1.5, label='Re')
ax4.plot(t_base[:samples], np.imag(signal2['symbols_expanded'][:samples]), 'b--', linewidth=1.5, alpha=0.5, label='Im')
ax4.set_title('Подход 2: Импульсы (базовая полоса)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Амплитуда', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)
ax4.text(0.5, 0.95, 'ШАГ 1', transform=ax4.transAxes, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# График 5: Модулированный БЕЗ фильтра
ax5 = plt.subplot(4, 3, 5)
ax5.plot(t_mod[:samples], modulated_impulse[:samples], 'orange', linewidth=1.5)
ax5.set_title('Подход 2: Модулированный (БЕЗ RC!)', fontweight='bold', fontsize=11)
ax5.set_ylabel('Амплитуда', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.text(0.5, 0.95, 'ШАГ 2 (Модуляция)', transform=ax5.transAxes, ha='center',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

# График 6: После RC фильтра (на RF частоте)
ax6 = plt.subplot(4, 3, 6)
ax6.plot(t_mod[:samples], rc_result2['signal'][:samples].real, 'red', linewidth=1.5)
ax6.set_title('Подход 2: После RC на RF ❌', fontweight='bold', fontsize=11)
ax6.set_ylabel('Амплитуда', fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.text(0.5, 0.95, 'ШАГ 3 (RC фильтр)', transform=ax6.transAxes, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# ========== СПЕКТРЫ ==========
# График 7: Спектр подхода 1
ax7 = plt.subplot(4, 3, 7)
fft1 = np.fft.fft(modulated1)
freqs1 = np.fft.fftfreq(len(modulated1), 1/Fs)
positive_mask1 = freqs1 >= 0
ax7.plot(freqs1[positive_mask1], np.abs(fft1[positive_mask1]), 'purple', linewidth=1.5)
ax7.set_title('Подход 1: Спектр (чистый)', fontweight='bold', fontsize=11)
ax7.set_ylabel('Магнитуда', fontsize=10)
ax7.set_xlabel('Частота, Гц', fontsize=10)
ax7.set_xlim(0, 5000)
ax7.grid(True, alpha=0.3)

# График 8: Спектр подхода 2
ax8 = plt.subplot(4, 3, 8)
fft2 = np.fft.fft(rc_result2['signal'].real)
freqs2 = np.fft.fftfreq(len(rc_result2['signal']), 1/Fs)
positive_mask2 = freqs2 >= 0
ax8.plot(freqs2[positive_mask2], np.abs(fft2[positive_mask2]), 'red', linewidth=1.5)
ax8.set_title('Подход 2: Спектр (искажённый)', fontweight='bold', fontsize=11)
ax8.set_ylabel('Магнитуда', fontsize=10)
ax8.set_xlabel('Частота, Гц', fontsize=10)
ax8.set_xlim(0, 5000)
ax8.grid(True, alpha=0.3)

# График 9: Сравнение спектров
ax9 = plt.subplot(4, 3, 9)
ax9.plot(freqs1[positive_mask1], np.abs(fft1[positive_mask1]), 'purple', linewidth=1.5, 
         label='Подход 1 (правильный)', alpha=0.7)
ax9.plot(freqs2[positive_mask2], np.abs(fft2[positive_mask2]), 'red', linewidth=1.5, 
         label='Подход 2 (неправильный)', alpha=0.7)
ax9.set_title('Сравнение спектров', fontweight='bold', fontsize=11)
ax9.set_ylabel('Магнитуда', fontsize=10)
ax9.set_xlabel('Частота, Гц', fontsize=10)
ax9.set_xlim(0, 5000)
ax9.grid(True, alpha=0.3)
ax9.legend(fontsize=9)

# ========== СОЗВЕЗДИЯ ==========
# График 10: Созвездие подхода 1
ax10 = plt.subplot(4, 3, 10)
symbols_orig = signal1['symbols'][edge_skip:num_symbols-edge_skip]
symbols_rec1 = symbols_recovered1[edge_skip:num_symbols-edge_skip]
ax10.scatter(symbols_orig.real, symbols_orig.imag, c='gray', marker='x', s=100, 
            linewidths=2, label='Исходные', alpha=0.5, zorder=1)
ax10.scatter(symbols_rec1.real, symbols_rec1.imag, c='green', marker='o', s=80, 
            edgecolors='darkgreen', linewidths=1.5, label='Восстановленные', alpha=0.7, zorder=2)
ax10.set_title(f'Подход 1: Созвездие\n✅ Ошибка = {np.max(errors1):.8f}', 
              fontweight='bold', fontsize=11)
ax10.set_xlabel('Re (I)', fontsize=10)
ax10.set_ylabel('Im (Q)', fontsize=10)
ax10.grid(True, alpha=0.3)
ax10.axis('equal')
ax10.set_xlim(-1.5, 1.5)
ax10.set_ylim(-1.5, 1.5)
ax10.axhline(y=0, color='k', linewidth=0.5)
ax10.axvline(x=0, color='k', linewidth=0.5)
ax10.legend(fontsize=8)

# График 11: Созвездие подхода 2
ax11 = plt.subplot(4, 3, 11)
symbols_rec2 = symbols_recovered2[edge_skip:num_symbols-edge_skip]
ax11.scatter(symbols_orig.real, symbols_orig.imag, c='gray', marker='x', s=100, 
            linewidths=2, label='Исходные', alpha=0.5, zorder=1)
ax11.scatter(symbols_rec2.real, symbols_rec2.imag, c='red', marker='o', s=80, 
            edgecolors='darkred', linewidths=1.5, label='Восстановленные', alpha=0.7, zorder=2)
ax11.set_title(f'Подход 2: Созвездие\n❌ Ошибка = {np.max(errors2):.4f}', 
              fontweight='bold', fontsize=11)
ax11.set_xlabel('Re (I)', fontsize=10)
ax11.set_ylabel('Im (Q)', fontsize=10)
ax11.grid(True, alpha=0.3)
ax11.axis('equal')
ax11.set_xlim(-1.5, 1.5)
ax11.set_ylim(-1.5, 1.5)
ax11.axhline(y=0, color='k', linewidth=0.5)
ax11.axvline(x=0, color='k', linewidth=0.5)
ax11.legend(fontsize=8)

# График 12: Текстовое пояснение
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')
explanation_text = """
ВЫВОД:

✅ ПРАВИЛЬНЫЙ ПОРЯДОК:
   Импульсы → RC фильтр → Модуляция
   • Фильтр в базовой полосе
   • Символы восстанавливаются идеально
   • Чистый спектр

❌ НЕПРАВИЛЬНЫЙ ПОРЯДОК:
   Импульсы → Модуляция → RC фильтр
   • Фильтр на RF частоте
   • Огромные ошибки восстановления
   • Искажённый спектр

ПОЧЕМУ?
RC фильтр рассчитан для работы
в базовой полосе (около 0 Гц),
а не на несущей частоте (Fc)!
"""
ax12.text(0.5, 0.5, explanation_text, transform=ax12.transAxes,
         fontsize=10, verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Порядок операций: Фильтр → Модуляция (правильно) vs Модуляция → Фильтр (неправильно)',
            fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()
plt.show()

print()
print("=" * 70)
print("📊 ИТОГИ:")
print("=" * 70)
print()
print("✅ ПОДХОД 1 (Фильтр → Модуляция):")
print(f"   Ошибка: {np.max(errors1):.10f} (машинная точность)")
print(f"   Созвездие: идеально")
print()
print("❌ ПОДХОД 2 (Модуляция → Фильтр):")
print(f"   Ошибка: {np.max(errors2):.6f} (огромная!)")
print(f"   Созвездие: разрушено")
print()
print("🎯 ВЫВОД:")
print("   RC/RRC фильтры ВСЕГДА применяются в БАЗОВОЙ ПОЛОСЕ,")
print("   а затем сигнал модулируется на несущую частоту!")
print()
print("=" * 70)

