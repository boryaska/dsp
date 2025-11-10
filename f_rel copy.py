import numpy as np
import matplotlib.pyplot as plt
from diff_method import rrc_filter
from diff_method import find_preamble_offset
from gardner2 import gardner_timing_recovery
# from fll import FrequencyLockedLoop

signal = np.fromfile('files/sig_symb_x4_ncr_77671952566_logon_id_1_tx_id_7616.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]

preamble = np.fromfile('files/preambule_logon_id_2_tx_id_7616_float32.pcm', dtype=np.float32)
preamble_iq = preamble[::2] + 1j * preamble[1::2]

signal_aligned = signal_iq*np.exp((-1j)*2*np.pi*0.11*np.arange(len(signal_iq)))

# Фильтрация сигналa ФНЧ с частотой среза 0.17 (отн. к дискретизации)
from scipy.signal import firwin, lfilter

cutoff = 0.17  # норм. частота среза (отн. к Fs=1)
numtaps = 101  # длина фильтра (можно варьировать)
lpf_taps = firwin(numtaps, cutoff, window='hamming')

# Применить ФНЧ к выровненному сигналу
signal_aligned = lfilter(lpf_taps, 1.0, signal_aligned)

# rrc = rrc_filter(4, 10, 0.35)
# signal_filtered = np.convolve(signal_aligned, rrc, mode='same')
# signal_filtered = signal_filtered / np.std(signal_filtered)

offset, signal_iq, conv_results, conv_max , phase_offset= find_preamble_offset(signal_aligned, preamble_iq, 4)
print(conv_max)

signal_for_f_rel = signal_iq[::4]
plt.figure(figsize=(10, 10))
plt.plot(signal_for_f_rel.real, signal_for_f_rel.imag, 'o', markersize=3, alpha=0.6)
plt.title(f'Созвездие \n(f_rel = )')
plt.xlabel('I (Real)')
plt.ylabel('Q (Imag)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()

signal_for_f_rel = signal_for_f_rel[:len(preamble_iq)]

phase_signal = signal_for_f_rel * np.conj(preamble_iq)

phases = np.angle(phase_signal)
print(f"phases: {phases}")

phase_diffs = np.diff((np.unwrap(phases)))
print(f"phase_diffs: {phase_diffs}")
avg_phase_diff = np.mean(phase_diffs)
print(f"avg_phase_diff: {avg_phase_diff}")
f_rel_method1 = avg_phase_diff / (2 * np.pi * 4)
print(f"f_rel_method ----: {f_rel_method1}")


plt.figure(figsize=(10, 10))
plt.plot(phase_signal.real[:5], phase_signal.imag[:5], 'o', markersize=3, alpha=0.6)
plt.title(f'Созвездие фаз\n(f_rel = )')
plt.xlabel('I (Real)')
plt.ylabel('Q (Imag)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


print(f_rel_method1)

# 4. МЕТОД 2: Линейная регрессия (более точный)
n = np.arange(len(phases))
unwrapped_phases = np.unwrap(phases)

# Наклон прямой φ(n) = 2π·f_rel·n + φ₀
# Используем МНК: slope = Σ(n·φ) / Σ(n²)
slope = np.polyfit(n, unwrapped_phases, 1)[0]  # коэффициент при n
f_rel_method2 = slope / (2 * np.pi*4)
print(f_rel_method2)

# 5. МЕТОД 3: Через произведение соседних отсчетов (без unwrap)
# Более устойчив к циклическим скачкам фазы
prod = phase_signal[1:] * np.conj(phase_signal[:-1])
avg_rotation = np.angle(np.mean(prod))
f_rel_method3 = avg_rotation / (2 * np.pi*4)
print(f_rel_method3)


for i in [f_rel_method1, f_rel_method2, f_rel_method3]:
    signal = signal_iq.copy()
    n = np.arange(len(signal))
    signal = signal * np.exp(-1j * 2 * np.pi * i * n)
    signal = signal[0::4]
 
    plt.figure(figsize=(10, 10))
    plt.plot(signal.real[:-100], signal.imag[:-100], 'o', markersize=3, alpha=0.6)
    plt.title(f'Созвездие после коррекции частоты\n(f_rel = {i:.6f})')
    plt.xlabel('I (Real)')
    plt.ylabel('Q (Imag)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

n = np.arange(len(signal_iq))
# print(f"Длина n: {len(n)}")
signal_shifted =  signal_iq * np.exp(-1j * 2 * np.pi * f_rel_method2 * n)
# plt.figure(figsize=(10, 10))
# plt.plot(signal_shifted.real[:-400:4], signal_shifted.imag[:-400:4], 'o', markersize=3, alpha=0.6)
# plt.title(f'Созвездие после коррекции частоты\n(f_rel = {f_rel_method2:.6f})')
# plt.xlabel('I (Real)')
# plt.ylabel('Q (Imag)')
# plt.grid(True, alpha=0.3)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()   


# signal_recovered, errors, mu_history = gardner_timing_recovery(signal_shifted[:-400], 4, alpha=0.05)
# print(f"Последнее значение mu: {mu_history[-1]:.4f}")

# signal_recovered = signal_recovered*np.exp(1j * 2 * np.pi * 0.022)

# Создаем FLL с разными методами
methods = ['cross_product', 'atan2', 'decision_directed']
sps = 4

for method in methods:
    print(f"\n{'='*60}")
    print(f"Метод: {method}")
    print(f"{'='*60}")
    
    # Создаем FLL
    fll = FrequencyLockedLoop(
        detector_method=method,
        Kp=0.004,  # пропорциональный коэффициент
        Ki=0.00005,  # интегральный коэффициент
        freq_limit=0.2

    )
    
    # Обрабатываем сигнал
    corrected_signal, freq_estimate = fll.process_signal(signal_shifted)
    
    print(f"Оценка частоты FLL: {freq_estimate:.6f}")
    print(f"Ошибка оценки: {abs(freq_estimate - f_rel_method2):.6f}")
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # До коррекции
    axes[0].plot(signal_shifted[::sps].real, 
                signal_shifted[::sps].imag, 
                'o', markersize=4, alpha=0.5)
    axes[0].set_title(f'До FLL (f_offset={f_rel_method2:.4f})')
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # После коррекции
    axes[1].plot(corrected_signal[100::sps].real, 
                corrected_signal[100::sps].imag, 
                'o', markersize=4, alpha=0.5)
    axes[1].set_title(f'После FLL (метод: {method}, оценка={freq_estimate:.6f})')
    axes[1].set_xlabel('I')
    axes[1].set_ylabel('Q')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.show()




# plt.figure(figsize=(10, 10))
# plt.plot(signal_recovered.real, signal_recovered.imag, 'o', markersize=3, alpha=0.6)
# plt.title(f'Созвездие после коррекции частоты\n(f_rel = {f_rel_method2:.6f})')
# plt.xlabel('I (Real)')
# plt.ylabel('Q (Imag)')
# plt.grid(True, alpha=0.3)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

# # График mu (временной сдвиг)
# plt.figure(figsize=(12, 4))
# plt.plot(mu_history)
# plt.title('История значений mu (временной сдвиг интерполяции)')
# plt.xlabel('Номер символа')
# plt.ylabel('mu')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



