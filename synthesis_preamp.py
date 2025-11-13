import numpy as np
import matplotlib.pyplot as plt
import analys
from diff_method import rrc_filter, find_preamble_offset
from scipy.signal import find_peaks
from collections import defaultdict
from fll_func import fll_func
from scipy.ndimage import shift as array_shift
from gardner2 import gardner_timing_recovery


def generate_preamb(L=100, PSK = 4):
    if PSK == 4:
        preamb = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707], size=L)
    elif PSK == 2:
        preamb = np.random.choice([-1 + 1j*0, 1 + 1j*0], size=L)
    return preamb

def gen_packets(preamb, L, PSK = 4):
    if PSK == 4:
        inform = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707], size = (L-len(preamb)))
    elif PSK == 2:
        inform = np.random.choice([-1 + 1j*0, 1 + 1j*0], size = (L-len(preamb)))
    packet = np.concatenate([preamb, inform])   
    return packet

def gen_packet_symbols(packet, N, delay = 10):
    symbols = np.zeros(delay)
    for i in range(N):
        symbols = np.concatenate([symbols,  packet, np.zeros(delay, dtype=np.complex64)])
    return symbols

def gen_samples(symbols, sps):         
    samples = np.zeros(len(symbols) * sps, dtype=np.complex64)
    for i in range(len(symbols)):
        samples[i*sps] = symbols[i]
    return samples

def filter_samples(samples, sps):
    rrc = rrc_filter(sps, 10, 0.35)
    samples = np.convolve(samples, rrc, mode='same')
    samples = samples / np.std(samples)
    return samples

def frequency_offset(samples, Frel):
    samples = samples * np.exp(1j * 2 * np.pi * Frel * np.arange(len(samples)))    
    return samples

def add_noise(samples, power = 0.0):
    noise = (np.random.normal(0, power, len(samples)) + 1j * np.random.normal(0, power, len(samples))) / np.sqrt(2)
    return samples + noise

def decision_direct_func(signal):
    decision_list = []
    for sample in signal:
        decided_sample = (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample)))*0.707
        decision_list.append(decided_sample)
    return decision_list

def find_preamble_offset_with_interpolate_dots(signal_iq, preamble_iq, sps, interp = False):
    """
    Поиск начала преамбулы в сигнале методом дифференциальной корреляции.
    
    Параметры:
    ----------
    signal_iq : np.ndarray
        Входной сигнал (комплексный IQ)
    preamble_iq : np.ndarray
        Известная преамбула (комплексные символы)
    sps : int
        Samples per symbol (количество отсчетов на символ)
    
    Возвращает:
    -----------
    offset : int
        Смещение в символах, где найдена преамбула
    signal_aligned : np.ndarray
        Сигнал, обрезанный с найденного offset
    conv_results : list
        Результаты корреляции для каждой фазы (для отладки/визуализации)
    conv_max : list
        Позиции максимумов корреляции для каждой фазы
    
    Пример использования:
    --------------------
    >>> offset, aligned_signal, corr_results, corr_max = find_preamble_offset(
    ...     signal_iq, preamble_iq, sps=4
    ... )
    >>> print(f"Преамбула найдена на позиции {offset} символов")
    # """
    # Дифференциальное произведение для преамбулы
    d_pre = preamble_iq[1:] * np.conj(preamble_iq[:-1])
    
    # Дифференциальная корреляция для каждой фазы
    d_sig_list = []
    d_sig_list_interp = []
    if interp:
        interpolated_samples = array_shift(signal_iq, shift=0.5, mode='nearest')
        for phase in range(sps):
            sig_symbols = interpolated_samples[phase::sps]
            d_sig = sig_symbols[1:] * np.conj(sig_symbols[:-1])
            d_sig_list_interp.append(d_sig)    
            
    for phase in range(sps):
        # Берем символы с шагом sps, начиная с фазы phase
        sig_symbols = signal_iq[phase::sps]
        d_sig = sig_symbols[1:] * np.conj(sig_symbols[:-1])
        d_sig_list.append(d_sig)
    

    # Корреляция с нормализацией для каждой фазы
    conv_results = []
    conv_results_interp = []
    conv_max = []
    conv_max_interp = []
    if interp:
        for phase in range(sps):
            signal = d_sig_list_interp[phase]
            conv = np.zeros(len(signal), dtype=np.complex64)
        
            # Скользящая корреляция с нормализацией на СКО
            for k in range(len(signal) - len(d_pre) + 1):
                curr_signal = signal[k:k + len(d_pre)]
                Y = np.std(curr_signal)
                if Y > 0:  # Защита от деления на ноль
                    conv[k] = np.sum(curr_signal * np.conj(d_pre)) / Y
                else:
                    conv[k] = 0
        
            conv_results_interp.append(conv) 
            conv_max_interp.append((np.argmax(np.abs(conv)),np.max(np.abs(conv))))
    for phase in range(sps):
        signal = d_sig_list[phase]
        conv = np.zeros(len(signal), dtype=np.complex64)
        
        # Скользящая корреляция с нормализацией на СКО
        for k in range(len(signal) - len(d_pre) + 1):
            curr_signal = signal[k:k + len(d_pre)]
            Y = np.std(curr_signal)
            if Y > 0:  # Защита от деления на ноль
                conv[k] = np.sum(curr_signal * np.conj(d_pre)) / Y
            else:
                conv[k] = 0
        
        conv_results.append(conv) 
        conv_max.append((np.argmax(np.abs(conv)),np.max(np.abs(conv))))
    
    # Находим фазу с максимальным значением корреляции
    values = [val for idx, val in conv_max]
    offset_index = int(np.argmax(values))
    offset = conv_max[offset_index][0]
    phase_offset = offset_index
    # Обрезаем сигнал с найденного смещения
    signal_aligned = signal_iq[offset * sps + phase_offset:]
    
    return offset, signal_aligned, conv_results, conv_max, phase_offset, conv_results_interp
sps = 4
pre = generate_preamb(L=500)
# print(pre)
# print(type(pre))

pack = gen_packets(pre, 3000)
print(len(pack))

symbols = gen_packet_symbols(pack, 5, 30)
samples = gen_samples(symbols, sps)
samples = filter_samples(samples, sps)
samples = frequency_offset(samples, 0.01716302)


samples = array_shift(samples, shift=0.61, mode='nearest')


samples = add_noise(samples, 0.1)

print(len(symbols))
# print(symbols)
analys.plot_constellation(samples, 4)
analys.plot_signal(samples)


# recovered, timing_errors, mu_history = gardner_timing_recovery(samples, sps, alpha=0.007, mu_initial=0.0)
# # print(np.mean(mu_history[-100:]))
# samples = array_shift(samples, shift=-np.mean(mu_history[-100:]), mode='nearest')

analys.plot_signal(samples)
analys.plot_constellation(samples, 4)

# print(len(recovered))
print(len(samples))

offset, signal_aligned, conv_results, conv_max, phase_offset, conv_results_interp = find_preamble_offset_with_interpolate_dots(samples, pre, sps, interp = True)
# print(offset)
# print(conv_max)
# print(phase_offset)


index_offset = defaultdict(lambda: [0, 0, 0, 0])

for i in range(sps):
    corr_abs = abs(conv_results[i])
    max_corr = np.max(corr_abs)
    
    # Используем относительные пороги от максимума
    peaks, properties = find_peaks(
        corr_abs,
        height=max_corr * 0.7,   
        distance=len(pack) * 0.1,    
        prominence=max_corr * 0.1  )  
    for peak in peaks:
        index_offset[peak][i] = corr_abs[peak]
    print(f"Фаза {i}: найдено {len(peaks)} пиков, макс. корреляция = {max_corr:.2f}")

try_offset_phase = []
for key, value in index_offset.items():
    try_offset_phase.append((key, np.argmax(value)))
    # print(key, value)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Индексы пиков для каждой фазы (sps=4)', fontsize=14)

for i in range(4):
    row = i // 2
    col = i % 2
    ax = axes2[row, col]
    ax.plot(abs(conv_results[i]))
    ax.set_title(f'Фаза {i}')
    ax.set_xlabel('Индекс')

plt.tight_layout()
plt.show()        
f_rel_list = []
print('try_offset_phase: ', try_offset_phase)
n = 0
for offset, phase in try_offset_phase:
    n += 1
    print(f'{n} / {len(try_offset_phase)}') 
    if offset * sps + phase + len(pack) * sps > len(samples):
        continue

    signal_cutted = samples[(offset * sps + phase) : (offset * sps + phase) + (len(pack) * sps)] 
    
    signal_for_f_rel = signal_cutted[::sps]
    # print(f"Длина signal_for_f_rel: {len(signal_for_f_rel)}")

    signal_for_f_rel = signal_for_f_rel[:len(pre)]
    phase_signal = signal_for_f_rel * np.conj(pre)
    phases = np.angle(phase_signal)
    phase_diffs = np.diff((np.unwrap(phases)))
    avg_phase_diff = np.mean(phase_diffs)
    f_rel_method1 = avg_phase_diff / (2 * np.pi * sps)
    f_rel_list.append(f_rel_method1)

    # n = np.arange(len(phases))
    # unwrapped_phases = np.unwrap(phases)
    # slope = np.polyfit(n, unwrapped_phases, 1)[0] 
    # f_rel_method2 = slope / (2 * np.pi * sps)

    # prod = phase_signal[1:] * np.conj(phase_signal[:-1])
    # avg_rotation = np.angle(np.mean(prod))
    # f_rel_method3 = avg_rotation / (2 * np.pi * sps)

    for f_rel in [f_rel_method1]:
        shiftted_signal = signal_cutted.copy()
        shiftted_signal = shiftted_signal * np.exp(-1j * 2 * np.pi * f_rel * (np.arange(len(shiftted_signal)) + offset * sps + phase) )
        # analys.plot_constellation(shiftted_signal, 1)

        rrc = rrc_filter(sps, 10, 0.35)
        filtred_signal = np.convolve(shiftted_signal, rrc, mode='same')
        filtred_signal = filtred_signal / np.std(filtred_signal)

        signal_list, curr_freq, curr_freq_list, ef_n_list = fll_func(filtred_signal[::sps], Bn = 0.003)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Созвездие сигналов после FLL
        axs[0].plot(signal_list[500:].real, signal_list[500:].imag, 'o', markersize=3, alpha=0.6)
        axs[0].set_title("Созвездие после FLL")
        axs[0].set_xlabel('I (Real)')
        axs[0].set_ylabel('Q (Imag)')
        axs[0].grid(True, alpha=0.3)
        axs[0].axis('equal')

        # Текущая накопленная оценка частоты (вектор по времени)
        axs[1].plot(curr_freq_list)
        axs[1].set_title("curr_freq (после FLL)")
        axs[1].set_xlabel('n')
        axs[1].set_ylabel('curr_freq')

        # Эфф. изменение частоты во времени
        axs[2].plot(ef_n_list)
        axs[2].set_title("Эфф. изменение частоты (ef_n_list)")
        axs[2].set_xlabel('n')
        axs[2].set_ylabel('ef_n')

        plt.tight_layout()
        plt.show()

        decision_direct = decision_direct_func(signal_list)
        if len(decision_direct) == len(pack):
            count = 0
            for i in range(len(decision_direct)):
                if decision_direct[i] == pack[i]:
                    count += 1
            if count/len(decision_direct) >= 0.3:
                print(f'пакет принят, {count/len(decision_direct) * 100}% символов совпадают')
                
        
        

print(f_rel_list)

