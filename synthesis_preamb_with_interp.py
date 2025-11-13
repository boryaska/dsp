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
pre = generate_preamb(L=300)
# print(pre)
# print(type(pre))

pack = gen_packets(pre, 2000)
print(len(pack))

symbols = gen_packet_symbols(pack, 5, 30)
samples = gen_samples(symbols, sps)
samples = filter_samples(samples, sps)
samples = frequency_offset(samples, 0.01126302)

print(len(samples))
samples = array_shift(samples, shift=0.45, mode='nearest')
print(len(samples))
samples = add_noise(samples, 0.01)




# # print(np.mean(mu_history[-100:]))
# samples = array_shift(samples, shift=-np.mean(mu_history[-100:]), mode='nearest')

# analys.plot_signal(samples)
# analys.plot_constellation(samples, 4)

# print(len(recovered))


offset, signal_aligned, conv_results, conv_max, phase_offset, conv_results_interp = find_preamble_offset_with_interpolate_dots(samples, pre, sps, interp = True)
# print(offset)
# print(conv_max)
# print(phase_offset)

if conv_results_interp:
    # print('проверяем интерполированные корреляции')
    
    corr_abs = np.zeros(sps * 2 * len(conv_results[0]))
    for i in range(len(conv_results)):
        for j in range(len(conv_results[i])):
            corr_abs[j * sps * 2 + i * 2] = np.abs(conv_results[i][j])
            
    for i in range(len(conv_results_interp)):
        for j in range(len(conv_results_interp[i])):
            corr_abs[j * sps * 2 + i * 2 + 1] = np.abs(conv_results_interp[i][j])        
        
else:
    
    corr_abs = np.zeros(sps * len(conv_results[0]))
    for i in range(len(conv_results)):
        for j in range(len(conv_results[i])):
            corr_abs[j * sps + i] = np.abs(conv_results[i][j])    

max_corr = np.max(corr_abs)
# Используем относительные пороги от максимума

plt.figure(figsize=(18, 4))
plt.plot(np.abs(corr_abs))
plt.title('График corr_abs')
plt.xlabel('Индекс')
plt.ylabel('|corr_abs|')
plt.grid(True)
plt.show()
index_offset = defaultdict(int)
peaks, properties = find_peaks(
    corr_abs,
    height=max_corr * 0.7,   
    distance=len(pack) * 0.1,    
    prominence=max_corr * 0.1  )
print('пики корреляции')
print(peaks)      
for peak in peaks:
    index_offset[peak] = corr_abs[peak]
print(f"Фаза: найдено {len(peaks)} пиков, макс. корреляция = {max_corr:.2f}")

print(index_offset)
for peak in peaks:
    if conv_results_interp:
        samples = array_shift(samples, shift=0.5, mode='nearest')
        offset = (peak - 1) // 2
    else:
        offset = peak // 2
    if offset + len(pack) * sps > len(samples):
        continue

    # print(f'offsetв отсчетах: {offset}')

    signal_cutted = samples[(offset) : (offset) + (len(pack) * sps)]
    # print(f'длина сигнала в отсчетах после вырезания: {len(signal_cutted)}')
    signal_for_f_rel = signal_cutted[::sps]
    signal_for_f_rel = signal_for_f_rel[:len(pre)]
    phase_signal = signal_for_f_rel * np.conj(pre)
    phases = np.angle(phase_signal)
    phase_diffs = np.diff((np.unwrap(phases)))
    avg_phase_diff = np.mean(phase_diffs)
    f_rel_method1 = avg_phase_diff / (2 * np.pi * sps)

    for f_rel in [f_rel_method1]:
        shiftted_signal = signal_cutted.copy()
        shiftted_signal = shiftted_signal * np.exp(-1j * 2 * np.pi * f_rel * (np.arange(len(shiftted_signal)) + offset))

        rrc = rrc_filter(sps, 10, 0.35)
        filtred_signal = np.convolve(shiftted_signal, rrc, mode='same')
        filtred_signal = filtred_signal / np.std(filtred_signal)
        # print(f'длина сигнала в отсчетах после фильтрации: {len(filtred_signal)}')

        recovered, timing_errors, mu_history = gardner_timing_recovery(filtred_signal, sps, alpha=0.007, mu_initial=0.0)
        # print(f'длина сигнала в отсчетах после Гарднера: {len(recovered)}')
        signal_list, curr_freq, curr_freq_list, ef_n_list = fll_func(recovered, Bn = 0.003)
        # print(f'длина сигнала в отсчетах после FLL: {len(signal_list)}')
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
            # print('длина пакета совпадает')
            count = 0
            for i in range(len(decision_direct)):
                if decision_direct[i] == pack[i]:
                    count += 1
            if count/len(decision_direct) >= 0.2:
                print(f'пакет принят, {count/len(decision_direct) * 100}% символов совпадают') 
        else:
            print('длина пакета не совпадает')        




