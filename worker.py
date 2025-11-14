import numpy as np
from diff_method import rrc_filter
from scipy.ndimage import shift as array_shift
from scipy.signal import find_peaks
from gardner2 import gardner_timing_recovery
class Processing:
    def __init__(self, signal = None, preamble = None, packet_symb = None, sps = 4, constellation = 'qpsk', interp = False):
        self.signal = signal
        self.preamble = preamble
        # packet_symb - символы преамбулы + информационные
        self.packet_symb = packet_symb
        self.sps = sps
        self.constellation = constellation
        self.interp = interp

    def fll_func(self, signal, Bn = 0.001, y = 1):
        curr_freq = 0
        kp = 4*Bn/1000/(y + 1/(4*y))
        ki = 4*(Bn/1000/(y + 1/(4*y)))**2
        integrator = 0
        phase = 0
        ef_n_list = np.zeros(len(signal), dtype=float)
        curr_freq_list = np.zeros(len(signal), dtype=float)
        signal_list = np.zeros(len(signal), dtype=complex)


        for i in range(len(signal)):
            sample = signal[i]
            phase = 2 * np.pi * curr_freq * i
            sample = sample * np.exp(-1j * phase)
            signal_list[i] = sample
            # decide sample
            if self.constellation == 'bpsk':
                decided_sample = (np.sign(np.real(sample)) + 1j * 0)
            elif self.constellation == 'qpsk':
                decided_sample = (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample))) / np.sqrt(2)
            elif self.constellation == '8psk':
                angle = np.angle(sample)
                decid =  [np.pi/4 * np.arange(8) - np.pi ]
                for d in decid:
                    if abs((d - angle) % np.pi) <= np.pi/8:
                        decided_sample = np.exp(1j * d)
                        break
            elif self.constellation == '16psk':
                angle = np.angle(sample)
                decid =  [np.pi/8 * np.arange(16) - np.pi ]
                for d in decid:
                    if abs((d - angle) % np.pi) <= np.pi/16:
                        decided_sample = np.exp(1j * d)
                        break
            elif self.constellation == '16apsk':
                angle = np.angle(sample)
                if abs(sample) *  2 - 1 > 0.5:
                    decid =  [np.pi/6 * np.arange(12) - np.pi + np.pi/12]
                    for d in decid:
                        if abs((d - angle) % np.pi) <= np.pi/12:
                            decided_sample = np.exp(1j * d)
                            
                elif abs(sample) *  2 - 1 <= 0.5:
                    decided_sample = (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample))) / np.sqrt(2)  
                
            error = np.angle(sample * np.conj(decided_sample))
            integrator += error * ki
            ef_n = error * kp + integrator
            ef_n_list[i] = ef_n
            curr_freq += ef_n
            curr_freq_list[i] = curr_freq        


        return signal_list, curr_freq, curr_freq_list, ef_n_list

    def decision_direct(self, signal_list):

        # Определение символов в зависимости от созвездия, логика из fll_func

        def decide_symbol(sample, constellation):
            if constellation == 'bpsk':
                return (np.sign(np.real(sample)) + 1j * 0)
            elif constellation == 'qpsk':
                return (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample)))/np.sqrt(2)
            elif constellation == '8psk':
                angle = np.angle(sample)
                decid =  [np.pi/4 * np.arange(8) - np.pi ]
                for d in decid:
                    if abs((d - angle) % np.pi) <= np.pi/8:
                        return np.exp(1j * d)
            elif constellation == '16psk':
                angle = np.angle(sample)
                decid =  [np.pi/8 * np.arange(16) - np.pi ]
                for d in decid:
                    if abs((d - angle) % np.pi) <= np.pi/16:
                        return np.exp(1j * d)
            elif constellation == '16apsk':
                angle = np.angle(sample)
                if abs(sample) *  2 - 1 > 0.5:
                    decid =  [np.pi/6 * np.arange(12) - np.pi + np.pi/12]
                    for d in decid:
                        if abs((d - angle) % np.pi) <= np.pi/12:
                            return np.exp(1j * d)
                            
                elif abs(sample) *  2 - 1 <= 0.5:
                    return (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample))) / np.sqrt(2) / 2   
                    
        decision_direct = np.zeros(len(signal_list), dtype=complex)
        for i in range(len(signal_list)):
            decision_direct[i] = decide_symbol(signal_list[i], self.constellation)
        if self.packet_symb:
            if len(self.packet_symb) == len(decision_direct): 
                count = 0   
                for i in range(len(decision_direct)):
                    if decision_direct[i] == self.packet_symb[i]:
                        count += 1
                if count/len(decision_direct) >= 0.5:
                    print(f'пакет принят, {count/len(decision_direct) * 100}% символов совпадают')
                else:
                    for phase in range (16):
                        count = 0
                        for i in range(len(decision_direct)):
                            decision_direct[i] = decision_direct[i] * np.exp(1j * 2 * np.pi / 16)
                            if decision_direct[i] == self.packet_symb[i]:
                                count += 1
                        if count/len(decision_direct) >= 0.5:
                            print(f'пакет принят, {count/len(decision_direct) * 100}% символов совпадают, сдвиг фазы: {(phase + 1) * 22.5} градусов')
                        
        
    def filter_samples(samples, sps):
        rrc = rrc_filter(sps, 10, 0.35)
        samples = np.convolve(samples, rrc, mode='same')
        samples = samples / np.std(samples)
        return samples   
    
    def find_preamble_offset_with_interpolate_dots(self, signal_iq, preamble_iq, sps, interp = False):

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
    
        return conv_results, conv_results_interp

    def find_conv_peaks(self, conv_results, conv_results_interp, interp = False):

        if conv_results_interp:
            corr_abs = np.zeros(self.sps * 2 * len(conv_results[0]))
            for i in range(len(conv_results)):
                for j in range(len(conv_results[i])):
                    corr_abs[j * self.sps * 2 + i * 2] = np.abs(conv_results[i][j])
                    
            for i in range(len(conv_results_interp)):
                for j in range(len(conv_results_interp[i])):
                    corr_abs[j * self.sps * 2 + i * 2 + 1] = np.abs(conv_results_interp[i][j])        
                
        else:
            
            corr_abs = np.zeros(self.sps * len(conv_results[0]))
            for i in range(len(conv_results)):
                for j in range(len(conv_results[i])):
                    corr_abs[j * self.sps + i] = np.abs(conv_results[i][j])

        max_corr = np.max(corr_abs)

        peaks, properties = find_peaks(
            corr_abs,
            height=max_corr * 0.7,
            # сделать длину пакета в отсчетах вместо samples   
            distance=len(self.packet_symb) * 0.1,    
            prominence=max_corr * 0.1  )

        return peaks


    def process(self):
        conv_results, conv_results_interp = self.find_preamble_offset_with_interpolate_dots(self.signal, self.preamble, self.sps, self.interp)
        peaks = self.find_conv_peaks(conv_results, conv_results_interp, self.interp)

        for peak in peaks:
            if conv_results_interp:
                signal_iq = array_shift(self.signal, shift=0.5, mode='nearest')
                offset = (peak - 1) // 2
            else:
                offset = peak // 2
                signal_iq = self.signal
            if offset + len(self.packet_symb) * self.sps > len(signal_iq):
                continue
            
            signal_cutted = signal_iq[(offset) : (offset) + (len(self.packet_symb) * self.sps)]
            signal_for_f_rel = signal_cutted[::self.sps]
            signal_for_f_rel = signal_for_f_rel[:len(self.preamble)]
            phase_signal = signal_for_f_rel * np.conj(self.preamble)
            phases = np.angle(phase_signal)
            phase_diffs = np.diff((np.unwrap(phases)))
            avg_phase_diff = np.mean(phase_diffs)
            f_rel = avg_phase_diff / (2 * np.pi * self.sps)
            shiftted_signal = signal_cutted.copy()
            shiftted_signal = shiftted_signal * np.exp(-1j * 2 * np.pi * f_rel * (np.arange(len(shiftted_signal)) + offset))
            filtred_signal = self.filter_samples(shiftted_signal, self.sps)
            recovered, timing_errors, mu_history = gardner_timing_recovery(filtred_signal, self.sps, alpha=0.007, mu_initial=0.0)
            signal_list, curr_freq, curr_freq_list, ef_n_list = self.fll_func(recovered, Bn = 0.012)
            
            
            
            
        return peaks

class Generation:
    def __init__(self, constellation, preamble_length, signal_length, sps, Frel, Noise_power, N, Delay):
        self.constellation = constellation
        self.preamble_length = preamble_length
        self.signal_length = signal_length
        self.sps = sps
        self.Frel = Frel
        self.Noise_power = Noise_power
        self.N = N
        self.Delay = Delay

    def generate(self):
        preamble = self.generate_preamble()
        samples, symbols = self.generate_signal(preamble)
        # print(f"samples: {samples}")
        samples = self.upsample(samples)
        # samples = self.filter_samples(samples)
        samples = self.frequency_offset(samples)
        samples = self.add_noise(samples)
        return preamble, samples, symbols

    def generate_preamble(self):
        if self.constellation == 'bpsk':
            preamble = np.random.choice([-1 + 0j, 1 + 0j], size=self.preamble_length)
        elif self.constellation == 'qpsk':
            preamble = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707], size=self.preamble_length)
        elif self.constellation == '8psk':
            preamble = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707, 0 + 1j, 0 - 1j, 1 + 0j, -1 + 0j], size=self.preamble_length)
        elif self.constellation == '16psk':
             preamble = np.random.choice([np.exp(1j*np.pi/8), np.exp(2j*np.pi/8), np.exp(3j*np.pi/8),
            np.exp(4j*np.pi/8), np.exp(5j*np.pi/8), np.exp(6j*np.pi/8), np.exp(7j*np.pi/8), np.exp(8j*np.pi/8),
             np.exp(9j*np.pi/8), np.exp(10j*np.pi/8), np.exp(11j*np.pi/8), np.exp(12j*np.pi/8), np.exp(13j*np.pi/8), 
             np.exp(14j*np.pi/8), np.exp(15j*np.pi/8), np.exp(16j*np.pi/8)], size=self.preamble_length)
        elif self.constellation == '16apsk':
             preamble = np.random.choice(0.5 * np.exp(1j * (np.pi/4 +  np.arange(4) * np.pi/2)), 1 * np.exp(1j * (np.pi/4 + np.arange(12) * np.pi/6)), size=self.preamble_length)
        else:
            raise ValueError(f"Неподдерживаемая констелляция: {self.constellation}")
        return preamble

    def generate_signal(self, preamble):
        if self.constellation == 'bpsk':
            inform = np.random.choice([-1 + 0j, 1 + 0j], size=self.signal_length - self.preamble_length)
        elif self.constellation == 'qpsk':
            inform = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707], size=self.signal_length - self.preamble_length)
        elif self.constellation == '8psk':
            inform = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707, 0 + 1j, 0 - 1j, 1 + 0j, -1 + 0j], size=self.signal_length - self.preamble_length)
        elif self.constellation == '16psk': 
             inform = np.random.choice([np.exp(1j*np.pi/8), np.exp(2j*np.pi/8), np.exp(3j*np.pi/8),
            np.exp(4j*np.pi/8), np.exp(5j*np.pi/8), np.exp(6j*np.pi/8), np.exp(7j*np.pi/8), np.exp(8j*np.pi/8),
             np.exp(9j*np.pi/8), np.exp(10j*np.pi/8), np.exp(11j*np.pi/8), np.exp(12j*np.pi/8), np.exp(13j*np.pi/8), 
             np.exp(14j*np.pi/8), np.exp(15j*np.pi/8), np.exp(16j*np.pi/8)], size=self.signal_length - self.preamble_length)
        elif self.constellation == '16apsk':
             inform = np.random.choice(0.5 * np.exp(1j * (np.pi/4 +  np.arange(4) * np.pi/2)), 1 * np.exp(1j * (np.pi/12 + np.arange(12) * np.pi/6)), size=self.signal_length - self.preamble_length)
        else:
            raise ValueError(f"Неподдерживаемая констелляция: {self.constellation}")

        symbols = np.concatenate([preamble, inform])

        signal = np.zeros(self.Delay, dtype=np.complex64)
        for i in range(self.N):
            signal = np.concatenate([signal,  symbols, np.zeros(self.Delay, dtype=np.complex64)])

        return signal, symbols

    def upsample(self, symbols):
        samples = np.zeros(len(symbols) * self.sps, dtype=np.complex64)
        for i in range(len(symbols)):
            samples[i * self.sps] = symbols[i]
        return samples

    def filter_samples(self, samples):
        rrc = rrc_filter(self.sps, 10, 0.35)
        samples = np.convolve(samples, rrc, mode='same')
        samples = samples / np.std(samples)
        return samples   

    def frequency_offset(self, samples):
        samples = samples * np.exp(1j * 2 * np.pi * self.Frel * np.arange(len(samples)))    
        return samples

    def add_noise(self, samples):
        noise = (np.random.normal(0, self.Noise_power, len(samples)) + 1j * np.random.normal(0, self.Noise_power, len(samples))) / np.sqrt(2)
        return samples + noise         