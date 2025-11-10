import numpy as np
import matplotlib.pyplot as plt
def fll_func(signal):
    curr_freq = 0
    kp = 0.00005
    ki = 0.00
    integrator = 0
    phase = 0
    ef_n_list = np.zeros(len(signal), dtype=float)
    signal_list = np.zeros(len(signal), dtype=complex)


    for i in range(len(signal)):
        sample = signal[i]
        phase = 2 * np.pi * curr_freq * i
        sample = sample * np.exp(-1j * phase)
        signal_list[i] = sample
        decided_sample = (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample)))/np.sqrt(2)
        
        error = np.angle(sample * np.conj(decided_sample))
        integrator += error * ki
        ef_n = error * kp + integrator
        ef_n_list[i] = ef_n
        curr_freq += ef_n


    return signal_list, curr_freq, ef_n_list

signal = np.fromfile(f'files/sig_symb_x4_ncr_77671952566_logon_id_1_tx_id_7616.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]
signal_iq = signal_iq/np.std(signal_iq)

from diff_method import rrc_filter
rrc = rrc_filter(4, 10, 0.35)
signal_filtered = np.convolve(signal_iq, rrc, mode='same')
signal_iq = signal_filtered / np.std(signal_filtered)
signal_iq = signal_iq[505*4+3:-1000:4]

plt.plot(signal_iq.real, signal_iq.imag, 'o')
plt.show()


signal_list, curr_freq, ef_n_list = fll_func(signal_iq)

plt.plot(signal_list[500:].real, signal_list[500:].imag, 'o')
plt.show()

plt.plot(ef_n_list)
plt.show()

print(curr_freq)