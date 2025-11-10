import numpy as np
import matplotlib.pyplot as plt
def fll_func(signal):
    curr_freq = 0
    kp = 0.0001
    ki = 0.0001
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


symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 4000) / np.sqrt(2)

plt.plot(symbols.real, symbols.imag, 'o')
plt.show()


symbols_offset = symbols * np.exp(1j * 2 * np.pi * 0.005 * np.arange(len(symbols)))

# noise = (np.random.randn(len(symbols_offset)) + 1j * np.random.randn(len(symbols_offset))) * 0.003

# symbols_offset += noise



plt.plot(symbols_offset.real, symbols_offset.imag, 'o')
plt.show()

signal_list, curr_freq, ef_n_list = fll_func(symbols_offset)

plt.plot(signal_list[500:].real, signal_list[500:].imag, 'o')
plt.show()

plt.plot(ef_n_list)
plt.show()

print(curr_freq)