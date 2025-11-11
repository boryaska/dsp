import numpy as np
import matplotlib.pyplot as plt
def fll_func(signal, Bn = 0.001, y = 1):
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
        decided_sample = (np.sign(np.real(sample)) + 1j * np.sign(np.imag(sample)))/np.sqrt(2)
        
        error = np.angle(sample * np.conj(decided_sample))
        integrator += error * ki
        ef_n = error * kp + integrator
        ef_n_list[i] = ef_n
        curr_freq += ef_n
        curr_freq_list[i] = curr_freq


    return signal_list, curr_freq, curr_freq_list, ef_n_list

if __name__ == "__main__":
    symbols = np.random.choice([0.707 + 1j*0.707, 0.707 - 1j*0.707, -0.707 + 1j*0.707, -0.707 - 1j*0.707], size=50000)

    symbols_offset = symbols * np.exp(1j * 2 * np.pi * 0.002 * np.arange(len(symbols)))

    noise = (np.random.normal(0, 0.01, len(symbols_offset)) + 1j * np.random.normal(0, 0.01, len(symbols_offset))) / np.sqrt(2)

    # fft_len = len(noise)
    # fft_signal_aligned = np.fft.fftshift(np.fft.fft(noise))
    # freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1))
    # plt.figure(figsize=(12, 5))
    # plt.plot(freqs, np.abs(fft_signal_aligned))
    # plt.title("FFT of signal (normalized frequency $[-0.5, 0.5]$)")
    # plt.xlabel("Normalized Frequency ($F/F_d$)")
    # plt.ylabel("Amplitude")
    # plt.xlim(-0.5, 0.5)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    symbols_offset += noise

    # plt.plot(symbols_offset.real, symbols_offset.imag, 'o')
    # plt.show()



    signal_list, curr_freq, curr_freq_list, ef_n_list = fll_func(symbols_offset, 0.002, 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Созвездие сигналов после FLL
    axs[0].plot(signal_list.real, signal_list.imag, 'o', markersize=3, alpha=0.6)
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

    print(curr_freq)

    # fft_len = len(signal_list)
    # fft_signal_aligned = np.fft.fftshift(np.fft.fft(signal_list))
    # freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1))
    # plt.figure(figsize=(12, 5))
    # plt.plot(freqs, np.abs(fft_signal_aligned))
    # plt.title("FFT of signal (normalized frequency after FLL $[-0.5, 0.5]$)")
    # plt.xlabel("Normalized Frequency ($F/F_d$)")
    # plt.ylabel("Amplitude")
    # plt.xlim(-0.5, 0.5)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
