import numpy as np
import matplotlib.pyplot as plt
from diff_method import rrc_filter, find_preamble_offset
from estimate_freq import estimate_initial_frequency


def plot_fft(signal_iq):
    fft_len = len(signal_iq)
    fft_signal_aligned = np.fft.fftshift(np.fft.fft(signal_iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1))
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, np.abs(fft_signal_aligned))
    plt.title("FFT of signal (normalized frequency $[-0.5, 0.5]$)")
    plt.xlabel("Normalized Frequency ($F/F_d$)")
    plt.ylabel("Amplitude")
    plt.xlim(-0.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_signal(signal_iq):
    plt.figure(figsize=(12, 5))
    plt.plot(np.real(signal_iq), label="Real part")
    plt.plot(np.imag(signal_iq), label="Imag part")
    plt.title("Signal Components vs. Time (sample index)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_constellation(signal_iq, sps):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Созвездия для каждой фазы (sps={sps})', fontsize=14)
    for i in range(sps):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        signal_iq_symbols = signal_iq[i::sps]
        ax.plot(signal_iq_symbols[:-400].real, signal_iq_symbols[:-400].imag, 'o', markersize=2, alpha=0.5)
        ax.set_title(f'Фаза {i} (N={len(signal_iq_symbols)})')
        ax.set_xlabel('I (Real)')
        ax.set_ylabel('Q (Imag)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":

    signal = np.fromfile(f'files/sig_symb_x4_ncr_77671952566_logon_id_1_tx_id_7616.pcm', dtype=np.float32)
    signal_iq = signal[::2] + 1j * signal[1::2]
    signal_iq = signal_iq/np.std(signal_iq)


    plot_signal(signal_iq)
    plot_fft(signal_iq)
    plot_constellation(signal_iq, 4)
    preamble = np.fromfile(f'files/preambule_logon_id_2_tx_id_7616_float32.pcm', dtype=np.float32)
    preamble_iq = preamble[::2] + 1j * preamble[1::2]
    print(len(preamble_iq))



    plt.figure(figsize=(12, 5))
    plt.plot(np.real(signal_iq), label="Real part")
    plt.plot(np.imag(signal_iq), label="Imag part")
    plt.title("Signal Components vs. Time (sample index)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    offset, signal_aligned, conv_results, conv_max, phase_offset = find_preamble_offset(signal_iq[:len(signal_iq)//2], preamble_iq, 4)
    print(offset)
    print(conv_results)
    print(conv_max)
    print(phase_offset)

    signal_aligned = signal_iq[(offset-1)*4+0::4]




    # rrc = rrc_filter(4, 10, 0.35)
    # signal_filtered = np.convolve(signal_aligned, rrc, mode='same')
    # signal_aligned = signal_filtered / np.std(signal_filtered)

    avg_freq = estimate_initial_frequency(signal_aligned[::4], 2000)
    print(f"avg_freq: {avg_freq:.6f}")



    # fft_len = len(signal_iq)
    # fft_signal_aligned = np.fft.fftshift(np.fft.fft(signal_iq))
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

    signal_iq = signal_iq[offset*4+phase_offset:42000]  

    fft_len = len(signal_iq)
    fft_signal_aligned = np.fft.fftshift(np.fft.fft(signal_iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1))  # d=1, частота дискретизации нормирована


    plt.figure(figsize=(12, 5))
    plt.plot(freqs, np.abs(fft_signal_aligned))
    plt.title("FFT of signal (normalized frequency $[-0.5, 0.5]$)")
    plt.xlabel("Normalized Frequency ($F/F_d$)")
    plt.ylabel("Amplitude")
    plt.xlim(-0.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    signal_aligned = signal_iq * np.exp(-1j * 2 * np.pi * 0.01 * np.arange(len(signal_iq)))
    # Добавление ФНЧ (низкочастотного фильтра) к сигналу

    from scipy.signal import firwin, lfilter

    fs = 1.0   # частота дискретизации (нормирована, тк частоты в спектре нормированы)
    cutoff_hz = 0.2  # допустим, пропускаем до F/Fd = 0.15
    numtaps = 101     # число коэффициентов ФНЧ (чем больше, тем круче фильтр, например, 51-201)

    # Расчет коэффициентов ФНЧ
    lpf_coeff = firwin(numtaps, cutoff=cutoff_hz, window='hamming', fs=fs)

    # Применение фильтра к сигналу
    signal_aligned = lfilter(lpf_coeff, 1.0, signal_aligned)



    ft_len = len(signal_aligned)
    fft_signal_aligned = np.fft.fftshift(np.fft.fft(signal_aligned))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1))  # d=1, частота дискретизации нормирована


    plt.figure(figsize=(12, 5))
    plt.plot(freqs, np.abs(fft_signal_aligned))
    plt.title("FFT of signal (normalized frequency filtred $[-0.5, 0.5]$)")
    plt.xlabel("Normalized Frequency ($F/F_d$)")
    plt.ylabel("Amplitude")
    plt.xlim(-0.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    signal_for_f_rel = signal_aligned[::4]
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
        signal = signal_aligned.copy()
        n = np.arange(len(signal))
        signal = signal * np.exp(-1j * 2 * np.pi * i * n)
        signal = signal[0::4]
    
        plt.figure(figsize=(10, 10))
        plt.plot(signal.real[:-1000], signal.imag[:-1000], 'o', markersize=3, alpha=0.6)
        plt.title(f'Созвездие после коррекции частоты\n(f_rel = {i:.6f})')
        plt.xlabel('I (Real)')
        plt.ylabel('Q (Imag)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    n = np.arange(len(signal_aligned))
    # print(f"Длина n: {len(n)}")
    signal_shifted =  signal_aligned * np.exp(-1j * 2 * np.pi * f_rel_method2 * n)





    # signal_aligned = signal_aligned*np.exp((-1j)*2*np.pi*0.1*np.arange(len(signal_aligned)))

    for i in range(4):
        signal_iq_offset = signal_aligned[i::4]

        freq_estimate = estimate_initial_frequency(signal_iq_offset, 1000)
        print(f"freq_estimate: {freq_estimate:.6f}")
        plt.plot(signal_iq_offset.real, signal_iq_offset.imag, 'o', markersize=2)
        plt.title(f'Signal IQ offset {i}')
        plt.xlabel('Real')
        plt.ylabel('Imag')
        plt.grid(True, alpha=0.03)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()