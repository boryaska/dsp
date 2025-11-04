import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------- Параметры (настраивайте) ----------
pcm_path = "iq_int16.pcm"   # путь к вашему .pcm
f_rel = 0.00026      # относительная частота (cycles per sample), т.е. f_c / f_s
sps = 2            # samples per symbol (отсчётов на символ)
delay_int = 0      # целая задержка в отсчётах
delay_frac = 0.06   # дробная задержка (-0.5..+0.5)
apply_filter = True
max_points_plot = 20000
# ------------------------------------------------

def read_iq_int16(path):
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    # нормализация (опционально)
    scale = max(np.max(np.abs(i)), np.max(np.abs(q)), 1.0)
    return (i + 1j * q) / scale

def freq_correction(x, f_rel):
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * f_rel * n)

def fractional_delay_linear(x, frac):
    # Простая линейная дробная задержка: y[n] = (1-alpha)*x[n] + alpha*x[n-1]
    # frac in (-1,1) positive means delay forward (shift to right)
    alpha = frac
    if abs(alpha) < 1e-12:
        return x.copy()
    y = np.empty_like(x)
    # shift by floor part handled outside; здесь только дробная часть
    y[1:] = (1-alpha)*x[1:] + alpha*x[:-1]
    y[0] = x[0]  # грубая обработка граничного
    return y

def lowpass_fir(x, cutoff=0.2, numtaps=101):
    # normalized cutoff 0..1 (1 = Nyquist)
    b = signal.firwin(numtaps, cutoff)
    y = signal.lfilter(b, 1.0, x)
    # компенсируем group delay (примерно (numtaps-1)/2)
    gd = (numtaps - 1)//2
    return y[gd:]

def downsample_and_take_symbols(x, sps, delay_int=0):
    # выбираем отсчёт каждые sps, с учётом целой задержки
    start = delay_int
    symbols = x[start::sps]
    return symbols

def qpsk_decision(sym):
    # простейшее решение QPSK (без сдвига фазы)
    # возвращаем комплексные идеализированные символы и биты
    decisions = (np.sign(np.real(sym)) + 1j * np.sign(np.imag(sym)))
    # нормализуем на +-1 -> QPSK constellation points: (±1 ± j)
    return decisions

# ---------- pipeline ----------
x = read_iq_int16(pcm_path)
print("samples:", x.size)

# целая задержка
if delay_int != 0:
    if delay_int > 0:
        x = x[delay_int:]
    else:
        x = np.pad(x, (abs(delay_int),0), 'constant')

# дробная задержка
if abs(delay_frac) > 1e-12:
    x = fractional_delay_linear(x, delay_frac)

# частотная коррекция
x_corr = freq_correction(x, f_rel)

# optional filtering (lowpass to avoid aliasing before downsample)
if apply_filter:
    # cutoff выбираем как 0.5/sps (нормализовано по Nyquist=0.5*Fs)
    cutoff = 0.5 / sps
    # scipy firwin cutoff normalized to Nyquist=0.5 -> pass cutoff*2
    x_filtered = lowpass_fir(x_corr, cutoff=cutoff*2, numtaps=129)
else:
    x_filtered = x_corr

# downsample -> символы
symbols = downsample_and_take_symbols(x_filtered, sps, delay_int=0)

# take subset for plotting
Nplot = min(len(symbols), max_points_plot)
sym_plot = symbols[:Nplot]

# decision (QPSK example)
decisions = qpsk_decision(sym_plot)

# ---------- plots ----------
plt.figure(figsize=(6,6))
plt.scatter(np.real(sym_plot), np.imag(sym_plot), s=6, label='received')
plt.scatter(np.real(decisions), np.imag(decisions), s=10, marker='x', label='decisions')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.title("Constellation (QPSK) after freq correction")
plt.xlabel("I")
plt.ylabel("Q")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()