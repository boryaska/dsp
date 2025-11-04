import numpy as np
import matplotlib.pyplot as plt
from read_file import rrc_filter


Nsym = 2000         # число символов
sps = 2            # samples per symbol
T = 1              # символный период (условно)
mu = 0.0           # fractional timing offset (начальный)
alpha = 0.05       # шаг корректировки тайминга (петли)
np.random.seed(1)


data = np.random.randint(0, 4, Nsym)
symbols = np.exp(1j * 2 * np.pi * data / 4)
# plt.plot(symbols.real, symbols.imag, 'o')



samples = np.zeros(Nsym*sps, dtype=complex)
for i in range(Nsym):
    samples[i*sps] = symbols[i]

# plt.plot(np.abs(samples), 'o')    

rrc = rrc_filter(sps, 10, 0.35)
samples = np.convolve(samples, rrc, mode='same')
# plt.plot(samples.real, samples.imag, 'o')

    


# --- Вносим временной сдвиг (искусственно) ---
delay = 0.65  
t = np.arange(len(samples))
shifted = np.interp(t, t - delay, samples)



plt.figure(figsize=(8, 8))
plt.plot(shifted.real, shifted.imag, 'o', alpha=0.3, markersize=4)
plt.title('Сигнал со сдвигом')
plt.grid(True)

# --- Gardner timing recovery ---
timing_errors = []
recovered = []
mu_history = []

mu = 0.0  # дробная часть между отсчетами (0 <= mu < sps)
k = sps * 2  # начинаем с 2 символов запаса

# Функция интерполации
def interpolate(signal, idx):
    """Линейная интерполация для дробного индекса"""
    idx_int = int(idx)
    frac = idx - idx_int
    if idx_int + 1 < len(signal):
        return signal[idx_int] * (1 - frac) + signal[idx_int + 1] * frac
    else:
        return signal[idx_int]

while k < len(shifted) - sps:
    # Берем три отсчета с учетом дробной задержки mu
    idx_early = k - sps + mu
    idx_mid = k - sps//2 + mu
    idx_late = k + mu
    
    if idx_late + 1 >= len(shifted):
        break
    
    x_early = interpolate(shifted, idx_early)
    x_mid = interpolate(shifted, idx_mid)
    x_late = interpolate(shifted, idx_late)
    
    # Формула Гарднера: error = real(mid * conj(late - early))
    err = np.real(x_mid * np.conj(x_late - x_early))
    timing_errors.append(err)
    
    # Обновляем mu (дробную задержку)
    mu = mu - alpha * err
    mu_history.append(mu)
    
    # Берем отсчет на текущей позиции с коррекцией
    recovered.append(interpolate(shifted, k + mu))
    
    # Переходим к следующему символу
    k += sps

# --- Визуализация ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

recovered_arr = np.array(recovered)

# График исходных символов
ax = axes[0, 0]
ax.plot(symbols.real, symbols.imag, 'rx', markersize=8, alpha=0.7)
ax.set_title("Исходные символы", fontsize=12, fontweight='bold')
ax.set_xlabel("I (Real)")
ax.set_ylabel("Q (Imag)")
ax.grid(True, alpha=0.3)
ax.axis('equal')

# График восстановленных символов
ax = axes[0, 1]
ax.plot(recovered_arr.real, recovered_arr.imag, 'go', markersize=8, alpha=0.7)
ax.set_title("Восстановленные символы", fontsize=12, fontweight='bold')
ax.set_xlabel("I (Real)")
ax.set_ylabel("Q (Imag)")
ax.grid(True, alpha=0.3)
ax.axis('equal')

# График ошибок синхронизации
ax = axes[0, 2]
ax.plot(timing_errors)
ax.set_title("Ошибка синхронизации (Gardner TED)")
ax.set_xlabel("Символ")
ax.set_ylabel("Ошибка")
ax.grid(True, alpha=0.3)
ax.axhline(0, color='r', linestyle='--', alpha=0.5)

# График истории mu
ax = axes[1, 0]
ax.plot(mu_history)
ax.set_title("История mu (дробная задержка)")
ax.set_xlabel("Символ")
ax.set_ylabel("mu (отсчеты)")
ax.grid(True, alpha=0.3)
ax.axhline(0, color='r', linestyle='--', alpha=0.5)

# График фазовых ошибок символов
ax = axes[1, 1]
if len(recovered_arr) <= len(symbols):
    phase_errors = np.angle(recovered_arr) - np.angle(symbols[:len(recovered_arr)])
else:
    phase_errors = np.angle(recovered_arr[:len(symbols)]) - np.angle(symbols)
phase_errors = np.angle(np.exp(1j * phase_errors))  # wrap to [-pi, pi]
ax.plot(np.abs(phase_errors) * 180/np.pi)
ax.set_title("Фазовая ошибка символов")
ax.set_xlabel("Символ")
ax.set_ylabel("Ошибка (градусы)")
ax.grid(True, alpha=0.3)

# График сравнения (наложение)
ax = axes[1, 2]
ax.plot(symbols.real, symbols.imag, 'rx', markersize=10, alpha=0.4, label='Исходные')
ax.plot(recovered_arr.real, recovered_arr.imag, 'go', markersize=6, alpha=0.7, label='Восстановленные')
ax.set_title("Сравнение (наложение)")
ax.set_xlabel("I (Real)")
ax.set_ylabel("Q (Imag)")
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend()

plt.tight_layout()
plt.show()

print(f"Количество восстановленных символов: {len(recovered)}")
print(f"Количество исходных символов: {len(symbols)}")
print(f"Средняя ошибка тайминга: {np.mean(np.abs(timing_errors)):.4f}")
print(f"Финальное значение mu: {mu_history[-1]:.4f}")