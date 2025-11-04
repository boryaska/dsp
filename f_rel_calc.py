import numpy as np
from diff_method import find_preamble_offset
from gardner2 import gardner_timing_recovery
from diff_method import rrc_filter
from matplotlib import pyplot as plt
def estimate_freq_offset(r, M=1, Fs=1.0):
    """Оценка относительного сдвига частоты по автокорреляции."""
    prod = r[M:] * np.conj(r[:-M])
    angle = np.angle(np.sum(prod))
    f_rel = angle * Fs / (2 * np.pi * M)
    return f_rel


signal = np.fromfile('qpsk_high_snr_sps_4_float32.pcm', dtype=np.float32)
signal_iq = signal[::2] + 1j * signal[1::2]
print(len(signal_iq))

f_est = estimate_freq_offset(signal_iq)
print(f"{f_est:.6f} - Частотный сдвиг signal_iq")

preamble_data = np.fromfile('preamb_symbols_float32.pcm', dtype=np.float32)
preamble_iq = preamble_data[::2] + 1j * preamble_data[1::2]

offset, signal_aligned, conv_results, conv_max = find_preamble_offset(signal_iq, preamble_iq, sps = 4)
# def refine_freq_offset(r, f_est0, Fs=1.0, rel_range=0.1, steps=201):
#     """
#     Улучшенная оценка CFO перебором в диапазоне ±rel_range от f_est0.
#     Возвращает уточнённую частоту и массив (f_grid, metric).
#     """
#     # Диапазон поиска
#     f_grid = np.linspace(f_est0*(1-rel_range), f_est0*(1+rel_range), steps)
#     metric = []

#     n = np.arange(len(r))
#     for f in f_grid:
#         # Компенсация предполагаемого CFO
#         r_corr = r * np.exp(-1j * 2 * np.pi * f * n / Fs)
#         # Метрика — "прямолинейность" фазы (можно взять |∑ r[n+1]r*[n]|)
#         prod = r_corr[1:] * np.conj(r_corr[:-1])
#         metric.append(np.abs(np.sum(prod)))

#     f_best = f_grid[np.argmax(metric)]
#     return f_best, (f_grid, metric)

# f_est = estimate_freq_offset(signal_aligned)
# print(f"{f_est:.6f} - Частотный сдвиг signal_aligned")    

# f_rel, (f_grid, metric) = refine_freq_offset(signal_aligned, f_est, Fs=1.0, rel_range=0.2, steps=100000)
# print(f"{f_rel:.8f} - Уточненная частота")


def refine_freq_offset_multi(r, f_est0, Fs=1.0, rel_range=0.1, steps=201, plot=True):
    """
    Уточнение частотного сдвига (CFO) по трём метрикам.
    Возвращает: (f_best_1, f_best_2, f_best_3), результаты метрик
    """
    f_grid = np.linspace(f_est0 * (1 - rel_range), f_est0 * (1 + rel_range), steps)
    n = np.arange(len(r))
    
    metrics1 = []  # автокорреляция
    metrics2 = []  # дисперсия фазы
    metrics3 = []  # энергия вектора
    
    for f in f_grid:
        r_corr = r * np.exp(-1j * 2 * np.pi * f * n / Fs)

        # Метрика 1: корреляция соседних отсчётов
        prod = r_corr[1:] * np.conj(r_corr[:-1])
        metrics1.append(np.abs(np.sum(prod)))

        # Метрика 2: отрицательная дисперсия фазы
        phase_diff = np.angle(prod)
        metrics2.append(-np.var(phase_diff))

        # Метрика 3: модуль суммы сигнала
        metrics3.append(np.abs(np.sum(r_corr)))

    # Оптимальные частоты
    f_best_1 = f_grid[np.argmax(metrics1)]
    f_best_2 = f_grid[np.argmax(metrics2)]
    f_best_3 = f_grid[np.argmax(metrics3)]

    # if plot:
    #     plt.figure(figsize=(10, 6))
    #     plt.subplot(3, 1, 1)
    #     plt.plot(f_grid, metrics1)
    #     plt.title("Метрика 1: |Σ r[n+1]·r*[n]| (автокорреляция)")
    #     plt.axvline(f_best_1, color='r', linestyle='--')
    #     plt.grid(True)

    #     plt.subplot(3, 1, 2)
    #     plt.plot(f_grid, metrics2)
    #     plt.title("Метрика 2: -Var(Δфазы)")
    #     plt.axvline(f_best_2, color='r', linestyle='--')
    #     plt.grid(True)

    #     plt.subplot(3, 1, 3)
    #     plt.plot(f_grid, metrics3)
    #     plt.title("Метрика 3: |Σ r[n]| (энергия после компенсации)")
    #     plt.axvline(f_best_3, color='r', linestyle='--')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    print(f"\n🔹 Результаты уточнения CFO:")
    print(f"  Метрика 1 (автокорреляция): f_rel = {f_best_1:.6f}")
    print(f"  Метрика 2 (дисперсия фазы): f_rel = {f_best_2:.6f}")
    print(f"  Метрика 3 (векторная сумма): f_rel = {f_best_3:.6f}")

    return (f_best_1, f_best_2, f_best_3), (f_grid, metrics1, metrics2, metrics3)

(f_best_1, f_best_2, f_best_3), (f_grid, metrics1, metrics2, metrics3) = refine_freq_offset_multi(signal_aligned, f_est, Fs=1.0, rel_range=0.2, steps=500)
print(f"{f_best_1:.6f} - Метрика 1")
print(f"{f_best_2:.6f} - Метрика 2")
print(f"{f_best_3:.6f} - Метрика 3")

import numpy as np
import matplotlib.pyplot as plt

def refine_freq_offset_preamble(r, p, f_est0, Fs=1.0, rel_range=0.1, steps=201, plot=True):
    """
    Уточнение частотного сдвига (CFO) при известной преамбуле p[n].
    Возвращает: (f_best_1, f_best_2, f_best_3), метрики для анализа.
    """
    L = len(p)
    r = r[:L]  # обрежем до длины преамбулы, если нужно
    n = np.arange(L)
    f_grid = np.linspace(f_est0 * (1 - rel_range), f_est0 * (1 + rel_range), steps)

    metrics1 = []  # |Σ r[n]·p*[n]|
    metrics2 = []  # -Var(angle(r[n]·p*[n]))
    metrics3 = []  # |Σ (r[n+1]p*[n+1])(r[n]p*[n])^*|

    for f in f_grid:
        # Компенсируем предполагаемый частотный сдвиг
        r_corr = r * np.exp(-1j * 2 * np.pi * f * n / Fs)
        rp = r_corr * np.conj(p)

        # Метрика 1
        metrics1.append(np.abs(np.sum(rp)))

        # Метрика 2
        metrics2.append(-np.var(np.angle(rp)))

        # Метрика 3
        if L > 1:
            prod = rp[1:] * np.conj(rp[:-1])
            metrics3.append(np.abs(np.sum(prod)))
        else:
            metrics3.append(0)

    # Находим лучшие оценки
    f_best_1 = f_grid[np.argmax(metrics1)]
    f_best_2 = f_grid[np.argmax(metrics2)]
    f_best_3 = f_grid[np.argmax(metrics3)]

    # Визуализация
    # if plot:
    #     plt.figure(figsize=(10, 6))
    #     plt.subplot(3, 1, 1)
    #     plt.plot(f_grid, metrics1)
    #     plt.title("Метрика 1: |Σ r[n]·p*[n]| (корреляция с преамбулой)")
    #     plt.axvline(f_best_1, color='r', linestyle='--')
    #     plt.grid(True)

    #     plt.subplot(3, 1, 2)
    #     plt.plot(f_grid, metrics2)
    #     plt.title("Метрика 2: -Var(angle(r[n]·p*[n])) (ровность фазы)")
    #     plt.axvline(f_best_2, color='r', linestyle='--')
    #     plt.grid(True)

    #     plt.subplot(3, 1, 3)
    #     plt.plot(f_grid, metrics3)
    #     plt.title("Метрика 3: |Σ (r[n+1]p*[n+1])(r[n]p*[n])^*|")
    #     plt.axvline(f_best_3, color='r', linestyle='--')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    print(f"\n🔹 Результаты уточнения CFO по преамбуле:")
    print(f"  Метрика 1 (корреляция):     f_rel = {f_best_1:.6f}")
    print(f"  Метрика 2 (дисперсия фазы): f_rel = {f_best_2:.6f}")
    print(f"  Метрика 3 (фазовая связность): f_rel = {f_best_3:.6f}")

    return (f_best_1, f_best_2, f_best_3), (f_grid, metrics1, metrics2, metrics3)


def _parabolic_peak(xm1, x0, xp1, fm1, f0, fp1):
    """
    Параболическая интерполяция аппроксимации максимума по трём точкам.
    Возвращает смещение dx относительно центра (x0) и аппроксимированное значение.
    """
    # fit quadratic: y = a*x^2 + b*x + c, with x = -1,0,1
    # then vertex at x = -b/(2a)
    a = (fm1 - 2*f0 + fp1) / 2.0
    b = (fp1 - fm1) / 2.0
    if a == 0:
        return 0.0, f0
    dx = -b / (2*a)
    fpeak = a*dx*dx + b*dx + f0
    return dx, fpeak

def _decision_metric_evmsq(r_corr, constellation=None, downsample=None):
    """
    Decision-directed metric: mean squared error to nearest constellation point.
    If constellation is None, returns None.
    If downsample is integer, assumes one symbol per 'downsample' samples (simple).
    """
    if constellation is None:
        return None
    # if downsample provided, pick every downsample-th sample (assumes already symbol aligned)
    if downsample is not None:
        samples = r_corr[::downsample]
    else:
        samples = r_corr
    # map to nearest constellation point
    pts = np.array(constellation)
    # vectorized nearest
    # compute squared distances
    dists = np.abs(samples[:, None] - pts[None, :])**2
    idx = np.argmin(dists, axis=1)
    nearest = pts[idx]
    mse = np.mean(np.abs(samples - nearest)**2)
    return mse

def refine_cfo_with_local_search(r,
                                 f_est0,
                                 Fs=1.0,
                                 rel_range=0.2,
                                 coarse_steps=401,
                                 top_k=3,
                                 local_halfwidth=0.02,
                                 local_steps=200,
                                 use_preamble=None,
                                 preamble=None,
                                 constellation=None,
                                 downsample_for_decision=None):
    """
    Иерархическое уточнение CFO:
    - r: комплексный приёмный сигнал (можно обрезать до релевантной части)
    - f_est0: начальная (грубая) оценка
    - Fs: sampling rate (единица для нормированных единиц)
    - rel_range: относительный диапазон ±(rel_range) вокруг f_est0
    - coarse_steps: число точек грубой сетки
    - top_k: сколько пиков брать для локальной доработки по каждой метрике
    - local_halfwidth: доля от f_est0 диапазона вокруг пика для локального поиска (абсолютная доля от f_est0)
    - local_steps: число точек в локальной сетке
    - use_preamble / preamble: если задано, будет использована кореляция с преамбулой (preamble должно быть длины L)
    - constellation: список комплексных точек (например QPSK), для decision-directed EVM
    - downsample_for_decision: integer, если нужно взять каждый N-й сэмпл для оценивания EVM
    Возвращает: dict с кандидатами и лучшим вариантом по EVM/метрикам, и r_corrected по лучшей частоте.
    """
    N = len(r)
    n = np.arange(N)
    # формируем грубую сетку
    f_grid = np.linspace(f_est0*(1-rel_range), f_est0*(1+rel_range), coarse_steps)

    # вычисляем три метрики для каждой частоты
    metrics_corr = np.zeros_like(f_grid, dtype=float)   # корреляция с преамбулой, если есть
    metrics_autoc = np.zeros_like(f_grid, dtype=float)  # автокорреляция соседних отсчётов
    metrics_sum = np.zeros_like(f_grid, dtype=float)   # модуль суммы

    # если преамбула задана, обрежем r до длины преамбулы
    if (use_preamble is None and preamble is not None) or use_preamble:
        if preamble is None:
            raise ValueError("preamble must be provided when use_preamble=True")
        L = len(preamble)
        r_slice = r[:L]
        n_slice = np.arange(L)
    else:
        r_slice = r
        n_slice = n

    for i, f in enumerate(f_grid):
        corr_phase = np.exp(-1j * 2*np.pi * f * n / Fs)
        r_corr_full = r * corr_phase

        # autocorr metric on full signal
        prod = r_corr_full[1:] * np.conj(r_corr_full[:-1])
        metrics_autoc[i] = np.abs(np.sum(prod))

        metrics_sum[i] = np.abs(np.sum(r_corr_full))

        # correlation with preamble (if available) on slice
        if preamble is not None:
            corr_phase_slice = np.exp(-1j * 2*np.pi * f * n_slice / Fs)
            r_corr_slice = r_slice * corr_phase_slice
            metrics_corr[i] = np.abs(np.sum(r_corr_slice * np.conj(preamble)))

    # helper to get top-k indices per metric
    def top_k_indices(metric, k):
        inds = np.argsort(metric)[-k:][::-1]
        return inds

    candidates = {}  # keep candidate frequencies and refined values
    metrics = {'corr': metrics_corr, 'autoc': metrics_autoc, 'sum': metrics_sum}
    for name, met in metrics.items():
        inds = top_k_indices(met, top_k)
        cand_list = []
        for ind in inds:
            # local grid center at f_grid[ind]
            fc = f_grid[ind]
            # local absolute half-width: choose based on local_halfwidth (as fraction of |f_est0| or absolute if f_est0 small)
            # we use absolute halfwidth in normalized units relative to Fs: local_halfwidth * max(|f_est0|, 1e-6)
            abs_half = max(local_halfwidth * max(abs(f_est0), 1e-6), local_halfwidth*1e-6)
            f_loc = np.linspace(fc - abs_half, fc + abs_half, local_steps)

            met_loc = []
            evm_loc = []
            for f2 in f_loc:
                phase = np.exp(-1j * 2*np.pi * f2 * n / Fs)
                r_corr = r * phase

                # metric value (same as coarse)
                prod = r_corr[1:] * np.conj(r_corr[:-1])
                v_autoc = np.abs(np.sum(prod))
                v_sum = np.abs(np.sum(r_corr))
                if preamble is not None:
                    L = len(preamble)
                    r_corr_slice = (r[:L] * np.exp(-1j * 2*np.pi * f2 * np.arange(L) / Fs))
                    v_corr = np.abs(np.sum(r_corr_slice * np.conj(preamble)))
                else:
                    v_corr = 0.0

                # combine metrics for local selection (we'll keep the metric relevant for 'name')
                if name == 'corr':
                    mval = v_corr
                elif name == 'autoc':
                    mval = v_autoc
                else:
                    mval = v_sum
                met_loc.append(mval)

                # decision-directed metric if constellation provided
                evm = _decision_metric_evmsq(r_corr, constellation=constellation, downsample=downsample_for_decision)
                evm_loc.append(evm if evm is not None else np.nan)

            met_loc = np.asarray(met_loc)
            evm_loc = np.asarray(evm_loc)

            # find local max index
            imax = np.nanargmax(met_loc)
            # refine by parabola using imax-1, imax, imax+1 if available
            if 0 < imax < len(met_loc)-1:
                dx, fpeak_val = _parabolic_peak(f_loc[imax-1], f_loc[imax], f_loc[imax+1],
                                                met_loc[imax-1], met_loc[imax], met_loc[imax+1])
                f_refined = f_loc[imax] + dx * (f_loc[1]-f_loc[0])
            else:
                f_refined = f_loc[imax]

            cand = {
                'metric_name': name,
                'coarse_index': ind,
                'coarse_freq': f_grid[ind],
                'local_freq_grid': f_loc,
                'local_metric': met_loc,
                'local_evm': evm_loc,
                'local_best_index': imax,
                'refined_freq': f_refined,
                'refined_metric': met_loc[imax],
                'refined_evm': evm_loc[imax] if not np.isnan(evm_loc[imax]) else None
            }
            cand_list.append(cand)
        candidates[name] = cand_list

    # теперь агрегируем кандидатов и выберем лучший окончательно:
    all_candidates = []
    for name, clist in candidates.items():
        for c in clist:
            all_candidates.append(c)
    # если есть decision-directed evm, предпочитаем минимальную MSE
    best_by_evm = None
    if constellation is not None:
        evms = []
        for c in all_candidates:
            # take refined freq and compute precise evm
            fref = c['refined_freq']
            r_corr = r * np.exp(-1j * 2*np.pi * fref * n / Fs)
            evm = _decision_metric_evmsq(r_corr, constellation=constellation, downsample=downsample_for_decision)
            c['final_evm'] = evm
            evms.append(evm)
        evms = np.array(evms)
        idx_best = np.nanargmin(evms)
        best_by_evm = all_candidates[idx_best]
    else:
        # если нет constellation, выбираем максимально согласованный метрический (суммарный норм.)
        # например, нормируем каждую метрику и берём max of normalized refined_metric
        vals = []
        for c in all_candidates:
            vals.append(c['refined_metric'])
        vals = np.array(vals)
        idx_best = np.nanargmax(vals)
        best_by_evm = all_candidates[idx_best]

    # prepare returned corrected signal for best frequency
    f_best = best_by_evm['refined_freq']
    r_corrected = r * np.exp(-1j * 2*np.pi * f_best * n / Fs)

    result = {
        'f_est0': f_est0,
        'f_grid': f_grid,
        'metrics': metrics,
        'candidates': candidates,
        'best_candidate': best_by_evm,
        'f_best': f_best,
        'r_corrected': r_corrected
    }
    return result


(f_best_preamble_1, f_best_preamble_2, f_best_preamble_3), (f_grid, metrics1, metrics2, metrics3) = refine_freq_offset_preamble(signal_aligned, preamble_iq, f_est, Fs=1.0, rel_range=0.2, steps=500)
print(f"{f_best_preamble_1:.6f} - Метрика 1 (с преамбулой)")
print(f"{f_best_preamble_2:.6f} - Метрика 2 (с преамбулой)")
print(f"{f_best_preamble_3:.6f} - Метрика 3 (с преамбулой)")

# Используем результаты предыдущих метрик для инициализации
# Берем среднее значение лучших оценок
f_est_refined = (f_best_1 + f_best_preamble_1) / 2.0
print(f"\n🔸 Усредненная оценка из предыдущих метрик: {f_est_refined:.8f}")

# Вызываем продвинутую функцию с локальным поиском
print("\n🔹 Запуск иерархического уточнения CFO...")
qpsk_constellation = [1+1j, 1-1j, -1+1j, -1-1j]  # QPSK созвездие
result = refine_cfo_with_local_search(
    r=signal_aligned,
    f_est0=f_est_refined,
    Fs=1.0,
    rel_range=0.2,
    coarse_steps=500,
    top_k=3,
    local_halfwidth=0.01,
    local_steps=101,
    use_preamble=True,
    preamble=preamble_iq,
    constellation=qpsk_constellation,
    downsample_for_decision=4  # так как sps=4
)

print(f"\n✅ РЕЗУЛЬТАТЫ ИЕРАРХИЧЕСКОГО ПОИСКА:")
print(f"   Начальная оценка f_est0: {result['f_est0']:.8f}")
print(f"   Лучшая частота f_best:   {result['f_best']:.8f}")

best_cand = result['best_candidate']
print(f"\n📊 Лучший кандидат:")
print(f"   Метрика: {best_cand['metric_name']}")
print(f"   Грубая оценка: {best_cand['coarse_freq']:.8f}")
print(f"   Уточненная оценка: {best_cand['refined_freq']:.8f}")
if best_cand['refined_evm'] is not None:
    print(f"   EVM (MSE): {best_cand['refined_evm']:.6f}")
if 'final_evm' in best_cand:
    print(f"   Финальная EVM: {best_cand['final_evm']:.6f}")

print(f"\n📈 Топ-3 кандидата по каждой метрике:")
for metric_name, cand_list in result['candidates'].items():
    print(f"\n  {metric_name.upper()}:")
    for i, cand in enumerate(cand_list):
        evm_str = f", EVM={cand['refined_evm']:.6f}" if cand['refined_evm'] is not None else ""
        print(f"    {i+1}. f={cand['refined_freq']:.8f}{evm_str}")



# if len(signal_aligned) >= len(preamble_iq):
#     corr = np.sum(signal_aligned[:len(preamble_iq)] * np.conj(preamble_iq))
#     f_rel_est2 = np.angle(corr) / (2 * np.pi * len(preamble_iq))
#     print(f"{f_rel_est2:.6f} - Оценка частоты по корреляции с преамбулой до фильтрации")
# else:
#     print("Восстановленный сигнал короче преамбулы!")

rrc = rrc_filter(sps = 4, span = 10, alpha = 0.35)
signal_filtered = np.convolve(signal_aligned, rrc, mode='same')
signal_filtered = signal_filtered / np.std(signal_filtered)

f_est = estimate_freq_offset(signal_filtered)
print(f"{f_est:.6f} - Частотный сдвиг signal_filtered")

if len(signal_filtered) >= len(preamble_iq):
    corr = np.sum(signal_filtered[:len(preamble_iq)] * np.conj(preamble_iq))
    f_rel_est2 = np.angle(corr) / (2 * np.pi * len(preamble_iq))
    print(f"{f_rel_est2:.6f} - Оценка частоты по корреляции с преамбулой после фильтрации")
else:
    print("Восстановленный сигнал короче преамбулы!")

signal_recovered, errors, mu_history = gardner_timing_recovery(signal_filtered, sps = 4, alpha = 0.05)
print(f"Финальное значение mu: {mu_history[-1]:.4f}")
print(f"Длина восстановленного сигнала: {len(signal_recovered)}")
print(f"Длина преамбулы: {len(preamble_iq)}")


f_est = estimate_freq_offset(signal_recovered)
print(f"{f_est:.8f} - Частотный сдвиг signal_recovered")




# f_rel = f_est
f_rel = 0.0164284
n = np.arange(signal_recovered.size, dtype=np.float32)
signal_recovered_corrected = signal_recovered * np.exp(-1j * 2 * np.pi * f_rel * n)

# Визуализация
plt.figure(figsize=(10, 10))
plt.plot(signal_recovered_corrected.real, signal_recovered_corrected.imag, 'o', markersize=3, alpha=0.6)
plt.title(f'Созвездие после коррекции частоты\n(f_rel = {f_rel:.6f})', fontsize=14)
plt.xlabel('I (Real)')
plt.ylabel('Q (Imag)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()


