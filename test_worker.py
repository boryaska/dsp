import worker
import analys

gen = worker.Generation(
    constellation='qpsk',
     preamble_length=300,
      signal_length=2000,
       sps=4,
        Frel=0.000,
         Noise_power=0.00,
         N = 5,
         Delay = 30)

preamble, samples, symbols = gen.generate()

print(f'len(samples) {len(samples)}')
analys.plot_signal(samples)
analys.plot_constellation(samples, 4)
# print(preamble)
# print(symbols)


process = worker.Processing(
    signal=samples,
     preamble=preamble,
      packet_symb=symbols,
      sps=4,
      constellation='qpsk',
      interp=True)

peaks, finded_siganls = process.process()
print(peaks)
# print(finded_siganls)
# for signal in finded_siganls:
#     analys.plot_constellation(signal, 1)

