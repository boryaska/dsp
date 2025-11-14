import worker
import analys

gen = worker.Generation(
    constellation='16apsk',
     preamble_length=100,
      signal_length=1000,
       sps=4,
        Frel=0.00,
         Noise_power=0.01,
         N = 3,
         Delay = 10)

preamble, samples, symbols = gen.generate()

analys.plot_signal(samples)
analys.plot_constellation(samples, 4)
print(preamble)
print(symbols)