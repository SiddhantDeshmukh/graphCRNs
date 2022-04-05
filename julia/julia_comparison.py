# Compare temperature-dependent ring reaction network with Julia implementation
import numpy as np
from gcrn.network import Network
import matplotlib.pyplot as plt


network = Network.from_krome_file('../res/ring_temperature.ntw')
network.number_densities = np.array([1., 1., 1.])
network.temperature = 300.

times = np.linspace(1e-8, 1., num=50)
n = network.solve(times)

fig, axes = plt.subplots()
for i, s in enumerate(network.species):
  axes.plot(times, n.T[i], label=s)

axes.legend()
plt.savefig('ring_python.png')
