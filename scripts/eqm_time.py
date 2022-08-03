# Find equilibrium times from network solver output
from typing import Dict, List

import numpy as np
from abundances import mm00_abundances
from gcrn.network import Network
from dynamics_test import calculate_number_densities
from gcrn.timescales import find_equilibrium
import matplotlib.pyplot as plt


def setup_network(network_file: str, density: float, temperature: float,
                  abundances: Dict):
  network = Network.from_krome_file(network_file)
  n = calculate_number_densities(abundances, np.log10(density))
  network.temperature = temperature
  network.number_densities = n

  return network


def plot_number_densities(times: np.ndarray, n: np.ndarray, network: Network,
                          eqm_times: List,
                          plot_species=['C', 'O', 'C2', 'CN', 'OH', 'CO']):
  # 'n' should be a 2D array of (time, species)
  n = np.log10(n).T
  print(n.shape)
  fig, ax = plt.subplots()
  for i, s in enumerate(network.species):
    if s in plot_species:
      p = ax.plot(np.log10(times), n[i], label=s)
      c = p[0].get_color()
      ax.axvline(eqm_times[i], ls='--', c=c)

  ax.legend()
  ax.set_xlabel("log time [s]")
  ax.set_ylabel(r"log n [cm$^{-3}$]")
  return fig, ax


def main():
  density = 1e-8
  temperature = 6000.
  network_file = "../res/cno.ntw"
  network = setup_network(network_file, density, temperature, mm00_abundances)

  # Solve to estimate steady-state
  times = np.logspace(-8, 6, num=1000)
  n = network.solve(times, n_subtime=1, return_eqm_times=False)
  eqm_times, eqm_n = find_equilibrium(times, n, threshold=1e-6)

  fig, ax = plot_number_densities(times, n, network, eqm_times)
  for i, s in enumerate(network.species):
    print(f"{s}: {eqm_times[i]:.2e}")

  plt.show()


if __name__ == "__main__":
  main()
