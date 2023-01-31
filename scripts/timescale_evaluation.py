# Figure out the best chemical timescales to use
# Compare simple inverse reaction rates and the Jacobian eigenvalues
from gcrn.network import Network
from gcrn.helper_functions import jacobian_timescales
from sadchem.postprocessing import calculate_number_densities
from abundances import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def to_arr(*args):
  return [np.array(arg) for arg in args]


def main():
  PROJECT_DIR = "/home/sdeshmukh/Documents/graphCRNs"
  RES_DIR = f"{PROJECT_DIR}/res"
  # network_id = "solar_co_w05.ntw"
  network_id = "cno.ntw"
  network = Network.from_krome_file(f"{RES_DIR}/{network_id}",
                                    initialise_jacobian=True)
  # temperature = 5700.
  # rho = 1e-8
  temperature = 3500.
  rho = 1e-11
  # abundances = mm00_abundances
  abundances = mm30a04_abundances
  n = calculate_number_densities(abundances, np.log10(rho))
  network.number_densities = n
  network.temperature = temperature
  # Step through network, keeping track of various timescales
  initial_time = 1e-7
  times = np.logspace(-6, 8, num=20)
  timescales = [jacobian_timescales(network.evaluate_jacobian(temperature,
                                                              network.number_densities))]
  number_densities = [network.number_densities]
  prev_time = 0.
  for i, time in enumerate(times):
    # solve, then evaluate timescale
    network.solve([time], initial_time=prev_time, n_subtime=1)
    number_densities.append(deepcopy(network.number_densities))
    timescales.append(jacobian_timescales(network.evaluate_jacobian(temperature,
                                                                    network.number_densities)))

    prev_time = time
  times = np.array([initial_time, *times])
  number_densities, timescales = to_arr(number_densities, timescales)
  fig, axes = plt.subplots(2, 1)
  species = ["CH", "CO", "OH", "CN", "C2"]
  # species = network.species
  # print(network.species)
  for i, s in enumerate(network.species):
    # Only plot relevant species, but enumerate over all bc of indices
    if s in species:
      # evolution
      axes[0].plot(np.log10(times), np.log10(number_densities[:, i]),
                   marker='o', label=s)
      # timescales
      axes[1].plot(np.log10(times), np.log10(timescales[:, i]), marker='o')

  # Aesthetics
  for ax in axes:
    ax.set_xlabel("Time [s]")
  axes[0].set_ylabel(r"log $n$ [cm$^{-3}$]")
  axes[1].set_ylabel(r"log $\tau$ [s]")
  axes[0].legend()
  fig.suptitle(rf"T = {temperature:.0f} [K], $\rho$ = {rho:1.1e}")
  plt.show()


if __name__ == "__main__":
  main()
