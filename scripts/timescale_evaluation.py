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
  network = Network.from_krome_file(f"{RES_DIR}/solar_co_w05.ntw",
                                    initialise_jacobian=True)
  temperature = 5700.
  rho = 1e-8
  # abundances = mm00_abundances
  abundances = mm20a04_abundances
  n = calculate_number_densities(abundances, np.log10(rho))
  network.number_densities = n
  network.temperature = temperature
  # Step through network, keeping track of various timescales
  initial_time = 1e-7
  times = np.logspace(-6, 3, num=20)
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
  species = ["C", "O", "CH", "CO", "OH"]
  species.sort()
  print(network.species)
  for i, s in enumerate(species):
    # evolution
    axes[0].plot(np.log10(times), np.log10(number_densities[:, i]),
                 marker='o', label=s)
    # timescales
    axes[1].plot(np.log10(times), np.log10(timescales[:, i]), marker='o')

  plt.show()


if __name__ == "__main__":
  main()
