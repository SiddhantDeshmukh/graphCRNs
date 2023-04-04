import numpy as np
import matplotlib.pyplot as plt
from gcrn.network import Network
from gcrn.helper_functions import number_densities_from_abundances
from abundances import mm00_abundances
from copy import deepcopy


def main():
  # Setup networks, compute number densities, evolve for 1e4 seconds and check
  # outputs
  temperatures = np.linspace(4000., 10000., num=50)
  densities = np.logspace(-10., -6., num=50)

  network_dir = "../res"
  # networks = ["solar_co_w05.ntw", "solar_co_w05_4097.ntw"]
  networks = ["cno_extra.ntw", "cno_fix.ntw"]
  output_number_densities = {}
  for network_name in networks:
    network_id = network_name.replace(".ntw", "")
    network = Network.from_krome_file(f"{network_dir}/{network_name}")
    output_number_densities[network_id] = np.zeros(shape=(len(temperatures),
                                                          len(network.species)))
    for j, (T, rho) in enumerate(zip(temperatures, densities)):
      network.number_densities = number_densities_from_abundances(mm00_abundances,
                                                                  rho,
                                                                  network.species)
      network.temperature = T
      output_number_densities[network_id][j] = deepcopy(network.solve([1e4])[0])

  # Plot trends against density and temperature
  # Network 1 is points, Network 2 is lines
  network_keys = list(output_number_densities.keys())
  plt.style.use("standard-scientific")
  fig, axes = plt.subplots(1, 2)
  for i, s in enumerate(network.species):
    # Ax 1: Temperature
    l = axes[0].plot(temperatures, output_number_densities[network_keys[0]][:, i],
                     ls="none", marker="o")
    axes[0].plot(temperatures, output_number_densities[network_keys[1]][:, i],
                 ls="-", marker=None, c=l[0].get_color(), label=s)

    # Ax 2: Density
    l = axes[1].plot(np.log10(densities), output_number_densities[network_keys[0]][:, i],
                     ls="none", marker="o", mfc="none")
    axes[1].plot(np.log10(densities), output_number_densities[network_keys[1]][:, i],
                 ls="-", marker=None, c=l[0].get_color(), label=s)

  # Aesthetics
  axes[0].set_xlabel("Temperature [K]")
  axes[1].set_xlabel(r"log $\rho$ (cgs)")

  axes[0].set_ylabel("Number density (cgs)")
  for ax in axes:
    ax.legend()
    ax.set_yscale("log")

  plt.show()


if __name__ == "__main__":
  main()
