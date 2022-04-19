# Compare temperature-dependent ring reaction network with Julia implementation
from typing import Dict
import numpy as np
from gcrn.helper_functions import setup_number_densities
from gcrn.network import Network
import matplotlib.pyplot as plt
import time


mass_hydrogen = 1.67262171e-24  # [g]


def densities_from_abundances(abundances, gas_density, return_array=False):
  # Convert abundances to number densities by means of the hydrogen number
  # density from the gas density
  # Epsilon notation where H normalised to 1
  abu_eps_lin = {
      key: 10**(val - 12) for (key, val) in abundances.items()
  }

  abundances_linear = {
      key: 10**val for (key, val) in abundances.items()
  }

  total_abundances = sum(abundances_linear.values())

  abundances_ratio = {
      key: val / total_abundances for (key, val) in abundances_linear.items()
  }

  # TODO:
  # Include all hydrogenic species! Doesn't matter here bc at start they're
  # negligible but should add for completion
  h_number_density = gas_density / mass_hydrogen * abundances_ratio['H']

  densities = {
      key: abu * h_number_density for (key, abu) in abu_eps_lin.items()
  }

  return np.array(densities.values()) if return_array else densities


def calculate_number_densities(abundances: Dict, log_gas_density: float):
  # Compute densities from abundances
  gas_density = 10**log_gas_density
  densities = densities_from_abundances(abundances, gas_density)

  return densities


def main():
  abundances = {
      "H": 12,
      "H2": -4,
      "OH": -12,
      "C": 8.39,  # solar
      "O": 8.66,  # solar
      "N": 7.83,
      "CH": -12,
      "CO": -12,
      "CN": -12,
      "NH": -12,
      "NO": -12,
      "C2": -12,
      "O2": -12,
      "N2": -12,
      "M": 11,
  }

  network = Network.from_krome_file('../res/solar_co_w05.ntw')
  # network = Network.from_krome_file('../res/mass_action.ntw')
  densities = np.logspace(-12, -6., num=100)
  temperatures = np.linspace(1000., 15000., num=100)
  times = np.linspace(1e-8, 1e6, num=1000)
  start_time = time.time()
  for i, density in enumerate(densities):
    for j, temperature in enumerate(temperatures):
      network.number_densities = calculate_number_densities(abundances,
                                                            np.log10(density))
      network.temperature = temperature
      n = network.solve(times, n_subtime=1)

  end_time = time.time()

  print(f"Total time: {(end_time - start_time):.2f} [s]")

  # fig, axes = plt.subplots()
  # network.number_densities = {'A': 1., 'B': 2., 'C': 3.}
  # network.number_densities = calculate_number_densities(abundances,
  #                                                       np.log10(1e-8))
  # network.temperature = 5000.
  # print(network.number_densities)
  # n = network.solve(times, n_subtime=1)
  # print(network.species)
  # print(n[-1])
  # for i, s in enumerate(network.species):
  #   axes.plot(times, n.T[i], label=s)

  # axes.legend()
  # plt.loglog()
  # plt.show()


if __name__ == "__main__":
  main()
