# Compare Julia to Python GCRN
from typing import Dict
import numpy as np
from gcrn.helper_functions import setup_number_densities
from gcrn.network import Network
import matplotlib.pyplot as plt
import time
import pandas as pd


mass_hydrogen = 1.67262171e-24  # [g]

prop_cycle = plt.rcParams['axes.prop_cycle']
colours = prop_cycle.by_key()['color']


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
      # "C": 8.39,  # solar
      # "O": 8.66,  # solar
      # "N": 7.83,  # solar
      "C": 6.41,  # mm20a04
      "O": 7.06,  # mm20a04
      "N": 5.80,  # mm20a04
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
  # densities = np.logspace(-12, -6., num=100)
  # temperatures = np.linspace(1000., 15000., num=100)
  # Load from file
  # arr = np.loadtxt('./res/rho_T_test.csv', delimiter=',')
  # densities, temperatures = arr[:, 0], arr[:, 1]
  # number_densities = np.zeros((len(densities), len(network.species)))
  # times = np.linspace(1e-8, 1e6, num=1000)
  # # start_time = time.time()
  # for i, (density, temperature) in enumerate(zip(densities, temperatures)):
  #   network.number_densities = calculate_number_densities(abundances,
  #                                                         np.log10(density))
  #   if i == 0:
  #     print(network.number_densities_dict)
  #     print(density)
  #   network.temperature = temperature
  #   number_densities[i] = network.solve(times, n_subtime=1)[-1]
  # number_densities[i] = network.number_densities  # don't solve!

  # end_time = time.time()

  # print(f"Total time: {(end_time - start_time):.2f} [s]")

  # print(number_densities.shape)
  # np.savetxt('./out/gcrn_test.csv', number_densities, delimiter=',',
  #            header=','.join(network.species), comments='')

  # Plot differences between Python and Julia
  df_python = pd.read_csv('./out/gcrn_test.csv', delimiter=',')
  df_julia = pd.read_csv('./out/catalyst_test.csv', delimiter=',')

  # fig, axes = plt.subplots(3, 3)
  # for i, key in enumerate(df_python.keys()):
  #   i_x, i_y = i // 3, i % 3
  #   # axes[i_x, i_y].plot(np.log10(df_python[key]), ls='none', marker='o')
  #   # axes[i_x, i_y].plot(np.log10(df_julia[key]), ls='-', label=key)
  #   difference = np.log10(df_python[key]) - np.log10(df_julia[key])
  #   # mean_difference = np.mean(np.abs(difference))
  #   mean_difference = np.median(np.abs(difference))
  #   axes[i_x, i_y].plot(difference,  # quick calc shows 1e-10 numerics
  #                       label=f"{key}\nMedian Diff = {mean_difference:1.3e}")
  #   axes[i_x, i_y].axhline(c='k', ls=':')
  #   # axes[i_x, i_y].set_ylim(-1e-5, 1e-5)

  #   axes[i_x, i_y].legend()

  # plt.show()

  # Check reshaping
  nz, ny, nx = 10, 5, 7
  n_python = {k: np.reshape(v.values, (nz, ny, nx))
              for k, v in df_python.items()}
  n_julia = {k: np.reshape(v.values, (nz, ny, nx))
             for k, v in df_julia.items()}
  
  fig, axes = plt.subplots(3, 3)
  for i, key in enumerate(n_python.keys()):
    i_x, i_y = i // 3, i % 3
    # log_n_python = np.log10(np.mean(n_python[key], axis=(1, 2)))
    # log_n_julia = np.log10(np.mean(n_julia[key], axis=(1, 2)))
    log_n_python = np.log10(n_python[key].flatten())
    log_n_julia = np.log10(n_julia[key].flatten())

    # axes[i_x, i_y].plot(log_n_python, c=colours[i], marker='o', ls='none')
    # axes[i_x, i_y].plot(log_n_julia, c=colours[i], ls='-', marker='o', mfc='none',
    #                     label=key)
    axes[i_x, i_y].plot(log_n_python - log_n_julia, c=colours[i], label=key)

    axes[i_x, i_y].legend()
    axes[i_x, i_y].set_ylabel("log n")
  plt.show()

  # Compare GCRN and Catalyst outputs
  # fig, axes = plt.subplots()
  # # network.number_densities = {'A': 1., 'B': 2., 'C': 3.}
  # network.number_densities = calculate_number_densities(abundances,
  #                                                       np.log10(1e-12))
  # network.temperature = 3000.
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
