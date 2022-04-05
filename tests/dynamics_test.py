from itertools import product
from typing import Dict
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from gcrn.helper_functions import enumerated_product
from gcrn.network import Network
import numpy as np
import time
import matplotlib.pyplot as plt


mass_hydrogen = 1.67262171e-24  # [g]

plt.style.use('standard-scientific')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def ratio(arr: np.ndarray, axis=0):
  return np.exp(-np.diff(np.log10(arr), axis=axis))


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


# Dummy network dynamics to profile efficiency
res_dir = '../res'
network_file = f'{res_dir}/solar_co_w05.ntw'
# network_file = f'{res_dir}/cno.ntw'

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

network = Network.from_krome_file(network_file)

temperatures = np.linspace(3000, 25000, num=100)
gas_densities = np.logspace(-12, -6, num=100)
times = np.logspace(-8, 8, num=50)
start_time = time.time()
for (i, j), (T, rho) in enumerated_product(temperatures, gas_densities):
  n = calculate_number_densities(abundances, np.log10(rho))
  network.temperature = T
  network.number_densities = n
  n = network.solve(times)
  print(f"Done {i+1} / {len(gas_densities)} densities "
        f"{j+1} / {len(temperatures)} temperatures",
        end="\r")

end_time = time.time()
print(
    f"\n{len(times)} timescales,\n"
    f"{len(temperatures)} temperatures,\n"
    f"{len(gas_densities)} densities.\n"
    f"Total time: {(end_time - start_time):.3f} seconds."
)

# # Test case for plotting
# times = np.logspace(-8, 8, num=10)
# evaluation_temperature = 7000
# evaluation_density = 1e-8
# atol, rtol = 1e-30, 1e-6
# fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
# n = calculate_number_densities(abundances, np.log10(evaluation_density))
# network.temperature = evaluation_temperature
# network.number_densities = n
# n, eqm_times = network.solve(times, atol=atol, rtol=rtol,
#                              eqm_tolerance=1e-10, n_subtime=10,
#                              return_eqm_times=True)

# n = n.T

# print(network.species)
# print(eqm_times)

# for j, s in enumerate(network.species):
#   axes[0].plot(np.log10(times), np.log10(n[j]))
#   axes[1].plot(np.log10(times[:-1]), np.log10(np.abs(np.diff(n[j]))), label=s)
#   axes[2].plot(np.log10(times[:-1]), np.log10(1. - ratio(n[j], axis=0)))

#   axes[0].axvline(np.log10(eqm_times[j]), c=colors[j], ls='--')

# axes[1].axhline(np.log10(atol), c='k', ls=':')
# axes[2].axhline(np.log10(rtol), c='k', ls=':')

# for ax in axes:
#   ax.plot(np.log10(times), [0.] * len(times), ls='none', marker='o', c='k')

# axes[1].legend()
# axes[2].set_xlabel("log time [s]")
# axes[0].set_ylabel(r"log n [cm$^{-3}$]")
# axes[1].set_ylabel(r"diff log n [cm$^{-3}$]")
# axes[2].set_ylabel(r"ratio log n [cm$^{-3}$]")

# plt.show()
