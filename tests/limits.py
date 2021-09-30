# %%
# Test temperature limits and fading functions
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density, log_abundance_to_number_density
from scipy.special import expit
from typing import Callable, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from src.network import Network
from src.dynamics import NetworkDynamics

alpha = 2
beta = 0.5
gamma = 3


def test_rate_1(temperature: float) -> float:
  # simple linear
  return alpha * temperature + 3


def test_rate_2(temperature: float) -> float:
  # simplified alpha, beta formulation
  return alpha * (temperature/300)**beta


def test_rate_3(temperature: float) -> float:
  # simplified alpha, gamma formulation
  return alpha * np.exp(-gamma / temperature)


def test_rate_4(temperature: float) -> float:
  # simplified alpha, beta, gamma formulation
  return alpha * (temperature / 300)**beta * np.exp(-gamma / temperature)


def lower_limit_sigmoid(temperature: float, Tmin: float, scale=1) -> float:
  return 1 / (1 + np.exp(-scale * (temperature - Tmin)))


def upper_limit_sigmoid(temperature: float, Tmax: float, scale=1) -> float:
  return 1 / (1 + np.exp(scale * (temperature - Tmax)))


def fading_function(rate: Callable, temperature: float, Tmin: float, Tmax: float,
                    lower_scale=1, upper_scale=1) -> float:
  return rate(temperature) *\
      (lower_limit_sigmoid(temperature, Tmin, lower_scale) +
       upper_limit_sigmoid(temperature, Tmax, upper_scale) - 1)


def cutoff_function(rate: Callable, temperature: float,
                    Tmin: float, Tmax: float) -> float:
  # Define the rate only between specified limits, zero otherwise
  if isinstance(temperature, np.ndarray):
    mask = (temperature >= Tmin) & (temperature <= Tmax)
    out = np.zeros(len(temperature))
    out[mask] = rate(temperature[mask])
  else:
    out = rate(temperature) if Tmin <= temperature <= Tmax else 0

  return out


def compare_limits(rate: Callable, temperatures: np.ndarray,
                   Tmin: float, Tmax: float) -> Dict:
  # Calculate original rates, cutoff rates and limited rates for a few choices
  # of 'lower_scale' & 'upper_scale'
  rates = {
      'original': rate(temperatures),
      'cutoff': cutoff_function(rate, temperatures, Tmin, Tmax),
      'fading_weak': fading_function(rate, temperatures, Tmin, Tmax, 1, 1),
      'fading_medium': fading_function(rate, temperatures, Tmin, Tmax, 50, 50),
      'fading_strong': fading_function(rate, temperatures, Tmin, Tmax, 100, 100)
  }

  return rates


def setup_number_densities(number_densities_dict, hydrogen_density):
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(number_densities_dict[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n

  return number_densities

# %%
# # Dummy rates for testing
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# # First plot fading functions applied to a few dummy reactions
# temperatures = np.logspace(1.3, 4.7, num=500)
# Tmin = 100
# Tmax = 23000
# test_rates = [test_rate_1, test_rate_2, test_rate_3, test_rate_4]

# for i, rate_function in enumerate(test_rates):
#   idx_x = i // 2
#   idx_y = i % 2
#   limited_rates = compare_limits(rate_function, temperatures, Tmin, Tmax)

#   for key, rate in limited_rates.items():
#     axes[idx_x, idx_y].plot(temperatures, rate, label=key)

#   # Limits
#   axes[idx_x, idx_y].axvline(Tmin, c='k', ls='--')
#   axes[idx_x, idx_y].axvline(Tmax, c='k', ls='--')

#   axes[idx_x, idx_y].set_xlabel("Temperature [K]")
#   axes[idx_x, idx_y].set_ylabel("Rate")
#   axes[idx_x, idx_y].set_xscale("log")
#   axes[idx_x, idx_y].set_yscale("log")
#   axes[idx_x, idx_y].legend()
#   axes[idx_x, idx_y].set_ylim(1e-3, np.max(limited_rates["original"]) + 100)

# plt.show()


# %%
# # CO network rates
res_dir = '../res'
network_file = f'{res_dir}/solar_co_w05.ntw'
# network_file = f'{res_dir}/catalyst_co.ntw'
# network_file = f'{res_dir}/mp_cno.ntw'

initial_number_densities = {
    "H": 1e12,
    "H2": 1e-4,
    "OH": 1e-12,
    "C": 10**(8.39),  # solar
    "O": 10**(8.66),  # solar
    "N": 10**(7.83),
    # "C": 10**(8.66),  # C-rich
    # "O": 10**(8.39),  # C-rich
    "CH": 1e-12,
    "CO": 1e-12,
    "CN": 1e-12,
    "NH": 1e-12,
    "NO": 1e-12,
    "C2": 1e-12,
    "O2": 1e-12,
    "M": 1e11,
}

gas_density = 1e-6  # [g cm^-3]
hydrogen_density = gas_density_to_hydrogen_number_density(gas_density)

network = Network.from_krome_file(network_file)

network.number_densities = setup_number_densities(initial_number_densities,
                                                  hydrogen_density)
print(len(network.reactions))  # 30 reactions
nrows = 6
ncols = 6

# fig, axes = plt.subplots(nrows, ncols, figsize=(32, 24), sharex=True)
# temperatures = np.logspace(1.3, 4.7, num=500)
# for i, reaction in enumerate(network.reactions):
#   idx_x = i // 6
#   idx_y = i % 6

#   limited_rates = [reaction(temperature, True) for temperature in temperatures]
#   unlimited_rates = [reaction(temperature, False)
#                      for temperature in temperatures]

#   # Change limit to 'boundary' to compute boundary rate
#   reaction.limit = 'boundary'
#   boundary_rates = [reaction(temperature, True) for temperature in temperatures]

#   # axes[idx_x, idx_y].plot(temperatures, limited_rates,
#   #                         label=f"{reaction.idx}: Limited")
#   axes[idx_x, idx_y].plot(temperatures, unlimited_rates,
#                           label=f"{reaction.idx}: Unlimited", ls='--')
#   # axes[idx_x, idx_y].plot(temperatures, boundary_rates,
#   #                         label=f"{reaction.idx}: Boundary", ls=':')

#   # axes[idx_x, idx_y].set_ylim(1e-20, 1e0)

#   axes[idx_x, idx_y].axvline(reaction.min_temperature, c='k', ls='--')
#   axes[idx_x, idx_y].axvline(reaction.max_temperature, c='k', ls='--')

#   axes[idx_x, idx_y].set_title(str(reaction))
#   if idx_x == 4:
#     axes[idx_x, idx_y].set_xlabel("Temperature [K]")
#   if idx_y == 0:
#     axes[idx_x, idx_y].set_ylabel("Rate")
#   # axes[idx_x, idx_y].set_xscale("log")
#   axes[idx_x, idx_y].set_yscale("log")
#   axes[idx_x, idx_y].legend()

# plt.show()
# exit()


# %%
def setup_dynamics(network: Network, temperature: float):
  # Initialise number densities
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(initial_number_densities[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n
  # Initialise dynamics at given temperature
  dynamics = NetworkDynamics(
      network, number_densities, temperature=temperature)

  return dynamics


def solve_dynamics(dynamics: NetworkDynamics, times: List,
                   limit_rates=False) -> np.ndarray:
  final_number_densities = []
  # TODO:
  # Refactor 'dynamics' to use dynamics.number_densities instead of passing in
  for i, time in enumerate(times):
    final = dynamics.solve(time, dynamics.number_densities,
                           create_jacobian=False,
                           limit_rates=limit_rates)[0]
    final_number_densities.append(final)
    print(f"Done {i+1} time of {len(times)}")

  final_number_densities = np.array(final_number_densities).T
  return final_number_densities


def run_and_plot(temperatures: List, times: List, network: Network,
                 filename: str, limit_rates=False):
  fig, axes = plt.subplots(3, 3, figsize=(10, 12), sharex=True)
  # for i, (density, temperature) in enumerate(product(densities, temperatures)):
  for i, temperature in enumerate(temperatures):
    idx_x = i // 3
    idx_y = i % 3
    dynamics = setup_dynamics(network, temperature)
    number_densities = solve_dynamics(dynamics, times, limit_rates=limit_rates)

    print(f"Done T = {temperature} K: {i+1} of {len(temperatures)}")

    total_number_density = np.sum(number_densities, axis=0)
    hydrogen_number_density = number_densities[network.species.index('H')]
    for s, n, c in zip(network.species, number_densities, colours):
      # if s in ['C', 'O', 'H', 'M']:
      #   continue
      # Number density ratio
      # axes[idx_x, idx_y].plot(np.log10(times), np.log10(n/total_number_density),
      #                         label=s, color=c, ls='-')
      # Abundance
      abundance = 12 + np.log10(n / hydrogen_number_density)
      # axes[idx_x, idx_y].set_ylim(-13, 13)
      axes[idx_x, idx_y].set_ylim(-2, 13)
      axes[idx_x, idx_y].plot(np.log10(times), abundance,
                              label=s, color=c, ls='-')
      # Plot problem area where M abundance is higher than 11
      if s == 'M':
        problem_mask = (abundance > 11.005)
        if len(times[problem_mask]) > 0:
          axes[idx_x, idx_y].axvline(np.log10(times)[problem_mask][0],
                                     c='k', ls='--')

    axes[idx_x, idx_y].set_title(f"{temperature} K")
    axes[idx_x, idx_y].legend()
    if idx_x == 2:
      axes[idx_x, idx_y].set_xlabel("log time [s]")
    if idx_y == 0:
      # axes[idx_x, idx_y].set_ylabel("log number density ratio")
      axes[idx_x, idx_y].set_ylabel("Abundance")

  plt.savefig(filename, bbox_inches="tight")


# network.to_krome_format('./test.ntw')
# network.to_cobold_format('./test.dat')
# test_network = Network.from_krome_file('./test.ntw')
# print([f"{reaction}\n" for reaction in test_network.reactions])

# Kinetics with limits
# network = Network.from_krome_file("../res/simplified_co.ntw")
# temperatures = [300, 1000, 3000, 5000, 7500, 10000, 15000, 20000, 30000]
temperatures = [5000, 10000, 20000]
times = np.logspace(-6, 3, num=10)
colours = ['b', 'g', 'r', 'gold', 'purple', 'violet', 'sienna', 'teal']
print(f"Solving unlimited rates case.")
filename = f"../out/figs/solar_network_unlimited_simplified.png"
run_and_plot(temperatures, times, network, filename, limit_rates=False)
# for limit in ['boundary', 'weak', 'sharp']:
#   network.set_reaction_limit(limit)
#   print(f"Solving with {limit} limit.")
#   filename = f"../out/figs/solar_network_{limit}.png"
#   run_and_plot(temperatures, times, network, filename, limit_rates=True)
plt.show()
