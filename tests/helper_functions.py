
# %%
# Test temperature limits and fading functions
from itertools import product
from sadtools.utilities.chemistryUtilities import log_abundance_to_number_density
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from src.network import Network
from src.dynamics import NetworkDynamics


def enumerated_product(*args):
  # TODO:
  # Move to utilities and reference
  # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
  yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def setup_number_densities(number_densities_dict: Dict, hydrogen_density: float,
                           network: Network):
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(number_densities_dict[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n

  return number_densities


def setup_dynamics(network: Network, temperature: float,
                   initial_number_densities: Dict, hydrogen_density: float):
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
    solver_kwargs = {
        'min_step': None,
        # 'max_step': time / 1e2
        'max_step': None
    }

    final = dynamics.solve(time, dynamics.number_densities,
                           create_jacobian=True,
                           limit_rates=limit_rates,
                           **solver_kwargs)[0]
    final_number_densities.append(final)
    print(f"Done {i+1} time of {len(times)}")

  final_number_densities = np.array(final_number_densities).T
  return final_number_densities


def run_and_plot(temperatures: List, times: List, network: Network,
                 filename: str, initial_number_densities: Dict,
                 hydrogen_density: float,
                 limit_rates=False):
  fig, axes = plt.subplots(3, 3, figsize=(10, 12), sharex=True)
  # for i, (density, temperature) in enumerate(product(densities, temperatures)):
  for i, temperature in enumerate(temperatures):
    idx_x = i // 3
    idx_y = i % 3
    dynamics = setup_dynamics(network, temperature,
                              initial_number_densities, hydrogen_density)
    number_densities = solve_dynamics(dynamics, times, limit_rates=limit_rates)

    print(f"Done T = {temperature} K: {i+1} of {len(temperatures)}")

    total_number_density = np.sum(number_densities, axis=0)
    # Total number density
    # axes[idx_x, idx_y].plot(np.log10(times), np.log10(total_number_density),
    #                         label="Total")
    axes[idx_x, idx_y].plot(np.log10(times), total_number_density / total_number_density[0],
                            label="Total")
    # axes[idx_x, idx_y].set_ylim(17, 18)
    hydrogen_number_density = number_densities[network.species.index('H')]
    for s, n in zip(network.species, number_densities):
      # if s in ['C', 'O', 'H', 'M']:
      #   continue
      # Number density ratio
      axes[idx_x, idx_y].plot(np.log10(times), np.log10(n/total_number_density),
                              label=s, ls='-')
      # axes[idx_x, idx_y].plot(np.log10(times), np.log10(n),
      #                         label=s, ls='-')
      # Abundance
      # abundance = 12 + np.log10(n / hydrogen_number_density)
      # axes[idx_x, idx_y].set_ylim(-13, 13)
      # axes[idx_x, idx_y].set_ylim(-2, 13)
      # axes[idx_x, idx_y].plot(np.log10(times), abundance,
      #                         label=s,  ls='-')
      # Plot problem area where M abundance is higher than 11
      # if s == 'M':
      #   problem_mask = (abundance > 11.005)
      #   if len(times[problem_mask]) > 0:
      #     axes[idx_x, idx_y].axvline(np.log10(times)[problem_mask][0],
      #                                c='k', ls='--')

    axes[idx_x, idx_y].set_title(f"{temperature} K")
    axes[idx_x, idx_y].legend()
    if idx_x == 2:
      axes[idx_x, idx_y].set_xlabel("log time [s]")
    if idx_y == 0:
      axes[idx_x, idx_y].set_ylabel("log number density")
      # axes[idx_x, idx_y].set_ylabel("log number density ratio")
      # axes[idx_x, idx_y].set_ylabel("Abundance")

  plt.savefig(filename, bbox_inches="tight")
  plt.show()
