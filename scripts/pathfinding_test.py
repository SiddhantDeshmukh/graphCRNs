from typing import Dict, List
from gcrn.network import Network
from gcrn.helper_functions import number_densities_from_abundances
from gcrn.pathfinding import all_paths
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density

import numpy as np
from itertools import product
import networkx as nx

from abundances import *

mass_hydrogen = 1.67262171e-24  # [g]


def pprint_dict(dic: Dict):
  # Pretty-print dictionary
  print("{")
  for k, v in dic.items():
    print(f"\t{k}: {v}")
  print("}")


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


# Test custom pathfinding
def main():
  network_dir = '../res'
  # network_file = f"{network_dir}/solar_co_w05.ntw"
  # network_file = f"{network_dir}/ch_oh_co.ntw"
  network_file = f"{network_dir}/cno.ntw"
  network = Network.from_krome_file(network_file)

  abundances = mm30a04_abundances
  # abundances = mm30a04c20n20o04_abundances
  # abundances = mm30a04c20n20o20_abundances
  print("Initial abundances: ")
  pprint_dict(abundances)
  source_targets = [
      ('C', 'CO'),
      ('O', 'CO'),
      ('C', 'CH'),
      ('C', 'CN'),
      ('C', 'C2'),
      ('O', 'OH')
  ]
  # reverse paths
  source_targets += [(t, s) for s, t in source_targets]
  source_target_pairs = [f'{s}-{t}' for s, t in source_targets]
  sources, targets = [[l[i] for l in source_targets] for i in range(2)]

  # Evaluate at a single density-temperature
  temperature = 3500.
  density = 1e-8
  times = np.logspace(-5, 5, num=100)
  n = calculate_number_densities(abundances, np.log10(density))
  network.temperature = temperature
  network.number_densities = n
  n = network.solve(times, eqm_tolerance=1e-20, n_subtime=10,
                    return_eqm_times=False)
  print(f"\nT = {temperature:.1f} [K], log rho = {np.log10(density):.1f}")
  print(f"t = {times[-1]:.1e} [s]\n")
  print("="*80)
  # TODO:
  # Evaluate at various times
  for source, target in zip(sources, targets):
    paths, path_lengths = all_paths(network, source, target,
                                    cutoff=3, max_paths=5)

    # TODO:
    # Function that computes path lengths from the path
    stitched_paths = []
    for path, lengths in zip(paths, path_lengths):
      stitched_path = {}
      for i in range(len(path)):
        if i == len(path) - 1:
          break
        stitched_path[f'{path[i]} -> {path[i+1]}'] = lengths[i+1]

      stitched_paths.append(stitched_path)

    # Sort from shortest to longest total timescale
    stitched_paths = sorted(stitched_paths, key=lambda x: sum(x.values()))
    print(f'{len(stitched_paths)} paths from {source} -> {target}.')
    for s_path in stitched_paths:
      print(f'\t{len(s_path)} steps:')
      total_timescale = 0.
      current_step = 1
      for step, timescale in s_path.items():
        if timescale != 0:
          print(
              f'\t\t{current_step}. {step} with log timescale {np.log10(timescale):.2f}')
          current_step += 1
        else:
          print(f'\t\t(s.j.) {step}')
        total_timescale += timescale
      print(f'\t\tTotal = {np.log10(total_timescale):.2f} (log) [s / cm^3]')
      current_step = 0

  print(network.crea)


if __name__ == "__main__":
  main()


"""
TODO:
For thesis (not necessary for paper):
  - Run pathfinding at each full-step in GCRN evolution (switch)
  - Only run equilibrium cut-off if specified
"""
