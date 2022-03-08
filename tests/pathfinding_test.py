from typing import List
from gcrn.network import Network
from gcrn.dynamics import NetworkDynamics
from gcrn.helper_functions import setup_number_densities
from gcrn.pathfinding import all_paths
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density

import numpy as np
from itertools import product
import networkx as nx


# Test custom pathfinding
def main():
  network_dir = '../res'
  network_file = f"{network_dir}/solar_co_w05.ntw"
  # network_file = f"{network_dir}/ch_oh_co.ntw"
  # network_file = f"{network_dir}/cno.ntw"
  network = Network.from_krome_file(network_file)

  print("Pathfinding")

  initial_number_densities = {
      "H": 1e12,
      "H2": 1e-4,
      "OH": 1e-12,
      "C": 10**(8.39),  # solar
      "O": 10**(8.66),  # solar
      "N": 10**(7.83),
      "CH": 1e-12,
      "CO": 1e-12,
      "CN": 1e-12,
      "NH": 1e-12,
      "NO": 1e-12,
      "C2": 1e-12,
      "O2": 1e-12,
      "N2": 1e-12,
      "M": 1e11,
  }
  source_targets = [
      ('C', 'CO'),
      ('O', 'CO'),
      ('C', 'CH'),
      # ('C', 'CN'),
  ]
  # reverse paths
  source_targets += [(t, s) for s, t in source_targets]
  source_target_pairs = [f'{s}-{t}' for s, t in source_targets]
  sources, targets = [[l[i] for l in source_targets] for i in range(2)]
  # sources = [l[0] for l in source_targets]
  # targets = [l[1] for l in source_targets]

  temperatures = np.linspace(3000, 15000, num=10)
  densities = np.logspace(-12, -6, num=10)

  # TODO:
  # Run over histogram, not a product
  for T, rho in product(temperatures, densities):
    network.temperature = T
    hydrogen_density = gas_density_to_hydrogen_number_density(rho)
    network.number_densities = \
        setup_number_densities(initial_number_densities,
                               hydrogen_density, network)
    n = np.zeros(len(network.species))
    for i, s in enumerate(network.species):
      n[i] = network.number_densities[s]

    # Solve dynamics for 100 seconds to get reasonable number densities
    dynamics = NetworkDynamics(network, network.number_densities, T)
    n = dynamics.solve([1e2], n)[0]

    # Package back into dictionary for network
    for i, s in enumerate(network.species):
      network.number_densities[s] = n[i]

    # Find unique pathways for specified {source, target} pairs
    paths, path_lengths = all_paths(network, 'C', 'CO', cutoff=3, max_paths=5)
    # paths, path_lengths = all_paths(network, 'O', 'CO', cutoff=3, max_paths=10)
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
    print(f'{len(stitched_paths)} paths.')
    for s_path in stitched_paths:
      print(f'{len(s_path)} steps:')
      total_timescale = 0.
      for step, timescale in s_path.items():
        print(f'\t{step} with timescale {timescale:.2e}')
        total_timescale += timescale
      print(f'\tTotal = {total_timescale:.2e} [s / cm^3]')
    exit()  # one T, rho iteration


if __name__ == "__main__":
  main()
