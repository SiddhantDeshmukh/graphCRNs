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
  # network_file = f"{network_dir}/solar_co_w05.ntw"
  network_file = f"{network_dir}/ch_oh_co.ntw"
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
    paths, path_lengths = all_paths(network, 'C', 'CO', cutoff=3, max_paths=10)
    # paths, path_lengths = all_paths(network, 'O', 'CO', cutoff=3, max_paths=10)
    # TODO:
    # Function that computes path lengths from the path
    for path, lengths in zip(paths, path_lengths):
      print(len(path), path, lengths)
      # Stitch path components and lengths together (remember first 'None' is
      # the input heading into the source node so we can ignore it)
      # there should be n-1 lengths (edges) for n elements in the path (nodes)
    exit()


if __name__ == "__main__":
  main()
