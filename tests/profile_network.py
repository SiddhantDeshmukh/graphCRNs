# Profile a CRN to investigate species, reactant-product balance, pathfinding
# and timescales
from itertools import product
from typing import Dict, List

from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from gcrn.network import Network
from gcrn.dynamics import NetworkDynamics
from gcrn.pathfinding import find_network_paths, rxn_idx_paths_from_rxn_paths,\
    PointPaths
from gcrn.helper_functions import setup_number_densities, count_all_rxns, count_rxns_by_pairs, species_counts_from_rxn_counts
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from gcrn.profiling import most_important_species_by_pairs, most_travelled_pair_paths, total_counts_across_paths

np.random.seed(42)


def network_balance(species_count: Dict):
  # Compare the reactant and product sides of reactions and compute balance
  # between formation and disassociation reactions
  # Balance metric (for each species):
  # metric = (# product appearances - # reactant appearances)
  # Positive balance: Product-heavy
  # Negative balance: Reactant-heavy
  # Zero balance: Same number of reactant and product appearances
  balance_dict = {}
  for key, value in species_count.items():
    balance = value['P'] - value['R']
    if balance > 0:
      comment = "Skewed towards products"
    elif balance < 0:
      comment = "Skewed towards reactants"
    else:
      comment = "Balanced"

    balance_dict[key] = (balance, comment)

  return balance_dict


def sum_dicts(d1: Dict, d2: Dict) -> Dict:
  # Sum the values of common keys in 'd1', 'd2', and add all other unique values
  d = {}
  for k1, v1 in d1.items():
    for k2, v2 in d2.items():
      if k1 == k2:
        d[k2] = v1 + v2
      else:
        d[k1] = v1
        d[k2] = v2

  return d


def main():
  network_dir = '../res'
  # network_file = f"{network_dir}/solar_co_w05.ntw"
  # network_file = f"{network_dir}/co_test.ntw"
  network_file = f"{network_dir}/cno.ntw"
  network = Network.from_krome_file(network_file)

  print(len(network.species))
  print(len(network.reactions))
  species_count = network.network_species_count()
  balance_dict = network_balance(species_count)
  print("\n".join([f"{key}: {value}" for key, value in species_count.items()]))
  print("Network balance:")
  print("\n".join([f"{key}: {value}" for key, value in balance_dict.items()]))

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
  sources, targets = [l[0] for l in source_targets], [l[1]
                                                      for l in source_targets]
  # print(sources)
  # print(targets)
  plt.figure()
  options = {
      "node_color": "white",
      "with_labels": True,
      'font_size': 6,
      'arrows': False,
      'linewidths': 0,
      'width': 0

  }
  pos = {}
  num_nodes = len(network.species_graph.nodes)
  for i, n in enumerate(network.species_graph.nodes):
    x, y = i // num_nodes, i % num_nodes
    pos[n] = (x, y)
  nx.draw_random(network.species_graph,  **options)
  plt.show()
  # temperatures = np.linspace(3000, 15000, num=10)
  # densities = np.logspace(-12, -6, num=10)

  # points_list: List[PointPaths] = []
  # for T, rho in product(temperatures, densities):
  #   # print(f'rho = {rho:.3e}, T = {T:.3e}')
  #   network.temperature = T
  #   hydrogen_density = gas_density_to_hydrogen_number_density(rho)
  #   network.number_densities = \
  #       setup_number_densities(initial_number_densities,
  #                              hydrogen_density, network)
  #   # TODO:
  #   # Move to networkdynamics init, ensuring number densities is an array
  #   # indexed as species there (it is a dict in network)
  #   n = np.zeros(len(network.species))
  #   for i, s in enumerate(network.species):
  #     n[i] = network.number_densities[s]

  #   # Solve dynamics for 100 seconds to get reasonable number densities
  #   # TODO:
  #   # Add timescale as a parameter
  #   dynamics = NetworkDynamics(network, network.number_densities, T)
  #   n = dynamics.solve(1e2, n)[0]

  #   # Package back into dictionary for network
  #   for i, s in enumerate(network.species):
  #     network.number_densities[s] = n[i]

  #   # Find unique pathways for specified {source, target} pairs
  #   unique_paths, unique_lengths, rxn_paths = find_network_paths(
  #       network, sources, targets, cutoff=10, max_paths=100)

  #   rxn_idx_paths = rxn_idx_paths_from_rxn_paths(rxn_paths)

  #   all_rxn_counts = count_all_rxns(rxn_idx_paths)
  #   counts_by_pairs = count_rxns_by_pairs(rxn_idx_paths)
  #   points_list.append(PointPaths(rho, T, 1e2, unique_paths, unique_lengths,
  #                                 rxn_idx_paths, all_rxn_counts, counts_by_pairs))

  # for r in points_list:
  #   r.sort()

  # key_paths = most_travelled_pair_paths(points_list, source_target_pairs)

  # print("Most travelled pathways by source-target pair:")
  # for pair, paths in key_paths.items():
  #   print(pair)
  #   print("\n".join([f'\t{k}: {v}' for k, v in paths.items()]))

  # total_counts, species_counts = total_counts_across_paths(points_list, network)

  # print("Most important reactions / total reactions:")
  # print(f'{len(total_counts)} / {len(network.reactions)}')
  # print("Total species counts across grid")
  # print("\n".join([f'{k}: {v}' for k, v in species_counts.items()]))

  # pair_species_counts = most_important_species_by_pairs(points_list,
  #                                                       source_target_pairs,
  #                                                       network)
  # print("Most important species by source-target pair:")
  # for pair, counts in pair_species_counts.items():
  #   print(pair)
  #   print("\n".join([f'\t{k}: {v}' for k, v in counts.items()]))


if __name__ == "__main__":
  main()

# NOTE:
# All the counts for the pathways are the same, which is a little odd. Make sure
# there's no copy shenanigans with lists going on when creating PointPaths and
# counting instances

# TODO:
# Write a test network to exploit the above!

# Iterate over many thermodynamic states to find the most favourable reactions
# TODO:
# Do this for finding pathways instead! It is more natural.

# TODO:
# - Create network simplification pipeline
# - Create a metric for network balancing
#   - order current balance by number of occurrences
# - Investigate zero- and one-deficiency

# Network balancing
# Current implementation just counts number of times species appears as
# reactant/product, but this doesn't necessarily take into account the rate
# e.g. if H shows up twice but it's in the same reaction on the reactant side.
# mass-action law predicts a [H]^2 relation, not 2H if it had showed up once
# in 2 separate reactions.
# For this, then, it would be useful to profile the current network for a given
# density-temperature cell and compute rates of change, final densities, and
# balance based on rates. Can still have an "overall linear balance" metric that
# simply counts species on L/RHS of reactions and orders based on how many times
# the species appears

# Current implementation:
# Balance = n(P) - n(R)
# + gives a simple linear picture of how many times species on L/RHS
# - no picture on rates of change (mass-action not implemented)

# New idea:
# Balance = (n(P) - n(R)) / N
# N = total number of occurrences of species X
# OR
# N = total number of occurrences of all species
# Benefits of including all species is we can see which _species_ are dominating
# the network; however we can already do this with the linear metric, just
# adding up the n(P) and n(R) values.
# Can try both metrics, at the end of the day the outcomes of this are to find
# - which species dominate a given network
# - missing symmetries for reactions (if certain species are "one-side-heavy")

# New idea with rates:
# Compute rates for given density-temperature cell and _then_ check balance
# This balance is no longer the number of times a species appears on L/RHS but
# instead d[X]/dt (which is just the RHS of the ODE system)
# Then we can find dominant rates of change in a network at a given
# density-temperature point
