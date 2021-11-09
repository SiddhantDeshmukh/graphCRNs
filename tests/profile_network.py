# Profile a CRN to investigate species, reactant-product balance, pathfinding
# and timescales
from copy import deepcopy
from itertools import product, chain
from typing import Dict, List

from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from gcrn.network import Network
from gcrn.dynamics import NetworkDynamics
from gcrn.pathfinding import find_network_paths, rxn_idx_paths_from_rxn_paths
from gcrn.helper_functions import setup_number_densities, find_reaction_from_idx
import numpy as np

from gcrn.reaction import Reaction
import re
from dataclasses import dataclass

np.random.seed(42)


@dataclass
class PointPaths:
  density: float
  temperature: float
  unique_paths: Dict
  unique_lengths: Dict
  rxn_paths: Dict
  all_rxn_counts: Dict
  pair_rxn_counts: Dict

  # TODO:
  # View rho-T grid from source-target perspective:
  # foreach pair, order reactions based on path length


def sort_dict(d: Dict, reverse=True):
  # Sort dictionary by values
  return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


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


def rxn_idxs_from_path(path: str) -> List[str]:
  # From a path e.g. 'C -> 75 -> 244 -> CO' return indices of reactions
  # in the order they appear [75, 244]
  # For a single-path reaction, returns a list of one entry, e.g.
  # 'C -> 76 -> CO': [76]
  return re.findall(r"(?<=-> )\d+(?= ->)", path)


def count_rxns_for_paths(paths: List) -> Dict:
  # Count the occurrences of reactions in the specified dictionary,
  # returning a dictionary {rxn_idx: count}
  counts = {}
  for path in paths:
    rxn_idxs = rxn_idxs_from_path(path)
    for idx in rxn_idxs:
      if not idx in counts:
        counts[idx] = 1
      else:
        counts[idx] += 1

  return counts


def count_rxns_by_pairs(rxn_idx_paths: Dict) -> Dict:
  # Count the occurrences of reactions in the specified dictionary of paths,
  # returning a dictionary of dictionaries {pair: {rxn_idx: count}}
  rxn_counts = {}
  for key, paths in rxn_idx_paths.items():
    rxn_counts[key] = count_rxns_for_paths(paths)

  return rxn_counts


def count_all_rxns(rxn_idx_paths: Dict) -> Dict:
  # Count the occurrences of reactions in the specified dictionary of paths,
  # returning a dictionary {rxn_idx: count}
  rxn_counts = {}
  for key in rxn_idx_paths.keys():
    for path in rxn_idx_paths[key]:
      rxn_idxs = rxn_idxs_from_path(path)
      for idx in rxn_idxs:
        if not idx in rxn_counts:
          rxn_counts[idx] = 1
        else:
          rxn_counts[idx] += 1

  return rxn_counts


def species_counts_from_rxn_counts(rxn_counts: Dict, network: Network) -> Dict:
  # From a dict {rxn_idx: count}, find the associated reaction's reactant
  # species and associate the count of the reaction with the count of the
  # species
  species_counts = {}
  for k, v in rxn_counts.items():
    reactants = find_reaction_from_idx(k, network).reactants
    for r in reactants:
      if r in species_counts:
        species_counts[r] += reactants.count(r) * v
      else:
        species_counts[r] = reactants.count(r) * v

  return species_counts


def main():
  network_dir = '../res'
  # network_file = f"{network_dir}/solar_co_w05.ntw"
  # network_file = f"{network_dir}/co_test.ntw"
  network_file = f"{network_dir}/cno.ntw"
  network = Network.from_krome_file(network_file)

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
      # ('C', 'CH'),
      # ('C', 'CN'),
  ]
  # reverse paths
  source_targets += [(t, s) for s, t in source_targets]
  source_target_pairs = [f'{s}-{t}' for s, t in source_targets]
  sources, targets = [l[0] for l in source_targets], [l[1]
                                                      for l in source_targets]
  # print(sources)
  # print(targets)
  # plt.figure()
  # nx.draw(network.species_graph)
  # plt.show()
  temperatures = np.linspace(3000, 15000, num=10)
  densities = np.logspace(-12, -6, num=10)

  rxn_tuple: List[PointPaths] = []
  for T, rho in product(temperatures, densities):
    # print(f'rho = {rho:.3e}, T = {T:.3e}')
    network.temperature = T
    hydrogen_density = gas_density_to_hydrogen_number_density(rho)
    network.number_densities = \
        setup_number_densities(initial_number_densities,
                               hydrogen_density, network)
    # TODO:
    # Move to networkdynamics init, ensuring number densities is an array
    # indexed as species there (it is a dict in network)
    n = np.zeros(len(network.species))
    for i, s in enumerate(network.species):
      n[i] = network.number_densities[s]

    # Solve dynamics for 100 seconds to get reasonable number densities
    # TODO:
    # Add timescale as a parameter
    dynamics = NetworkDynamics(network, network.number_densities, T)
    n = dynamics.solve(1e2, n)[0]

    # Package back into dictionary for network
    for i, s in enumerate(network.species):
      network.number_densities[s] = n[i]

    # Find unique pathways for specified {source, target} pairs
    unique_paths, unique_lengths, rxn_paths = find_network_paths(
        network, sources, targets, cutoff=5, max_paths=5)

    rxn_idx_paths = rxn_idx_paths_from_rxn_paths(rxn_paths)

    all_rxn_counts = count_all_rxns(rxn_idx_paths)
    counts_by_pairs = count_rxns_by_pairs(rxn_idx_paths)
    rxn_tuple.append(PointPaths(rho, T, unique_paths, unique_lengths,
                                rxn_idx_paths, all_rxn_counts, counts_by_pairs))

  # print("\n".join([f'{k}: {v:.3e}' for k, v in rxn_counts.items()]))
  # TODO:
  # Fix sorting, sort dictionaries instead, add into PointPaths
  for r in rxn_tuple:
    for key in r.unique_lengths.keys():
      # Sort by descending path length
      lengths, paths, r_paths = map(list,
                                    zip(*sorted(zip(r.unique_lengths[key],
                                                    r.unique_paths[key],
                                                    r.rxn_paths[key]))))
      r.unique_lengths[key] = lengths
      r.unique_paths[key] = paths
      r.rxn_paths[key] = r_paths

  total_counts = {}
  for r in rxn_tuple:
    # print(f'{r.density:.3e}, {r.temperature:.3e}')
    for pair in source_target_pairs:
      counts = {}
      # print('\n'.join([f'\t{rp}: {l:.3e}'
      #       for p, l, rp in zip(r.unique_paths[pair][:n_rxns],
      #                           r.unique_lengths[pair][:n_rxns],
      #                           r.rxn_paths[pair][:n_rxns])]))
    for k, v in r.all_rxn_counts.items():
      try:
        counts[k] += v
      except KeyError:
        counts[k] = v

    for k, v in counts.items():
      try:
        total_counts[k] += v
      except KeyError:
        total_counts[k] = v

  # Sort by values
  total_counts = sort_dict(total_counts)

  print(total_counts)
  print(len(total_counts), len(network.reactions))

  # Find most important species by looking up reactants of most important rxns
  species_counts = species_counts_from_rxn_counts(total_counts, network)

  # Sort by values
  species_counts = sort_dict(species_counts)
  print(species_counts)

  # Counts by pairs across grid
  pair_counts = {p: {} for p in source_target_pairs}
  for pair in source_target_pairs:
    for point in rxn_tuple:
      for k, v in point.pair_rxn_counts[pair].items():
        try:
          pair_counts[pair][k] += v
        except KeyError:
          pair_counts[pair][k] = v

  # Species count by pairs
  pair_species_counts = {k: sort_dict(species_counts_from_rxn_counts(v, network))
                         for k, v in pair_counts.items()}

  print(pair_species_counts)


if __name__ == "__main__":
  main()


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
