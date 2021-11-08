# Profile a CRN to investigate species, reactant-product balance, pathfinding
# and timescales
from itertools import product
from typing import Dict, List

from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from gcrn.network import Network
from gcrn.dynamics import NetworkDynamics
from gcrn.pathfinding import find_network_paths
from gcrn.helper_functions import setup_number_densities
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
  rxn_counts: Dict

  # TODO:
  # View rho-T grid from source-target perspective:
  # foreach pair, order reactions based on path length


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
      ('C', 'CH'),
      ('C', 'CN'),
  ]
  # reverse paths
  source_targets += [(t, s) for s, t in source_targets]
  source_target_pairs = [f'{s}-{t}' for s, t in source_targets]
  sources, targets = [l[0] for l in source_targets], [l[1]
                                                      for l in source_targets]
  print(sources)
  print(targets)
  # plt.figure()
  # nx.draw(network.species_graph)
  # plt.show()
  temperatures = np.linspace(3000, 15000, num=10)
  densities = np.logspace(-12, -6, num=10)

  rxn_counts = {}
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
    dynamics = NetworkDynamics(network, network.number_densities, T)
    n = dynamics.solve(1e3, n)[0]

    # Package back into dictionary for network
    for i, s in enumerate(network.species):
      network.number_densities[s] = n[i]

    # Find unique pathways for specified {source, target} pairs
    unique_paths, unique_lengths, rxn_paths = find_network_paths(
        network, sources, targets)

    rxn_idx_paths = {}
    for key in unique_paths.keys():
      rxn_idx_paths[key] = []
      for path, length, rxn_path in zip(unique_paths[key], unique_lengths[key], rxn_paths[key]):
        # Cast elements of rxn_path to str
        rxn_idx_path = " -> ".join([f"{e.idx}" if isinstance(e, Reaction)
                                    else f"{str(e)}" for e in rxn_path])
        rxn_idx_paths[key].append(rxn_idx_path)
        # print(rxn_idx_path)

    # Count occurrences of rxn indices weighted by path length to find most
    # travelled pathways
    for key in unique_paths.keys():
      for path, length in zip(rxn_idx_paths[key], unique_lengths[key]):
        rxn_idxs = re.findall(r" \d+ ", path)
        for idx in rxn_idxs:
          idx = idx.strip()
          if not idx in rxn_counts:
            rxn_counts[idx] = length
          else:
            rxn_counts[idx] += length
    rxn_tuple.append(PointPaths(rho, T, unique_paths, unique_lengths,
                                rxn_idx_paths, rxn_counts))

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

  n_rxns = 5  # find 'n' fastest pathways
  for r in rxn_tuple:
    print(f'{r.density:.3e}, {r.temperature:.3e}')
    for pair in source_target_pairs:
      print('\n'.join([f'\t{p}: {l:.3e}'
            for p, l, rp in zip(r.unique_paths[pair][:n_rxns],
                                r.unique_lengths[pair][:n_rxns],
                                r.rxn_paths[pair][:n_rxns])]))


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

# NOTE:
# - Just finding 'reaction counts' can be done independently of path lengths,
#   it only depends on connections, not weights. Weight come into play when
#   finding shortest paths! Hence finding shortest paths is doing both problems
#   simultaneously, and I can keep track of the entire formation/disassociation
#   path lengths. When calculating just the counts, I'm looking at the network
#   as a whole to find which reactions are the most important. So these are two
#   separate ways of viewing simplification which hopefully give the same result
#   since they come from the same assumptions and also use the same inputs.


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
