# Profile a CRN to investigate species, reactant-product balance, pathfinding
# and timescales
from itertools import product
from typing import Dict, List, Union

from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density, log_abundance_to_number_density
from src.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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


def find_paths(network: Network, source: str, target: str, cutoff=4,
               max_paths=100) -> Union[List, List]:
  # Find the 'num_paths' shortest paths between 'target' and 'source' of max
  # length 'cutoff'
  unique_paths = []
  unique_lengths = []
  shortest_paths = nx.all_simple_paths(network.species_graph, source, target,
                                       cutoff=cutoff)
  count = 0
  for path in shortest_paths:
    total_length = 0
    for i in range(len(path) - 1):
      source, target = path[i], path[i+1]
      edge = network.species_graph[source][target][0]
      length = edge['weight']
      total_length += length

    string_path = ','.join(path)
    if not string_path in unique_paths:
      unique_paths.append(string_path)
      unique_lengths.append(total_length)

    count += 1
    if count >= max_paths:
      break

  return unique_paths, unique_lengths


def find_network_paths(network: Network, sources: List, targets: List, cutoff=4,
                       max_paths=100) -> Union[Dict, Dict]:
  # Find 'num_paths' paths with a max length of 'cutoff' between every source
  # and target provided
  all_unique_paths = {}
  all_unique_lengths = {}
  for source, target in product(sources, targets):
    key = f"{source}-{target}"
    paths, lengths = find_paths(network, source, target,
                                cutoff=cutoff, max_paths=max_paths)
    all_unique_paths[key] = paths
    all_unique_lengths[key] = lengths

  return all_unique_paths, all_unique_lengths


network_dir = '../res'
network_file = f"{network_dir}/react-solar-umist12"
# network_file = f"{network_dir}/mp_cno.ntw"
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
    "C": 10**(8.39),
    "O": 10**(8.66),
    "CH": 1e-12,
    "CO": 1e-12,
    "M": 1e11,
}

gas_density = 1e-6  # [g cm^-3]
hydrogen_density = gas_density_to_hydrogen_number_density(gas_density)

# Index number densities same as network.species
for i, s in enumerate(network.species):
  initial_number_densities[s] = log_abundance_to_number_density(np.log10(initial_number_densities[s]),
                                                                np.log10(hydrogen_density.value))

network.number_densities = initial_number_densities
print(initial_number_densities)
sources = ['C', 'O']
targets = ['CO', 'CH']

# plt.figure()
# nx.draw(network.species_graph)
# plt.show()

unique_paths, unique_lengths = find_network_paths(network, sources, targets)

for key in unique_paths.keys():
  print(f"{len(unique_paths[key])} paths and lengths for {key}:")
  for path, length in zip(unique_paths[key], unique_lengths[key]):
    print("\n".join([f"  {path}\tLength = {length:.2e}"]))
