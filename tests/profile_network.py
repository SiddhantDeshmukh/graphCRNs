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


def add_species_paths(network: Network, source: str, target: str):
  # Add species nodes and edges (complex composition) into network graph
  # source/target dependent: don't add edges to species other than the
  # target species for any complex that contains the target species
  # Source node only has outgoing edges (source)
  # Target node only has incoming edges (sink)
  G = network.complex_graph
  edges_to_add = []

  # Add source and target (sink) connections
  for complex in G.nodes():
    # Split the complex so we can match the parts instead of potentially
    # matching strings (e.g. 'H' would match 'in H2')
    complex_species = [c.strip() for c in complex.split('+')]
    print(complex_species, source, target)
    if target in complex_species:
      print(f"{complex} to {target} sink connection")
      edges_to_add.append((complex, target, 0))

    # Add source connections
    if source in complex_species:
      print(f"{source} to {complex} source connection")
      edges_to_add.append((source, complex, 0))

  # Add helper species connections
  for i, s in enumerate(network.species):
    if s == target or s == source:
      continue
    for complex in G.nodes():
      # Add species connection to node
      complex_species = [c.strip() for c in complex.split('+')]
      if not complex in network.species and s in complex_species:
        print(f"{s} to {complex} undirected connection")
        edges_to_add.append((s, complex, 0))
        edges_to_add.append((complex, s, 0))
  G.add_weighted_edges_from(edges_to_add)

  return G


def find_paths(network: Network, source: str, target: str, cutoff=4,
               max_paths=100) -> Union[List, List]:
  # Find the 'num_paths' shortest paths between 'target' and 'source' of max
  # length 'cutoff'
  unique_paths = []
  unique_lengths = []

  plt.figure()

  search_graph = add_species_paths(network, source, target)
  edges, weights = zip(*nx.get_edge_attributes(search_graph, 'weight').items())

  options = {
      # "font_size": 24,
      # "node_size": 2000,
      "node_color": "white",
      "edgelist": edges,
      "edge_color": weights,
      # "linewidths": 5,
      # "width": 5,
      "with_labels": True
  }
  nx.draw(search_graph, **options)
  nx.spring_layout(search_graph)

  print(source, target)
  plt.show()
  # exit()
  shortest_paths = nx.all_simple_paths(search_graph, source, target,
                                       cutoff=cutoff)
  count = 0
  for path in shortest_paths:
    total_length = 0
    for i in range(len(path) - 1):
      source, target = path[i], path[i+1]
      edge = search_graph[source][target][0]
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
    if source == target:
      continue
    key = f"{source}-{target}"
    paths, lengths = find_paths(network, source, target,
                                cutoff=cutoff, max_paths=max_paths)
    all_unique_paths[key] = paths
    all_unique_lengths[key] = lengths

  return all_unique_paths, all_unique_lengths


network_dir = '../res'
# network_file = f"{network_dir}/solar_co_w05.ntw"
network_file = f"{network_dir}/co_test.ntw"
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
# sources = ['C', 'O']
# targets = ['CO', 'CH']
sources = ['H', 'H2']
targets = ['H2', 'H']

# plt.figure()
# nx.draw(network.species_graph)
# plt.show()

unique_paths, unique_lengths = find_network_paths(network, sources, targets)

for key in unique_paths.keys():
  print(f"{len(unique_paths[key])} paths and lengths for {key}:")
  for path, length in zip(unique_paths[key], unique_lengths[key]):
    print("\n".join([f"  {path}\tLength = {length:.2e}"]))
