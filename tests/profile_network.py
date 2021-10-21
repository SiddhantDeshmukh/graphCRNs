# Profile a CRN to investigate species, reactant-product balance, pathfinding
# and timescales
from itertools import product
from typing import Dict, List, Union

from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density, log_abundance_to_number_density
from gcrn.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from copy import copy, deepcopy

from gcrn.reaction import Reaction
import re

np.random.seed(42)


def find_reaction_in_network(source_complex: str, target_complex: str,
                             network: Network) -> Reaction:
  # Find the first reaction corresponding to 'source_complex -> target_complex'
  # in 'network'. Used in pathfinding to relate paths back to reactions.
  for rxn in network.reactions:
    if rxn.reactant_complex == source_complex and rxn.product_complex == target_complex:
      return rxn

  return None


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
  G = deepcopy(network.complex_graph)
  edges_to_add = []

  # Add source and target (sink) connections
  for complex in G.nodes():
    # Split the complex so we can match the parts instead of potentially
    # matching strings (e.g. 'H' would match 'in H2')
    complex_species = [c.strip() for c in complex.split('+')]
    # print(complex_species, source, target)
    if target in complex_species:
      # print(f"{complex} to {target} sink connection")
      edges_to_add.append((complex, target, 0))

    # Add source connections
    if source in complex_species:
      # print(f"{source} to {complex} source connection")
      edges_to_add.append((source, complex, 0))

  # NOTE 20/10/21:
  # Upon reconsideration, I don't see why I need these helper species. They just
  # allow jumps between species, but to answer the question "source -> target",
  # I don't want any of these jumps!

  # Add helper species connections
  # for i, s in enumerate(network.species):
  #   if s == target or s == source:
  #     continue
  #   for complex in G.nodes():
  #     # Add species connection to node
  #     complex_species = [c.strip() for c in complex.split('+')]
  #     if not complex in network.species and not target in complex_species and s in complex_species:
  #       # TODO:
  #       # Don't connect to complex if the target is in it
  #       # - is this actually what we want? think about the logic a bit more
  #       # print(f"{s} to {complex} undirected connection")
  #       edges_to_add.append((s, complex, 0))
  #       edges_to_add.append((complex, s, 0))
  G.add_weighted_edges_from(edges_to_add)

  return G


def find_paths(network: Network, source: str, target: str, cutoff=4,
               max_paths=100) -> Union[List, List]:
  # Find the 'num_paths' shortest paths between 'target' and 'source' of max
  # length 'cutoff'
  unique_paths = []
  unique_lengths = []
  reaction_paths = []

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
  # plt.figure()
  # nx.draw(search_graph, **options)
  # nx.spring_layout(search_graph)

  # print(source, target)
  # plt.show()
  # exit()
  shortest_paths = nx.all_simple_paths(search_graph, source, target,
                                       cutoff=cutoff)
  count = 0
  # Measure length of each path
  for path in shortest_paths:
    total_length = 0
    for i in range(len(path) - 1):
      source, target = path[i], path[i+1]
      edge = search_graph[source][target][0]
      length = edge['weight']
      total_length += length

    # Replace strings with reactions
    rxn_path = []
    had_prev_reaction = False
    # print("new path")
    # print(path)
    for i, entry in enumerate(path):
      if i == len(path) - 1:
        continue
      rxn = find_reaction_in_network(entry, path[i+1], network)
      # print(f"{entry} -> {path[i+1]}:  {rxn}")
      if rxn:
        rxn_path.append(rxn)
        had_prev_reaction = True
      else:
        if not had_prev_reaction:
          rxn_path.append(entry)
        had_prev_reaction = False
    # Add final (target) species
    rxn_path.append(path[-1])
    # print(" -> ".join([f"{e.idx}" if isinstance(e, Reaction) else f"{str(e)}"
    #                    for e in rxn_path]))

    string_path = ' -> '.join(path)
    if total_length > 0 and not string_path in unique_paths:
      unique_paths.append(string_path)
      unique_lengths.append(total_length)
      reaction_paths.append(rxn_path)

    count += 1
    if count >= max_paths:
      break

  return unique_paths, unique_lengths, reaction_paths


def find_network_paths(network: Network, sources: List, targets: List, cutoff=4,
                       max_paths=100) -> Union[Dict, Dict]:
  # Find 'num_paths' paths with a max length of 'cutoff' between
  # each source-target pair (zip(sources, targets))
  # Hence, sources and targest should be the same length
  all_unique_paths = {}
  all_unique_lengths = {}
  all_unique_rxn_paths = {}
  for source, target in zip(sources, targets):
    if source == target:
      continue
    key = f"{source}-{target}"
    paths, lengths, rxn_paths = find_paths(network, source, target,
                                           cutoff=cutoff, max_paths=max_paths)
    all_unique_paths[key] = paths
    all_unique_lengths[key] = lengths
    all_unique_rxn_paths[key] = rxn_paths

  return all_unique_paths, all_unique_lengths, all_unique_rxn_paths


network_dir = '../res'
network_file = f"{network_dir}/solar_co_w05.ntw"
# network_file = f"{network_dir}/co_test.ntw"
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
network.temperature = 5000
# print(initial_number_densities)
sources = ['H', 'H2', 'C', 'C', 'CH', 'CO']
targets = ['H2', 'H', 'CH', 'CO', 'C', 'C']
# sources = ['H', 'H2']
# targets = ['H2', 'H']

# plt.figure()
# nx.draw(network.species_graph)
# plt.show()

unique_paths, unique_lengths, rxn_paths = find_network_paths(
    network, sources, targets)

rxn_idx_paths = {}
for key in unique_paths.keys():
  rxn_idx_paths[key] = []
  print(f"{len(unique_paths[key])} paths and lengths for {key}:")
  for path, length, rxn_path in zip(unique_paths[key], unique_lengths[key], rxn_paths[key]):
    print("\n".join([f"  {path}\tLength = {length:.2e}"]))
    # Cast elements of rxn_path to str
    rxn_idx_path = " -> ".join([f"{e.idx}" if isinstance(e, Reaction)
                                else f"{str(e)}" for e in rxn_path])
    print("\n".join([f"  {rxn_idx_path}"]))
    rxn_idx_paths[key].append(rxn_idx_path)


# Count occurrences of rxn indices weighted by path length to find most
# travelled pathways
rxn_counts = {}
for key in unique_paths.keys():
  for path, length in zip(rxn_idx_paths[key], unique_lengths[key]):
    # print(path, length)
    rxn_idxs = re.findall(r" \d+ ", path)
    for idx in rxn_idxs:
      idx = idx.strip()
      # print(path, rxn_idxs, idx)
      if not idx in rxn_counts:
        rxn_counts[idx] = 1
      else:
        rxn_counts[idx] += 1

print("\n".join([f'{k}: {v}' for k, v in rxn_counts.items()]))

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
