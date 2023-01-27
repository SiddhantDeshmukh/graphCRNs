from copy import deepcopy
import re
from typing import Union, List, Dict
import networkx as nx
from gcrn.graphs import create_complex_graph
from gcrn.network import Network
from gcrn.reaction import Reaction
from dataclasses import dataclass


@dataclass
class PointPaths:
  density: float
  temperature: float
  time: float
  unique_paths: Dict
  unique_lengths: Dict
  rxn_idx_paths: Dict
  all_rxn_counts: Dict
  pair_rxn_counts: Dict

  # TODO:
  # View rho-T grid from source-target perspective:
  # foreach pair, order reactions based on path length
  def sort(self, inplace=True):
    # Sort each dict by lengths (ascending)
    unique_lengths, unique_paths, rxn_idx_paths = {}, {}, {}
    for key in self.unique_lengths.keys():
      # Sort by path length
      lengths, paths, r_paths = map(list,
                                    zip(*sorted(zip(self.unique_lengths[key],
                                                    self.unique_paths[key],
                                                    self.rxn_idx_paths[key]))))
      unique_lengths[key] = lengths
      unique_paths[key] = paths
      rxn_idx_paths[key] = r_paths

    if inplace:
      self.unique_lengths = unique_lengths
      self.unique_paths = unique_paths
      self.rxn_idx_paths = rxn_idx_paths
    else:
      return PointPaths(self.density, self.temperature, self.time, unique_paths,
                        unique_lengths, rxn_idx_paths, self.all_rxn_counts,
                        self.pair_rxn_counts)


def is_in_complex(complex: str, species: str) -> bool:
  # Check if the species if present in any form in the complex, e.g.
  # C in C + O -> True
  # C in CH + O -> True
  # C in OH + H2 -> False
  complex_species = [c.strip() for c in complex.split('+')]
  return any([species in c for c in complex_species])


def create_search_graph(network: Network, source: str, target: str):
  # Add species nodes and edges (complex composition) into network graph
  # source/target dependent: don't add edges to species other than the
  # target species for any complex that contains the target species
  # Source node only has outgoing edges (source)
  # Target node only has incoming edges (sink)
  G = create_complex_graph(network)
  # G = deepcopy(network.complex_graph)
  # plt.figure()
  # options = {
  #     "node_color": "white",
  #     "with_labels": True
  # }
  # nx.draw(G, **options)
  # nx.spring_layout(G)
  edges_to_add = []

  # Add source and target (sink) connections
  for complex in G.nodes():
    # Split the complex so we can match the parts instead of potentially
    # matching strings (e.g. 'H' would match 'in H2')
    complex_species = [c.strip() for c in complex.split('+')]
    if target in complex_species:
      edges_to_add.append((complex, target, 0))

    # Add source connections
    if source in complex_species:
      edges_to_add.append((source, complex, 0))

  # Add helper species connections
  for i, s in enumerate(network.species):
    if s == target or s == source:
      continue
    for complex in G.nodes():
      # Add species connection to node
      # To ensure 'continuity' from source -> target, only add connections that
      # include the source in some form
      # e.g. for C -> CO, include every connection that has a 'C' in it, but
      # none that do not!
      complex_species = [c.strip() for c in complex.split('+')]
      if not complex in network.species and not target in complex_species and is_in_complex(complex, source) and s in complex_species:
        edges_to_add.append((s, complex, 0))
        edges_to_add.append((complex, s, 0))

  G.add_weighted_edges_from(edges_to_add)

  # Keep only the largest connected component (i.e. remove all complexes that
  # do not contribute to the current pathways)
  G = G.subgraph(max(nx.weakly_connected_components(G), key=len))
  return G


def find_paths(network: Network, source: str, target: str, cutoff=4,
               max_paths=100) -> Union[List, List]:
  # Find the 'num_paths' shortest paths between 'target' and 'source' of max
  # length 'cutoff'
  unique_paths = []
  unique_lengths = []
  reaction_paths = []

  search_graph = create_search_graph(network, source, target)

  # TODO:
  # Return the search
  # edges, weights = zip(*nx.get_edge_attributes(search_graph, 'weight').items())
  # options = {
  #     # "font_size": 24,
  #     # "node_size": 2000,
  #     "node_color": "white",
  #     "edgelist": edges,
  #     "edge_color": weights,
  #     # "linewidths": 5,
  #     # "width": 5,
  #     "with_labels": True
  # }
  # plt.figure(figsize=(12, 12))
  # nx.draw(search_graph, **options)
  # nx.spectral_layout(search_graph)

  # print(source, target)
  # plt.show()
  # exit()
  shortest_paths = nx.all_simple_paths(search_graph, source, target,
                                       cutoff=cutoff)
  count = 0
  # Measure length of each path
  for path in shortest_paths:
    # TODO:
    # Prune paths that do not include the 'source' in some form throughout
    # What about 'target'? For destruction, e.g. CO -> C, CO will disappear!
    species_in_path = [is_in_complex(p, source) or is_in_complex(p, target)
                       for p in path]
    if all(species_in_path):
      # print(path, species_in_path)

      # exit()
      total_length = 0
      for i in range(len(path) - 1):
        curr, next = path[i], path[i+1]
        edge = search_graph[curr][next][0]
        length = edge['weight']
        total_length += length

      # Replace strings with reactions
      rxn_path = rxn_path_from_path(path, network)
      string_path = ' -> '.join(path)
      if total_length > 0 and not string_path in unique_paths:
        unique_paths.append(string_path)
        unique_lengths.append(total_length)
        reaction_paths.append(rxn_path)

      # count += 1
      count += total_length  # scale count by length
      if count >= max_paths:
        break

  return unique_paths, unique_lengths, reaction_paths


def all_paths(network: Network, source: str, target: str,
              cutoff=4, max_paths=100):
  # Use a custom pathfinding alg to find paths on species-complex graph
  # with rules of species-jumps
  # Species-Complex graph:
  # Complex graph with edge connections + 0-weight edge connections to all
  # species if species in complex
  # This should allow the graph to become fully connected if the reaction
  # network in species is fully connected
  # Additionally, we use one graph for the entire network regardless of
  # {source,target}; the pathfinding takes care of zero-jumps
  G = create_complex_graph(network)
  edges_to_add = []
  for complex in G.nodes():
    if complex in network.species:
      continue
    # Split the complex so we can match the parts instead of potentially
    # matching strings (e.g. 'H' would match 'in H2')
    complex_species = [c.strip() for c in complex.split('+')]
    # Add species-complex connections (zero-weight)
    for s in network.species:
      if s in complex_species:
        edges_to_add.append((s, complex, 0))
        edges_to_add.append((complex, s, 0))

  G.add_weighted_edges_from(edges_to_add)

  # path_gen = nx.all_simple_paths(G, source, target, cutoff=6)
  # for path in path_gen:
  #   print(path)

  # plt.figure()
  # options = {
  #     "node_color": "white",
  #     "with_labels": True
  # }
  # nx.draw(G, **options)
  # nx.shell_layout(G)

  # plt.show()
  # exit()

  # RULES:
  # 1. Always start on a species 'source' node
  # 2. If current node is species 'target' node, return
  # 3. Find all nodes connected to current
  # 4. If last edge had a cost of zero, omit all potential nodes that have a
  #    cost of zero (prevents double species-jumps)
  # TODO:
  # Refactor this specific species-jump-prevention dfs to outside
  def dfs(current: str, target: str, visited_nodes: List, current_path: List,
          current_path_lengths: List, all_paths: List, all_path_lengths: List,
          previous_distance: float, cutoff=cutoff,
          max_paths=max_paths):
    visited_nodes.append(current)
    current_path.append(current)
    current_path_lengths.append(previous_distance)
    # print(current, target, len(visited_nodes),
    #       len(current_path), len(all_paths), max_paths)

    if current == target:
      # print("Reached target with path")
      # print(current_path)
      all_paths.append(deepcopy(current_path))
      all_path_lengths.append(deepcopy(current_path_lengths))

    if len(all_paths) >= max_paths:
      # print("Reached threshold for max paths!")
      return all_paths, all_path_lengths

    # if len(current_path) - 1 > cutoff:
    #   # print("Reached cutoff with")
    #   # print(current_path)
    #   return all_paths

    for neighbour in G.neighbors(current):
      # If previous distance was zero, omit all neighbours with zero cost
      neighbour_distance = G[current][neighbour][0]['weight']
      # print(f'Checking {current} - {neighbour}')
      # print(previous_distance, neighbour_distance)
      if previous_distance == 0 and neighbour_distance == 0:
        # print(f'Skipping {current} - {neighbour}')
        continue
      # No loops!
      if not (neighbour in visited_nodes):
        all_paths, all_path_lengths = dfs(neighbour, target, visited_nodes,
                                          current_path, current_path_lengths,
                                          all_paths, all_path_lengths,
                                          neighbour_distance, cutoff=cutoff, max_paths=max_paths)

    visited_nodes.pop()
    current_path.pop()
    current_path_lengths.pop()

    return all_paths, all_path_lengths

  visited_nodes, current_path, current_path_lengths, all_paths, all_path_lengths\
      = [], [], [], [], []
  return dfs(source, target, visited_nodes, current_path, current_path_lengths,
             all_paths, all_path_lengths, previous_distance=None, cutoff=cutoff,
             max_paths=max_paths)


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


def compute_point_paths(network: Network, sources: List, targets: List,
                        density: float, temperature: float,
                        timescale: float,
                        cutoff=5, max_paths=100):
  from gcrn.helper_functions import count_all_rxns, count_rxns_by_pairs
  # Uses the number densities stored in 'network'!
  paths, lengths, rxn_paths = find_network_paths(network, sources, targets,
                                                 cutoff=cutoff,
                                                 max_paths=max_paths)
  rxn_idx_paths = rxn_idx_paths_from_rxn_paths(rxn_paths)
  all_rxn_counts = count_all_rxns(rxn_idx_paths)
  counts_by_pairs = count_rxns_by_pairs(rxn_idx_paths)

  return PointPaths(density, temperature, timescale, deepcopy(paths),
                    deepcopy(lengths), deepcopy(rxn_idx_paths),
                    deepcopy(all_rxn_counts), deepcopy(counts_by_pairs))


# ==============================================================================
# Utility functions for pathfinding, including string manipulation methods
# ==============================================================================#


def rxn_path_from_path(path: List[str], network: Network) -> List:
  from gcrn.helper_functions import reaction_from_complex
  # Find the reaction associated with path segments and reformulate the path
  # 'A -> B -> C' as 'A -> r1 -> r2 -> D', with 'A', 'C' as source, target
  # terms, respectively, and reactions 'r1', 'r2' representing 'A -> B' and
  # 'B -> C', respectively
  rxn_path = []
  had_prev_reaction = False
  for i, entry in enumerate(path):
    is_first_reaction = i == 0
    if i == len(path) - 1:
      continue
    rxn = reaction_from_complex(entry, path[i+1], network)
    if rxn:
      # Add source species in case of single-substrate reaction
      if is_first_reaction:
        rxn_path.append(path[0])
      rxn_path.append(rxn)
      had_prev_reaction = True
    else:
      if not had_prev_reaction:
        rxn_path.append(entry)
      had_prev_reaction = False
  # Add final (target) species
  rxn_path.append(path[-1])

  return rxn_path


def rxn_idx_path_from_rxn_path(rxn_path: List[Reaction]) -> str:
  # Replace the 'Reaction' objects in path with their indices if available
  return " -> ".join([f"{e.idx}" if isinstance(e, Reaction)
                      else f"{str(e)}" for e in rxn_path])


def rxn_idx_paths_from_rxn_paths(rxn_paths: Dict) -> Dict:
  # From a dictionary of paths with keys (traditionally) "source-target",
  # evaluate each Reaction as its index if available
  rxn_idx_paths = {}
  for key in rxn_paths.keys():
    rxn_idx_paths[key] = []
    for rxn_path in rxn_paths[key]:
      # Cast elements of rxn_path to str
      rxn_idx_path = " -> ".join([f"{e.idx}" if isinstance(e, Reaction)
                                  else f"{str(e)}" for e in rxn_path])
      rxn_idx_paths[key].append(rxn_idx_path)

  return rxn_idx_paths


def species_in_path(path: List[str]) -> List[str]:
  species = map(str.strip, path.split("->"))
  return species


def species_in_rxn_idx_path(rxn_idx_path: List[str]) -> List[str]:
  species = rxn_idx_path[0]
  rxn_species = re.findall(r"(?<=-> )\d+(?= ->)", rxn_idx_path)
  species.append(*rxn_species)
  species.append(rxn_idx_path)
  return species
