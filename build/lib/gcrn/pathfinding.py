from copy import deepcopy
from typing import Union, List, Dict
import networkx as nx
from gcrn.network import Network
from gcrn.helper_functions import find_reaction_from_complex


def create_search_graph(network: Network, source: str, target: str):
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

  G.add_weighted_edges_from(edges_to_add)

  return G


def find_paths(network: Network, source: str, target: str, cutoff=4,
               max_paths=100) -> Union[List, List]:
  # Find the 'num_paths' shortest paths between 'target' and 'source' of max
  # length 'cutoff'
  unique_paths = []
  unique_lengths = []
  reaction_paths = []

  search_graph = create_search_graph(network, source, target)
  edges, weights = zip(*nx.get_edge_attributes(search_graph, 'weight').items())

  # TODO:
  # Return the search
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
    for i, entry in enumerate(path):
      if i == len(path) - 1:
        continue
      rxn = find_reaction_from_complex(entry, path[i+1], network)
      if rxn:
        rxn_path.append(rxn)
        had_prev_reaction = True
      else:
        if not had_prev_reaction:
          rxn_path.append(entry)
        had_prev_reaction = False
    # Add final (target) species
    rxn_path.append(path[-1])

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
