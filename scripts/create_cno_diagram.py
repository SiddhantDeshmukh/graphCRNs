from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from gcrn.network import Network
from gcrn.graphs import create_species_graph
import re
from collections import Counter


# ==============================================================================
# https://codereview.stackexchange.com/questions/232630/parsing-molecular-formula/232671#232671
def parse_molecule(molecule):
  array = [[]]

  for token in molecule:
    if token.isalpha() and token.istitle():
      last = [token]
      upper = token
      array[-1].append(token)
    elif token.isalpha():
      last = upper + token
      array[-1] = [last]
    elif token.isdigit():
      array[-1].extend(last*(int(token)-1))
    elif token == '(' or token == '[':
      array.append([])
    elif token == ')' or token == ']':
      last = array.pop()
      array[-1].extend(last)

  return dict(Counter(array[-1]))


def tokenize_molecule(molecule):
  return re.findall('[A-Z][a-z]?|\d+|.', molecule)

# ==============================================================================


def shares_keys(d1: Dict, d2: Dict):
  # Return 'True' if any keys in 'd1' match those in 'd2' (one-way dumb check)
  # and 'False' otherwise
  return bool(sum([k1 in d2.keys() for k1 in d1.keys()]))


def main():
  network = Network.from_krome_file("../res/cno.ntw")
  species = network.species

  # For plotting, create a smaller graph with single reaction pathways for
  # formation/dissociation
  # Make an edge if any elements between 'u' & 'v' match, e.g.
  # C <-> CO, CH <-> OH, C <-/-> O, O <-/ H->
  deconstructed_species = {s: parse_molecule(s) for s in species}
  edges = []
  for s1, m1 in deconstructed_species.items():
    for s2, m2 in deconstructed_species.items():
      # No self-loops
      if s1 == s2:
        continue
      # Make a connection between 's1' and 's2' if any of the keys
      # between 'm1' and 'm2' match
      if shares_keys(m1, m2):
        edge = (s1, s2, 1)
        if not edge in edges:
          edges.append(edge)

  G = nx.MultiDiGraph()
  G.add_edges_from(edges)
  fig, ax = plt.subplots()
  # Layout
  options = {
      # "node_color": "orange",
      "node_size": 500,
      # "with_labels": True,
      # 'font_size': 11,
      'linewidths': 1,
      "edgecolors": "black",
      # 'width': 1
  }
  pos = nx.shell_layout(G)
  # General edges and labels
  nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=1)
  nx.draw_networkx_labels(G, pos, ax=ax)
  # Atoms
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=["H", "C", "N", "O"],
                         node_color="tomato", **options)
  # Important Molecules
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=["C2", "OH", "CO", "CH", "CN"],
                         node_color="tab:cyan", **options)
  # Other Molecules
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=["H2", "N2", "O2", "NH", "NO"],
                         node_color="lightgrey", **options)
  plt.tight_layout()
  plt.savefig("../out/figs/cno_diagram.png")
  plt.show()


if __name__ == "__main__":
  main()
