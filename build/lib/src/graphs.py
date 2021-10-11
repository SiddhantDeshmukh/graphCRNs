from src.reaction import Reaction
from typing import List
import networkx as nx
import itertools


# -------------------------------------------------------------------------
# Chemical graph creation
# -------------------------------------------------------------------------
def species_graph(reactions: List[Reaction]) -> nx.MultiDiGraph:
  # Species (vertices) to reactions (edges)
  species_graph = nx.MultiDiGraph()
  for rxn in reactions:
    for r, p in itertools.product(rxn.reactants, rxn.products):
      species_graph.add_edge(r, p, weight=rxn.rate)

  return species_graph


def complex_graph(reactions: List[Reaction]) -> nx.MultiDiGraph:
  # Complexes (vertices) to reactions (edges)
  complex_graph = nx.MultiDiGraph()
  for rxn in reactions:
    complex_graph.add_edge(rxn.reactant_complex, rxn.product_complex,
                           weight=rxn.rate)

  return complex_graph
