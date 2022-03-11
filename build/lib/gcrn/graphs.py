from gcrn.network import Network
import networkx as nx
from itertools import product


# -------------------------------------------------------------------------
# Chemical graph creation
# -------------------------------------------------------------------------
def create_species_graph(network: Network) -> nx.MultiDiGraph:
  # Create graph of species
  species_graph = nx.MultiDiGraph()
  for rxn in network.reactions:
    for r, p in product(rxn.reactants, rxn.products):
      # Weight is timescale (inverse rate)
      weight = 1 / rxn.evaluate_mass_action_rate(network.temperature,
                                                 network.number_densities_dict)
      species_graph.add_edge(r, p, weight=weight)

  return species_graph


def create_complex_graph(network: Network) -> nx.MultiDiGraph:
  # Create graph of complexes
  complex_graph = nx.MultiDiGraph()
  for rxn in network.reactions:
    # Weight is timescale (inverse rate)
    weight = 1 / rxn.evaluate_mass_action_rate(network.temperature,
                                               network.number_densities_dict)
    complex_graph.add_edge(rxn.reactant_complex,
                           rxn.product_complex, weight=weight)

  return complex_graph


def create_complex_composition_graph(network: Network) -> nx.DiGraph:
  # From the complex composition matrix, create a Directed Graph
  # (species -> complex) for each species/complex in the network
  complex_composition_graph = nx.DiGraph()
  for i, species in enumerate(network.species):
    for j, complex in enumerate(network.complexes):
      if network.complex_composition_matrix[i, j] == 1:
        complex_composition_graph.add_edge(species, complex)

  return complex_composition_graph
