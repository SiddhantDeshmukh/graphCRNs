# First we have to write something to read in reactions
# Need some kind of 'intelligent' reader, the idea is to use KROME format so we
# read to find the 'format', then the lines after that have to follow this
# format. Need a Reaction class and a Network class for sure
import numpy as np
from networkx.drawing.nx_pydot import to_pydot
from typing import List
import networkx as nx
from itertools import product
from math import exp  # used in 'eval'


# Order list alphabetically so that complex combinations work out
# e.g. H + CH == CH + H
# TODO: Do it properly when creating the set of complexes!
def create_complex(lst): return ' + '.join(sorted(lst))


class Reaction:
  def __init__(self, reactants: List, products: List, rate_expression: str,
               idx=None) -> None:
    self.reactants = reactants
    self.products = products

    # Remove empty reactants and products
    self.reactants = [reactant for reactant in self.reactants if reactant]
    self.products = [product for product in self.products if product]

    self.reactant_complex = create_complex(self.reactants)
    self.product_complex = create_complex(self.products)

    # Need to evaluate this rate expression for temperature, so keep it as
    # something we can 'eval'
    self.rate_expression = rate_expression
    # Fortran -> Python format
    self.rate_expression = rate_expression.replace('d', 'e')

    self.rate = self.evaluate_rate_expression(300)  # default

    if idx:
      self.idx = idx

  def __str__(self) -> str:
    output = f"{create_complex(self.reactants)} -> {create_complex(self.products)}"
    output += f"\tRate: {self.rate_expression}"
    if self.idx:
      output += f"\tIndex = {self.idx}"

    return output

  def evaluate_rate_expression(self, temperature=None):
    # Evaluate (potentially temperature-dependent) rate expression
    # WARNING: Eval is evil!
    expression = self.rate_expression.replace("Tgas", str(temperature))
    rate = eval(expression)
    return rate


class Network:
  def __init__(self, reactions: List[Reaction]) -> None:
    self.reactions = reactions

    # Potential space saving with clever list comprehensions?
    # Doing species and complexes in this way is redundant, should do complexes
    # first and then reduce that to get the species
    species = []
    for rxn in reactions:
      species.extend(rxn.reactants + rxn.products)
    self.species = list(set(species))

    # TODO: Additional constraint: combinations, not permutations!
    # e.g. H + H + H2 == H2 + H + H
    # Also need to check this when creating the graph!
    complexes = []
    for rxn in reactions:
      complexes.append(rxn.reactant_complex)
      complexes.append(rxn.product_complex)
    self.complexes = list(set(complexes))

    # Create MultiDiGraphs for species and complexes using networkx
    self.species_graph = self.create_species_graph()
    self.complex_graph = self.create_complex_graph()

    # Create incidence matrices from graphs
    self.species_incidence_matrix = nx.incidence_matrix(self.species_graph)
    self.complex_incidence_matrix = nx.incidence_matrix(self.complex_graph)

    # Complex composition matrix (m x c)
    self.complex_composition_matrix = self.create_complex_composition_matrix()

    # Stoichiometric matrix (m x r)
    self.stoichiometric_matrix = self.complex_composition_matrix @ self.complex_incidence_matrix

    # Potential graphs from defined incidence matrices
    # self.complex_composition_graph = nx.from_numpy_matrix(
    #     incidence_to_adjacency(self.complex_composition_matrix))
    # self.complex_composition_graph = self.create_complex_composition_graph()

  def create_species_graph(self, temperature=None) -> nx.MultiDiGraph:
    # Create graph of species
    species_graph = nx.MultiDiGraph()
    for rxn in self.reactions:
      # TODO: Optimise with itertools.product()
      for r in rxn.reactants:
        for p in rxn.products:
          weight = rxn.evaluate_rate_expression(temperature) \
              if temperature else rxn.rate
          species_graph.add_edge(r, p, weight=weight)

    return species_graph

  def create_complex_graph(self, temperature=None) -> nx.MultiDiGraph:
    # Create graph of complexes
    complex_graph = nx.MultiDiGraph()
    for rxn in self.reactions:
      weight = rxn.evaluate_rate_expression(temperature) \
          if temperature else rxn.rate
      complex_graph.add_edge(rxn.reactant_complex,
                             rxn.product_complex, weight=weight)

    return complex_graph

  def create_complex_composition_matrix(self) -> np.ndarray:
    # Create matrix describing the composition of each complex 'c' based on its
    # constituent metabolites 'm' (m x c)
    complex_composition_matrix = np.zeros((len(self.species),
                                           len(self.complexes)))
    for i, species in enumerate(self.species):
      for j, complex in enumerate(self.complexes):
        split_complex = [c.strip() for c in complex.split("+")]
        if species in split_complex:
          complex_composition_matrix[i, j] = 1

    return complex_composition_matrix

  def create_complex_composition_graph(self) -> nx.DiGraph:
    # From the complex composition matrix, create a Directed Graph
    # (species -> complex) for each species/complex in the network
    complex_composition_graph = nx.DiGraph()
    for i, species in enumerate(self.species):
      for j, complex in enumerate(self.complexes):
        if self.complex_composition_matrix[i, j] == 1:
          complex_composition_graph.add_edge(species, complex)

    return complex_composition_graph

  def update_species_graph(self, temperature: float):
    # Given a certain temperature, update the species Graph (weights)
    self.species_graph = self.create_species_graph(temperature=temperature)

  def update_complex_graph(self, temperature: float):
    # Given a certain temperature, update the complex Graph (weights)
    self.complex_graph = self.create_complex_graph(temperature=temperature)


# Move to a different utils script for graphs
def incidence_to_adjacency(incidence_matrix: np.ndarray,
                           remove_self_loops=False) -> np.ndarray:
  # From an incidence matrix, create and return the respective adjacency matrix
  # Useful since networkx graphs can be created with adjacency matrices,
  # but not incidence matrices
  adjacency_matrix = (np.dot(incidence_matrix,
                             incidence_matrix.T) > 0).astype(int)
  if remove_self_loops:
    np.fill_diagonal(adjacency_matrix, 0)

  return adjacency_matrix


# cls method for Network!
def read_krome_file(filepath: str) -> Network:
  # Reads in a KROME rxn network as a Network
  rxn_format = None

  # Store all as list in case there are duplicates; should only have one
  # 'rxn_idx' and 'rate_expression', so we just pass in the first index of the
  # list as Reaction init
  format_dict = {
      'idx': [],
      'R': [],
      'P': [],
      'rate': []
  }

  reactions = []
  with open(filepath, 'r', encoding='utf-8') as infile:
    while True:
      line = infile.readline().strip()
      if not line:
        break

      # Check for 'format'
      if line.startswith('@format:'):
        # Reaction usually made up of 'idx', 'R', 'P', 'rate'
        rxn_format = line.replace('@format:', '').split(',')

      else:
        split_line = line.split(',')
        for i, item in enumerate(rxn_format):
          format_dict[item].append(split_line[i])

        reactions.append(Reaction(format_dict['R'], format_dict['P'],
                                  format_dict['rate'][0],
                                  idx=format_dict['idx'][0]))

      # Reset quantities
      for key in format_dict.keys():
        format_dict[key] = []

  return Network(reactions)


if __name__ == "__main__":
  krome_file = '../res/react-co-solar-umist12'
  network = read_krome_file(krome_file)

  print(f"{len(network.species)} Species:")
  print(network.species)
  print(f"{len(network.complexes)} Complexes:")
  print(network.complexes)
  # print("Rates")
  # for rxn in network.reactions:
  #   print(rxn.rate)

  to_pydot(network.species_graph).write_png("./species.png")
  to_pydot(network.complex_graph).write_png("./complex.png")

  # Check matrices
  print("Adjacency matrices")
  print(nx.to_numpy_array(network.species_graph).shape)
  print(nx.to_numpy_array(network.complex_graph).shape)
  print(incidence_to_adjacency(network.complex_composition_matrix).shape)

  print("Incidence matrices")
  print(network.species_incidence_matrix.shape)
  print(network.complex_incidence_matrix.shape)
  print(network.complex_composition_matrix.shape)

  print("Stoichiometric matrix")
  print(network.stoichiometric_matrix.shape)
