from typing import List
from src.reaction import Reaction
import networkx as nx
import numpy as np


class Network:
  def __init__(self, reactions: List[Reaction]) -> None:
    self.reactions = sorted(reactions, key=lambda rxn: rxn.idx)

    # Potential space saving with clever list comprehensions?
    # Doing species and complexes in this way is redundant, should do complexes
    # first and then reduce that to get the species
    species = []
    for rxn in self.reactions:
      species.extend(rxn.reactants + rxn.products)
    self.species = sorted(list(set(species)))

    # TODO: Additional constraint: combinations, not permutations!
    # e.g. H + H + H2 == H2 + H + H
    # Also need to check this when creating the graph!
    complexes = []
    for rxn in self.reactions:
      complexes.append(rxn.reactant_complex)
      complexes.append(rxn.product_complex)
    self.complexes = sorted(list(set(complexes)))

    # Create MultiDiGraphs for species and complexes using networkx
    self.species_graph = self.create_species_graph()
    self.complex_graph = self.create_complex_graph()

    # Create incidence matrices from graphs
    self.species_incidence_matrix = nx.incidence_matrix(self.species_graph)
    # nx incidence matrix does not give us what we want
    # self.complex_incidence_matrix = nx.incidence_matrix(self.complex_graph)
    self.complex_incidence_matrix = self.create_complex_incidence_matrix()

    # Complex composition matrix (m x c)
    self.complex_composition_matrix = self.create_complex_composition_matrix()

    # Stoichiometric matrix (m x r)
    self.stoichiometric_matrix = self.complex_composition_matrix @ self.complex_incidence_matrix

    # Outgoing co-incidence matrix of reaction rates (r x c)
    self.kinetics_matrix = self.create_kinetics_matrix()

    # Weighted Laplacian (transpose of conventional Laplacian) matrix (c x c)
    self.laplacian_matrix = self.create_laplacian_matrix()

    # Potential graphs from defined incidence matrices
    # self.complex_composition_graph = nx.from_numpy_matrix(
    #     incidence_to_adjacency(self.complex_composition_matrix))
    # self.complex_composition_graph = self.create_complex_composition_graph()

  @classmethod
  def from_krome_file(cls, krome_file: str):
    # Initialise the Network from a valid KROME network file
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
    with open(krome_file, 'r', encoding='utf-8') as infile:
      while True:
        line = infile.readline().strip()
        # TODO: Fix this condition, since if empty lines are present in a file,
        # it stops reading (happens often when adding comments)
        if not line:
          break

        if line.startswith("#"):
          continue

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

    return cls(reactions)

  # ----------------------------------------------------------------------------
  # Methods for creating graphs
  # ----------------------------------------------------------------------------

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

  def create_complex_composition_graph(self) -> nx.DiGraph:
    # From the complex composition matrix, create a Directed Graph
    # (species -> complex) for each species/complex in the network
    complex_composition_graph = nx.DiGraph()
    for i, species in enumerate(self.species):
      for j, complex in enumerate(self.complexes):
        if self.complex_composition_matrix[i, j] == 1:
          complex_composition_graph.add_edge(species, complex)

    return complex_composition_graph

  # ----------------------------------------------------------------------------
  # Methods for creating matrices
  # ----------------------------------------------------------------------------
  def create_complex_incidence_matrix(self) -> np.ndarray:
    # Create (c x r) incidence matrix linking complexes to reactions
    complex_incidence_matrix = np.zeros((len(self.complexes),
                                         len(self.reactions)))

    for i, complex in enumerate(self.complexes):
      for j, rxn in enumerate(self.reactions):
        if complex == rxn.reactant_complex:
          complex_incidence_matrix[i, j] = -1
        if complex == rxn.product_complex:
          complex_incidence_matrix[i, j] = 1

    return complex_incidence_matrix

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

  def create_kinetics_matrix(self, temperature=300) -> np.ndarray:
    # Create the (r x c) coindicidence matrix containing reaction rate constants
    kinetics_matrix = np.zeros((len(self.reactions), len(self.complexes)))

    for i, rxn in enumerate(self.reactions):
      for j, complex in enumerate(self.complexes):
        if complex == rxn.reactant_complex:
          kinetics_matrix[i, j] = rxn.evaluate_rate_expression(temperature)
        else:
          kinetics_matrix[i, j] = 0

    return kinetics_matrix

  def create_laplacian_matrix(self) -> np.ndarray:
    # Create (c x c) Laplacian matrix L := -DK
    D, K = self.complex_incidence_matrix, self.kinetics_matrix
    laplacian_matrix = -D @ K
    return laplacian_matrix

  # ----------------------------------------------------------------------------
  # Methods for updating graphs
  # ----------------------------------------------------------------------------

  def update_species_graph(self, temperature: float):
    # Given a certain temperature, update the species Graph (weights)
    self.species_graph = self.create_species_graph(temperature=temperature)

  def update_complex_graph(self, temperature: float):
    # Given a certain temperature, update the complex Graph (weights)
    self.complex_graph = self.create_complex_graph(temperature=temperature)
