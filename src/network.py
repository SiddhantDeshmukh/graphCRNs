from src.utilities import cofactor_matrix, normalise_2d
from typing import Dict, List
from src.reaction import Reaction
import networkx as nx
import numpy as np
import sympy


# TODO:
# Clean-up temperature implementation, use it as a private attribute

class Network:
  def __init__(self, reactions: List[Reaction], temperature=300) -> None:
    self.reactions = sorted(reactions, key=lambda rxn: rxn.idx)

    # Potential space saving with clever list comprehensions?
    # Doing species and complexes in this way is redundant, should do complexes
    # first and then reduce that to get the species
    species = []
    for rxn in self.reactions:
      species.extend(rxn.reactants + rxn.products)
    self.species = sorted(list(set(species)))
    self.symbols = [f"n_{s}" for s in self.species]

    # Rates and Jacobian attributes
    self.rate_dict = self.create_rate_dict()
    self.jacobian_str = self.create_jacobian()

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
    self.complex_kinetics_matrix = self.create_complex_kinetics_matrix()

    # Outgoing co-incidence matrix of reaction rates (r x m)
    self.species_kinetics_matrix = self.create_species_kinetics_matrix()

    # Weighted Laplacian (transpose of conventional Laplacian) matrix (c x c)
    self.complex_laplacian = self.create_complex_laplacian_matrix()

    # Weighted Laplacian matrix (m x m)
    self.species_laplacian = self.create_species_laplacian_matrix()

    # Potential graphs from defined incidence matrices
    # self.complex_composition_graph = nx.from_numpy_matrix(
    #     incidence_to_adjacency(self.complex_composition_matrix))
    # self.complex_composition_graph = self.create_complex_composition_graph()

    # Properties
    self._temperature = temperature

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
        'rate': [],
        'Tmin': [],
        'Tmax': [],
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
                                    idx=format_dict['idx'][0],
                                    min_temperature=format_dict['Tmin'],
                                    max_temperature=format_dict['Tmax']))

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

  def create_complex_kinetics_matrix(self, temperature=300, normalise_kinetics=False) -> np.ndarray:
    # Create the (r x c) coindicidence matrix containing reaction rate constants
    # 'K' in: x_dot =  ZDK Exp(Z.T Ln(x))
    kinetics_matrix = np.zeros((len(self.reactions), len(self.complexes)))
    for i, rxn in enumerate(self.reactions):
      for j, complex in enumerate(self.complexes):
        if complex == rxn.reactant_complex:
          kinetics_matrix[i, j] = rxn.evaluate_rate_expression(temperature)

    # Normalise such that all rates are between 0, 1
    if normalise_kinetics:
      normalise_2d(kinetics_matrix)

    return kinetics_matrix

  def create_species_kinetics_matrix(self, temperature=300, normalise_kinetics=False) -> np.ndarray:
    # Create (r x m) coincidence matrix containing reaction rate constants
    # 'K' in: x_dot =  SK x
    kinetics_matrix = np.zeros((len(self.reactions), len(self.species)))
    for i, rxn in enumerate(self.reactions):
      for j, species in enumerate(self.species):
        if species in rxn.reactants:
          # TODO:
          # Should check if we've already been on the reactant side of a certain
          # reaction since we don't need to do the same reaction more than once
          kinetics_matrix[i, j] = rxn.evaluate_rate_expression(temperature)

    return kinetics_matrix

  def create_complex_laplacian_matrix(self) -> np.ndarray:
    # Create (c x c) Laplacian matrix L := -DK
    D, K = self.complex_incidence_matrix, self.complex_kinetics_matrix
    laplacian_matrix = -D @ K
    return laplacian_matrix

  def create_species_laplacian_matrix(self) -> np.ndarray:
    # Create (m x m) Laplacian matrix L := -SK
    S, K = self.stoichiometric_matrix, self.species_kinetics_matrix
    laplacian_matrix = -S @ K
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

  # ----------------------------------------------------------------------------
  # Methods for updating attributes
  # ----------------------------------------------------------------------------
  def update(self, temperature: float):
    # TODO:
    # Use temperature property instead of passing in

    # Given a certain temperature, update all matrices and graphs
    # that depend on it
    print(f"Updating matrices with temperature {temperature} K")
    self.complex_kinetics_matrix =\
        self.create_complex_kinetics_matrix(temperature)
    self.species_kinetics_matrix =\
        self.create_species_kinetics_matrix(temperature)

    print(f"Updating graphs with temperature {temperature} K")
    self.update_species_graph(temperature)
    self.update_complex_graph(temperature)

  # ----------------------------------------------------------------------------
  # Methods for computing nullspaces
  # ----------------------------------------------------------------------------
  def compute_complex_balance(self, temperature: float) -> np.ndarray:
    # Using Kirchhoff's Matrix Tree theorem, compute the kernel of the Laplacian
    # corresponding to a positive, complex-balanced equilibrium for a given
    # temperature
    # Only need first row of cofactor matrix!
    C = cofactor_matrix(self.complex_laplacian)
    rho = C[0]
    return rho

  def compute_species_balance(self, temperature: float) -> np.ndarray:
    # Using Kirchhoff's Matrix Tree theorem, compute the kernel of the Laplacian
    # corresponding to a positive, complex-balanced equilibrium for a given
    # temperature
    # Only need first row of cofactor matrix!
    C = cofactor_matrix(self.species_laplacian)
    rho = C[0]
    return rho

  # ----------------------------------------------------------------------------
  # Methods for Jacobian and rate analysis
  # ----------------------------------------------------------------------------
  def create_rate_dict(self) -> Dict:
    # Create the dictionary describing the time-dependent system of equations
    # d[X]/dt = x_1 + x_2 + ...
    rate_dict = {}  # keys are species
    for reaction in self.reactions:
      expression = reaction.mass_action_rate_expression
      reactant_symbols = [f"n_{key}" for key in reaction.stoichiometry[0].keys()]
      product_symbols = [f"n_{key}" for key in reaction.stoichiometry[1].keys()]
      for symbol in reactant_symbols:
        if symbol in rate_dict.keys():
          rate_dict[symbol].append(f"-{expression}")
        else:
          rate_dict[symbol] = [f"-{expression}"]

      for symbol in product_symbols:
        if symbol in rate_dict.keys():
          rate_dict[symbol].append(expression)
        else:
          rate_dict[symbol] = [expression]

    return rate_dict


  def create_jacobian(self) -> np.ndarray:
    # Create the analytical Jacobian matrix as a 2D array of strings
    # Uses sympy for the differential (which uses eval)
    num_species = len(self.species)
    jacobian = np.zeros((num_species, num_species), dtype=object)
    # Set up d[A]/dt = a_1 + a_2 + ... for each differential expression with A
    for i, rate in enumerate(self.rate_dict.values()):
      expression = " + ".join(rate)
      for j, symbol in enumerate(self.symbols):
        differential = sympy.diff(expression, symbol)
        jacobian[i, j] = str(differential)

    return jacobian

  # ----------------------------------------------------------------------------
  # Methods for properties
  # ----------------------------------------------------------------------------
  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value: float):
    self._temperature = value
    self.update(value)
