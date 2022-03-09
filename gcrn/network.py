import numpy as np
import networkx as nx
from gcrn.reaction import Reaction
from typing import Dict, List, Union
from itertools import product
import fortranformat as ff
from gcrn.utilities import cofactor_matrix, group_rxns, list_to_krome_format,\
    constants_from_rate, pad_list, to_fortran_str
from datetime import datetime
from tabulate import tabulate


class Network:
  def __init__(self, reactions: List[Reaction], temperature=300,
               number_densities=None) -> None:
    self.reactions: List[Reaction] = sorted(reactions, key=lambda rxn: rxn.idx)
    self.find_duplicate_indices()

    # Determine species and complexes present in each reaction
    species: List[str] = []
    complexes: List[str] = []
    for rxn in self.reactions:
      species.extend(rxn.reactants + rxn.products)
      complexes.append(rxn.reactant_complex)
      complexes.append(rxn.product_complex)

    self.species: List[str] = sorted(list(set(species)))
    self.complexes: List[str] = sorted(list(set(complexes)))

    # Properties
    self._temperature = temperature
    if not number_densities:  # populate with random values
      number_densities = {s: np.random.randint(1, 100) for s in species}
    self._number_densities: Dict = number_densities

    # TODO: Additional constraint: combinations, not permutations!
    # e.g. H + H + H2 == H2 + H + H
    # Also need to check this when creating the graph!
    # self.complexes = sorted(list(set(chain.from_iterable(
    #     [[rxn.reactant_complex, rxn.product_complex]
    #      for rxn in self.reactions]))))

    # Create MultiDiGraphs for species and complexes using networkx
    self._species_graph = self.create_species_graph()
    self._complex_graph = self.create_complex_graph()

    # Incidence matrices from graphs
    # Species incidence matrix (m x r)
    self.species_incidence_matrix = nx.incidence_matrix(self._species_graph)
    # Complex incidence matrix (c x r)
    self.complex_incidence_matrix = self.create_complex_incidence_matrix()
    # Complex composition matrix (m x c)
    self.complex_composition_matrix = self.create_complex_composition_matrix()
    # Stoichiometric matrix (m x r)
    self.stoichiometric_matrix = self.complex_composition_matrix @ self.complex_incidence_matrix
    # Outgoing co-incidence matrix of reaction rates (r x c)
    self.eval_kinetics_matrix, self.eval_kinetics_idxs = self.create_eval_kinetics_matrix()
    self.complex_kinetics_matrix = self.create_complex_kinetics_matrix()
    # Weighted Laplacian (transpose of conventional Laplacian) matrix (c x c)
    self.complex_laplacian = self.create_complex_laplacian_matrix()

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
        # limits are one of 'none', 'sharp', 'weak', 'medium', 'strong'
        'limit': [],
        # TODO:
        # Use accuracy to determine limit strength, add 'sigmoid' limit
        'accuracy': [],  # A, B, C, D, E (see UMIST12 nomenclature)
        'ref': []  # reference to rate
        # TODO:
        # Add implementation to read files that don't have all keys present
    }

    single_entry_keys = ['idx', 'rate', 'Tmin', 'Tmax',
                         'limit', 'accuracy', 'ref']

    reactions = []
    with open(krome_file, 'r', encoding='utf-8') as infile:
      while True:
        line = infile.readline().strip()
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
          # print(split_line, rxn_format)
          for i, item in enumerate(rxn_format):
            format_dict[item].append(split_line[i])

          # Clean up dictionary to pass to Reaction
          for key in format_dict.keys():
            # Empty lists are 'None'
            if not format_dict[key]:
              format_dict[key] = None
            # Lists that should be single entries are unpacked
            if key in single_entry_keys and format_dict[key]:
              format_dict[key] = format_dict[key][0]

          reactions.append(Reaction(format_dict['R'], format_dict['P'],
                                    format_dict['rate'],
                                    idx=format_dict['idx'],
                                    min_temperature=format_dict['Tmin'],
                                    max_temperature=format_dict['Tmax'],
                                    limit=format_dict['limit'],
                                    reference=format_dict['ref']))

        # Reset quantities
        for key in format_dict.keys():
          format_dict[key] = []

    return cls(reactions)

  @classmethod
  def from_cobold_file(cls, cobold_file: str):
    # Initialise Network object from Fortran fixed format cobold 'chem.dat' file
    fformat = '(I4,5(A8,1X),2(1X,A4),1X,1PE8.2,3X,0PF5.2,2X,0PF8.1,A16)'
    reader = ff.FortranRecordReader(fformat)
    reactions = []
    with open(cobold_file, 'r') as infile:
      while True:
        line = infile.readline()
        if not line:
          break
        # Each line is a separate reaction of form:
        # idx R R R P P P P alpha beta gamma ref
        idx, R1, R2, R3, P1, P2, P3, P4, alpha, beta, gamma, ref =\
            [str(s).strip() for s in reader.read(line)]
        alpha, beta, gamma = float(alpha), float(beta), float(gamma)
        reactants = [r for r in [R1, R2, R3] if r]
        products = [p for p in [P1, P2, P3, P4] if p]
        rate_expression = ""
        if alpha:
          rate_expression += f'{to_fortran_str(float(alpha), fmt="1.4e")}'
        if beta:
          rate_expression += f' * (Tgas / 3d2)**({to_fortran_str(float(beta), fmt="1.4e")})'
        if gamma:
          if gamma > 0:
            gamma_str = f'-{to_fortran_str(float(gamma), fmt="1.4e")}'
          else:
            gamma_str = f'{to_fortran_str(-float(gamma), fmt="1.4e")}'
          rate_expression += f' * exp({gamma_str} / Tgas)'
        reactions.append(Reaction(reactants, products, rate_expression, idx,
                                  reference=ref))

    return cls(reactions)

  # ----------------------------------------------------------------------------
  # Output
  # ----------------------------------------------------------------------------

  def to_krome_format(self, path: str):
    # Write the Network to KROME-readable format
    # Determine reaction formats
    # 1. Group based on if temperature limits are present
    limit_idxs = [i for i, rxn in enumerate(self.reactions)
                  if rxn.min_temperature or rxn.max_temperature]

    limit_reactions = [self.reactions[i] for i in limit_idxs]
    unlimit_reactions = [self.reactions[i] for i in range(len(self.reactions))
                         if not i in limit_idxs]

    # 2. Group based on number of reactants / products
    header = "#" * 80
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = f"{header}\n# Automatically generated on {timestamp}\n"
    output += f'# {len(self.species)} species, {len(self.reactions)} reactions\n'
    output += f'# of which {len(limit_reactions)} have temperature limits\n'
    output += f'{header}\n'
    if limit_reactions:
      output += "# Reactions with temperature limits\n"
      output += f"{list_to_krome_format(limit_reactions)}\n"
    if unlimit_reactions:
      output += "# Reactions without temperature limits\n"
      output += f"{list_to_krome_format(unlimit_reactions)}\n"

    with open(path, 'w', encoding='utf-8') as outfile:
      outfile.write(output.rstrip())

  def to_cobold_format(self, path: str, with_limits=False):
    # Write Network to CO5BOLD 'chem.dat' format (provided a FORTRAN format)
    # NOTE:
    # CHEM_FORMAT differs slightly from CO5BOLD provided format to allow for
    # left-justification of strings and to then follow spacing convention
    CHEM_FORMAT = "(I4,1X, 5(A8, 1X), 2(A4,1X), 1PE8.2, 3X, 0PF5.2, 2X, 0PF8.1, A16)"
    CHEM_LIMIT_FORMAT = "(I4,5(A8,1X),2(1X,A4),1X,1PE8.2,3X,0PF5.2,2X,0PF8.1,2(1X,1PE8.2),A16)"
    output = ""
    fformat = CHEM_LIMIT_FORMAT if with_limits else CHEM_FORMAT
    writer = ff.FortranRecordWriter(fformat)
    for rxn in self.reactions:
      reactants = pad_list(rxn.reactants, 3)
      products = pad_list(rxn.products, 4)
      # Left-align by padding to 8 chars for reactants and first 2 products,
      # and to 4 chars for last 2 products
      # For some reason the list comprehension gets rid of the padded elements
      for i, r in enumerate(reactants):
        reactants[i] = r.ljust(8, ' ')
      for i, p in enumerate(products):
        width = 8 if i <= 1 else 4
        products[i] = p.ljust(width, ' ')

      alpha, beta, gamma = constants_from_rate(rxn.rate_expression)
      data = [int(rxn.idx), *reactants, *products,
              alpha, beta, gamma]
      if with_limits:
        data.append(float(rxn.min_temperature))
        data.append(float(rxn.max_temperature))

      data.append(rxn.reference.ljust(16, ' '))
      output += writer.write(data) + "\n"

    with open(path, 'w') as outfile:
      outfile.write(output)

  def to_latex_table(self, path: str):
    # Write the network to a LaTeX table
    format_dict = group_rxns(self.reactions)
    for i, (format_str, rxn_strs) in enumerate(format_dict.items()):
      num_reactants = list(format_str).count('R')
      num_products = list(format_str).count('P')
      subtitle = f'{num_reactants} reactants, {num_products} products'
      print(i, format_str)
      print(rxn_strs)
      print('==')

  # ----------------------------------------------------------------------------
  # String methods
  # ----------------------------------------------------------------------------

  def __str__(self) -> str:
    return "\n".join([str(rxn) for rxn in self.reactions])

  def description(self) -> str:
    output = f"{len(self.reactions)} reactions with {len(self.species)} species.\n"
    output += "\n".join([rxn.description() for rxn in self.reactions])

    return output

  # ----------------------------------------------------------------------------
  # Verification methods
  # ----------------------------------------------------------------------------

  def find_duplicate_indices(self):
    # Check if mulitple reactions have the same index
    duplicates = []
    added_idxs = []
    for i, rxn1 in enumerate(self.reactions):
      for j, rxn2 in enumerate(self.reactions):
        if i == j:
          continue
        if rxn1.idx == rxn2.idx:
          if not rxn1.idx in added_idxs:
            added_idxs.append(rxn1.idx)
            duplicates.append((rxn1, rxn2))

    if duplicates:
      print(
          f'Warning! {len(duplicates)} reactions with duplicate indices found:')
      print(f'==================================================')
      for duplicate in duplicates:
        print('\n'.join([d.description() for d in duplicate]))
        print(f'==================================================')

  # ----------------------------------------------------------------------------
  # Methods for creating graphs
  # ----------------------------------------------------------------------------
  # TODO:
  # Move graph creation into a generic functional script
  def create_species_graph(self) -> nx.MultiDiGraph:
    # Create graph of species
    species_graph = nx.MultiDiGraph()
    for rxn in self.reactions:
      for r, p in product(rxn.reactants, rxn.products):
        # Weight is timescale (inverse rate)
        weight = 1 / rxn.evaluate_mass_action_rate(self.temperature,
                                                   self.number_densities)
        species_graph.add_edge(r, p, weight=weight)

    return species_graph

  def create_complex_graph(self) -> nx.MultiDiGraph:
    # Create graph of complexes
    complex_graph = nx.MultiDiGraph()
    for rxn in self.reactions:
      # Weight is timescale (inverse rate)
      weight = 1 / rxn.evaluate_mass_action_rate(self.temperature,
                                                 self.number_densities)
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
        complex_composition_matrix[i, j] = split_complex.count(species)

    return complex_composition_matrix

  def create_eval_kinetics_matrix(self) -> np.ndarray:
    # Create the (r x c) coindicidence matrix containing reaction objects
    # 'K' in: x_dot =  ZDK Exp(Z.T Ln(x))
    eval_kinetics_matrix = np.zeros((len(self.reactions), len(self.complexes)),
                                    dtype=object)
    eval_kinetics_idxs = []
    # TODO:
    # Use a sparse array since most entries will be zero, and never evaluate
    # the zero entries
    for i, rxn in enumerate(self.reactions):
      for j, complex in enumerate(self.complexes):
        # rhs = rxn if complex == rxn.reactant_complex else eval('lambda _: 0')
        # eval_kinetics_matrix[i, j] = rhs
        # eval_kinetics_idxs.append((i, j))
        if complex == rxn.reactant_complex:
          eval_kinetics_matrix[i, j] = rxn
          eval_kinetics_idxs.append((i, j))
        else:
          eval_kinetics_matrix[i, j] = 0.

    # print(f"{len(eval_kinetics_idxs)} index pairs for {eval_kinetics_matrix.shape} matrix.")
    return eval_kinetics_matrix, eval_kinetics_idxs

  def create_complex_kinetics_matrix(self, limit_rates=False) -> np.ndarray:
    # Create the (r x c) coindicidence matrix containing reaction rate constants
    # 'K' in: x_dot =  ZDK Exp(Z.T Ln(x))
    kinetics_matrix = np.zeros((len(self.reactions), len(self.complexes)))
    for (i, j) in self.eval_kinetics_idxs:
      kinetics_matrix[i, j] = self.eval_kinetics_matrix[i, j](self.temperature)
    return kinetics_matrix

  def create_complex_laplacian_matrix(self) -> np.ndarray:
    # Create (c x c) Laplacian matrix L := -DK
    D, K = self.complex_incidence_matrix, self.complex_kinetics_matrix
    laplacian_matrix = -D @ K
    return laplacian_matrix

  # ----------------------------------------------------------------------------
  # Methods for updating graphs
  # ----------------------------------------------------------------------------

  def update_species_graph(self):
    # Given a certain temperature, update the species Graph (weights)
    self.species_graph = self.create_species_graph()

  def update_complex_graph(self):
    # Given a certain temperature, update the complex Graph (weights)
    self.complex_graph = self.create_complex_graph()

  # ----------------------------------------------------------------------------
  # Methods for updating attributes
  # ----------------------------------------------------------------------------
  def _update(self):
    # Update complex kinetics matrix
    # NOTE: Graphs are NOT updated!
    self.complex_kinetics_matrix =\
        self.create_complex_kinetics_matrix()

  # ----------------------------------------------------------------------------
  # Methods for computing nullspaces
  # ----------------------------------------------------------------------------
  def compute_complex_balance(self) -> np.ndarray:
    # Using Kirchhoff's Matrix Tree theorem, compute the kernel of the Laplacian
    # corresponding to a positive, complex-balanced equilibrium for a given
    # temperature
    # Only need first row of cofactor matrix!
    C = cofactor_matrix(self.complex_laplacian)
    rho = C[0]
    return rho

  # ----------------------------------------------------------------------------
  # Methods for counting
  # ----------------------------------------------------------------------------
  def count_reactant_instances(self, species: str) -> int:
    # For a specified species, count the number of times it appears as a
    # reactant and return the count
    count = 0
    for rxn in self.reactions:
      if species in rxn.reactants:
        count += 1
    return count

  def count_product_instances(self, species: str) -> int:
    # For a specified species, count the number of times it appears as a
    # product and return the count
    count = 0
    for rxn in self.reactions:
      if species in rxn.products:
        count += 1
    return count

  def network_species_count(self) -> Dict:
    # Count the occurrence of reactant/product occurrences of species in
    # network of reactions. Only counts each species once per reaction,
    # e.g. H + H + H -> H2 yields 1 appearance of H on LHS and 1 appearance
    # of H2 on RHS.
    # TODO:
    # Isn't this just from the adjacency matrix? Try to get it from that
    # key (str, species) to List(int, int; reactant_count, product_count)
    counts = {}
    for s in self.species:
      reactant_count = self.count_reactant_instances(s)
      product_count = self.count_product_instances(s)
      counts[s] = {"R": reactant_count, "P": product_count}

    return counts

  # ----------------------------------------------------------------------------
  # Setters for reactions
  # ----------------------------------------------------------------------------
  def set_reaction_limit(self, limit: str):
    # Set the limit type for each reaction in the network
    for rxn in self.reactions:
      rxn.limit = limit

  # ----------------------------------------------------------------------------
  # Methods for properties
  # ----------------------------------------------------------------------------
  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value: float):
    self._temperature = value
    self._update()

  @property
  def number_densities(self):
    return self._number_densities

  @number_densities.setter
  def number_densities(self, value: Union[np.ndarray, Dict]):
    if isinstance(value, np.ndarray):
      # Create a dictionary, assuming same indexing of array as self.species
      self._number_densities = {s: n for s, n in zip(self.species, value)}
    elif isinstance(value, dict):
      self._number_densities = value
    self._update()

  @property
  def complex_graph(self):
    self._complex_graph = self.create_complex_graph()
    return self._complex_graph

  @complex_graph.setter
  def complex_graph(self, value: nx.Graph):
    self._complex_graph = value

  @property
  def species_graph(self):
    self._species_graph = self.create_species_graph()
    return self._complex_graph

  @complex_graph.setter
  def species_grah(self, value: nx.Graph):
    self._species_graph = value
