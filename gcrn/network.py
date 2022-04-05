from __future__ import annotations
import numpy as np
import networkx as nx
from gcrn.reaction import Reaction
from typing import Callable, Dict, List, Union
from itertools import product
import fortranformat as ff
from gcrn.utilities import cofactor_matrix, group_rxns, list_to_krome_format,\
    constants_from_rate, pad_list, to_fortran_str
from datetime import datetime
from tabulate import tabulate
import re
import sympy
from math import exp  # used in 'eval'
from scipy.integrate import ode
from sadtools.utilities.abu_tools import load_abu


# REFACTOR TODO:
# - Add solver methods into Network and remove NetworkDynamics
# - Add method to initialise number densities from an abundance file
# - Add to_table() method
# - Move graph methods ot a separate script and make them functions(Network)
# - Add functions to compute timecsales from output number densities


class Network:
  # ----------------------------------------------------------------------------
  # Input
  # ----------------------------------------------------------------------------
  def __init__(self, reactions: List[Reaction], temperature=300,
               gas_density=1e-6, abundances=None, number_densities=None,
               initialise_jacobian=False) -> None:
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
    self.symbols: List[str] = [f"n_{s}" for s in self.species]

    # Properties
    self._temperature = temperature
    if abundances:
      from gcrn.helper_functions import number_densities_from_abundances
      # Initialise number densities from abundances and the gas density
      number_densities = number_densities_from_abundances(abundances,
                                                          gas_density, self.species)
    else:
      if not number_densities:  # populate with random values
        number_densities = {s: np.random.randint(1, 100) for s in species}

    self.initial_number_densities: np.ndarray = \
        self.setup_initial_number_densities(number_densities)
    self._number_densities: np.ndarray = self.initial_number_densities.copy()
    self.number_densities_dict: Dict = {s: self.number_densities[i]
                                        for i, s in enumerate(self.species)}
    # TODO: Additional constraint: combinations, not permutations!
    # e.g. H + H + H2 == H2 + H + H
    # Also need to check this when creating the graph!
    # self.complexes = sorted(list(set(chain.from_iterable(
    #     [[rxn.reactant_complex, rxn.product_complex]
    #      for rxn in self.reactions]))))
    # Complex composition matrix Z (m x c)
    self.complex_composition_matrix = self.create_complex_composition_matrix()
    # Complex incidence matrix D (c x r)
    self.complex_incidence_matrix = self.create_complex_incidence_matrix()
    # Stoichiometric matrix S (m x r)
    self.stoichiometric_matrix = self.complex_composition_matrix @ self.complex_incidence_matrix
    # Outgoing co-incidence matrix of reaction rates K (r x c)
    self.eval_kinetics_matrix, self.eval_kinetics_idxs = self.create_eval_kinetics_matrix()
    self.complex_kinetics_matrix = self.create_complex_kinetics_matrix()
    # Weighted Laplacian (transpose of conventional Laplacian) matrix (c x c)
    self.complex_laplacian = self.create_complex_laplacian_matrix()

    # Vectors for solver
    self.rates_vector: Callable = eval(
        f'lambda Z, K, n: K.dot(np.exp(Z.T.dot(np.log(n))))')
    self.dynamics_vector: np.ndarray = self.calculate_dynamics()

    if initialise_jacobian:
      self.rate_dict: Dict = self.create_rate_dict()
      self.jacobian_func: np.ndarray[Callable] = self.create_jacobian()

  @classmethod
  def from_krome_file(cls, krome_file: str, temperature=300, gas_density=1e-6,
                      abundances=None, number_densities=None,
                      initialise_jacobian=False):
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
        'ref': [],  # reference to rate
        'description': [],  # e.g. radiative association, species exchange
    }

    single_entry_keys = ['idx', 'rate', 'Tmin', 'Tmax',
                         'limit', 'accuracy', 'ref', 'description']

    reactions = []
    with open(krome_file, 'r', encoding='utf-8') as infile:
      while True:
        line = infile.readline().strip()
        if not line:
          break

        if line.startswith("#"):
          continue

        # Check for 'format'
        # TODO: allow for custom tokens like photo
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

    return cls(reactions, temperature=temperature, gas_density=gas_density,
               abundances=abundances, number_densities=number_densities,
               initialise_jacobian=initialise_jacobian)

  @classmethod
  def from_cobold_file(cls, cobold_file: str, temperature=300,
                       gas_density=1e-6, abundances=None, number_densities=None,
                       initialise_jacobian=False):
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

    return cls(reactions, temperature=temperature, gas_density=gas_density,
               abundances=abundances, number_densities=number_densities,
               initialise_jacobian=initialise_jacobian)

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
    from gcrn.helper_functions import reaction_from_complex
    # Write the network to a LaTeX table
    format_dict = group_rxns(self.reactions)

    # Determine tabular format by max number of reactants & products
    max_num_reactants = 1
    max_num_products = 1
    for format_str in format_dict.keys():
      num_reactants = list(format_str).count('R')
      num_products = list(format_str).count('P')
      if num_reactants > max_num_reactants:
        max_num_reactants = num_reactants
      if num_products > max_num_products:
        max_num_products = num_products

    tabular_format = 'r '  # idx
    for i in range(max_num_reactants):
      tabular_format += 'l c '  # for '+', and '->' at end
    for i in range(max_num_products):
      tabular_format += 'l '
      if i < max_num_products - 1:
        tabular_format += 'c '  # for '+'

    tabular_format += 'r r r r'  # alpha, beta, gamma, ref
    table_body = []
    for i, (format_str, rxn_strs) in enumerate(format_dict.items()):
      num_reactants = list(format_str).count('R')
      num_products = list(format_str).count('P')
      subtitle = f'{num_reactants} reactants, {num_products} products'
      table_body.append(r'\midrule')
      table_body.append(r'% ' + subtitle + r'\\')
      table_body.append(r'\midrule')
      # TODO:
      # Organise by description (optional)
      for rxn_str in rxn_strs:
        reactant_complex, product_complex = rxn_str.split(' -> ')
        reactants = reactant_complex.split(' + ')
        products = product_complex.split(' + ')
        rxn = reaction_from_complex(reactant_complex, product_complex, self)
        reactant_str, product_str = '', ''
        # TODO
        # abstract following to an internal function
        for i in range(max_num_reactants):
          try:
            reactant_str += f'{reactants[i]} & '
          except IndexError:
            reactant_str += ' & '
          if i < max_num_reactants - 1:
            reactant_str += ' + & ' if i < len(reactants) - 1 else '&'

        reactant_str += ' -> '
        for i in range(max_num_products):
          try:
            product_str += f'{products[i]} &'
          except IndexError:
            product_str += ' & '
          if i < max_num_products - 1:
            product_str += ' + &' if i < len(products) - 1 else '&'
        alpha, beta, gamma = constants_from_rate(rxn.rate_expression)
        line = f'{rxn.idx} & {reactant_str} & {product_str} '\
            f' {alpha:2.2e} & {beta:1.2f} & {gamma} & {rxn.reference}'
        table_body.append(line + r'\\')

    # TODO:
    # Create a write_table() function in utilities
    with open(path, 'w', encoding='utf-8') as outfile:
      # Table header
      outfile.write(r'\begin{table}' + '\n')
      # Centering
      outfile.write(r'\centering' + '\n')
      # Table caption
      outfile.write(r'\caption{' + '\n\n}\n')
      # Tabular
      outfile.write(r'\begin{tabular}{' + tabular_format + '}\n')
      table_body = '\n'.join(table_body)
      outfile.write(table_body)
      # End tabular
      outfile.write('\n' + r'\end{tabular}' + '\n')
      # Label
      outfile.write(r'\label{table:}' + '\n')
      # End table
      outfile.write(r'\end{table}' + '\n')

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
  # Solver methods
  # ----------------------------------------------------------------------------
  def setup_initial_number_densities(self, initial_number_densities: Union[Dict, List, np.ndarray]) -> np.ndarray:
    # Create the array of number densities with the same indexing as the species
    # in the network
    number_densities = np.zeros(len(self.species))
    if isinstance(initial_number_densities, dict):
      species = []
      for i, s in enumerate(self.species):
        number_densities[i] = initial_number_densities[s]
        species.append(s)
      self.species = species
    elif isinstance(initial_number_densities, List):
      number_densities = np.array(initial_number_densities)
    elif isinstance(initial_number_densities, np.ndarray):
      number_densities = initial_number_densities

    return number_densities

  def create_rate_dict(self) -> Dict:
    # Create the dictionary describing the time-dependent system of equations
    # d[X]/dt = x_1 + x_2 + ...
    # TODO:
    # Potentially build this from just evaluating ZDKExp(Z.TLn(x))?
    # Put in 'n_{key}' for 'x', evaluate RHS symbolically and it should be
    # the rates!
    rate_dict = {}  # keys are species
    for reaction in self.reactions:
      expression = reaction.mass_action_rate_expression
      reactant_symbols = [
          f"n_{key}" for key in reaction.stoichiometry[0].keys()]
      product_symbols = [
          f"n_{key}" for key in reaction.stoichiometry[1].keys()]
      for symbol in reactant_symbols:
        # TODO:
        # If symbol appears in both reactant and product sides, ignore it
        # But also check it appears the same number of times, e.g.
        # H2 + H2 -> H2 + H + H
        # has 'H2' on both sides, but one side has one more so it's still
        # a destruction
        # Alternatively, just leave it and check for zero rates at the end
        # Idea:
        # Check if the opposite symbol exists in the dictionary-key list, and
        # if so, remove it and don't add the new one since it means we're adding
        # e.g. dict['A'] = 'B' ' - B'
        # But again, might just be faster to catch them all at the end?
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
    # Create a Jacobian where each entry is a Callable, taking in number densities
    # as indexed in dynamics.network.species and a temperature
    # TODO:
    # Fix indexing for rate dict vs symbols/species
    num_species = len(self.species)
    # Fill with function that returns zero instead?
    jacobian = np.zeros((num_species, num_species), dtype=object)
    for i, s1 in enumerate(self.symbols):
      # for i, (s, rate) in enumerate(self.rate_dict.items()):
      expression = '+'.join(self.rate_dict[s1])
      # expression = "+".join(rate)
      for j, s2 in enumerate(self.symbols):
        differential = sympy.diff(expression, s2)
        rate = str(differential).replace("Tgas", "T")
        pattern = r"_[A-Z0-9]*"
        # TODO:
        # Do the whole replacement with regex!
        rate = re.sub(pattern,
                      lambda s: f"[{self.species.index(s.group()[1:])}]",
                      rate)
        jacobian[i, j] = eval(f"lambda T, n: {rate}")

    return jacobian

  def evaluate_jacobian(self, temperature: float,
                        number_densities: np.ndarray) -> np.ndarray:
    # Evaluate the function Jacobian by calling each element with the provided
    # temperature and number densities
    if not self.jacobian_func:
      self.rate_dict = self.create_rate_dict()
      self.jacobian_func = self.create_jacobian()
    jacobian = np.zeros_like(self.jacobian_func, dtype=float)
    x, y = jacobian.shape
    for i, j in product(range(x), range(y)):
      jacobian[i, j] = self.jacobian_func[i, j](temperature, number_densities)

    return jacobian

  def create_rates_vector(self, number_densities: np.ndarray) -> np.ndarray:
    # Create the vector v(x) = K Exp(Z.T Ln(x)) that includes the stoichiometry
    # into reaction rates
    # 'number_densities' must have the same indexing as species!
    # if (number_densities < 0).any():
    #   print("Error: number densities negative!")
    #   exit(-1)
    Z = self.complex_composition_matrix
    K = self.complex_kinetics_matrix
    # # Check if number densities are below minimum value of 1e-20 and set the
    # # rates to zero if so
    # boundary_mask = (number_densities <= 1e-20)
    # number_densities[boundary_mask] = 1e-20
    rates_vector = K.dot(np.exp(Z.T.dot(np.log(number_densities))))

    return rates_vector

  def calculate_dynamics(self) -> np.ndarray:
    # Calculate the RHS ZD v(x) = Sv(x)
    S = self.stoichiometric_matrix
    Z = self.complex_composition_matrix
    K = self.complex_kinetics_matrix
    dynamics_vector = S.dot(self.rates_vector(Z, K, self.number_densities))

    return dynamics_vector

  def solve(self, evolution_times: List[float],
            initial_time=0, create_jacobian=False, jacobian=None,
            atol=1e-30, rtol=1e-4, eqm_tolerance=1e-5, n_subtime=10,
            return_eqm_times=False,
            **solver_kwargs) -> List[np.ndarray]:
    def f(t: float, y: np.ndarray, temperature=None) -> List[np.ndarray]:
      # Create RHS ZDK Exp(Z.T Ln(x))
      # TODO:
      # How do we speed this up???
      # I think it's slow because the rates vector eval is in Python, whereas
      # it would be much faster if it were compiled
      S = self.stoichiometric_matrix
      Z = self.complex_composition_matrix
      K = self.complex_kinetics_matrix

      return S @ self.rates_vector(Z, K, y)

    if create_jacobian:
      # Use the analytical Jacobian stored in NetworkDynamics
      jacobian = self.evaluate_jacobian

    if jacobian:
      solver = ode(f, jacobian).set_integrator("vode", method='bdf',
                                               atol=atol, rtol=rtol,
                                               **solver_kwargs)
      solver.set_jac_params(self.temperature, self.number_densities)
    else:
      solver = ode(f).set_integrator("vode", method='bdf', atol=atol, rtol=rtol,
                                     with_jacobian=True,
                                     **solver_kwargs)

    # Initial values
    solver.set_initial_value(self.number_densities, initial_time)

    # TODO: Check for failure and return codes
    # TODO: Verbosity control
    number_densities = np.full((len(evolution_times),
                                len(self.number_densities)),
                               np.nan)

    # Store times when species reach equilibrium
    is_eqm = np.full(len(self.number_densities), False)
    eqm_times = np.full(len(self.number_densities), np.nan)
    prev_time = initial_time
    for i_time, current_time in enumerate(evolution_times):
      full_dt = current_time - prev_time
      # substeps in time
      dt = full_dt / n_subtime
      while solver.successful() and solver.t < current_time:
        solver.set_f_params(self.temperature)
        number_densities[i_time] = solver.integrate(solver.t + dt)

        if i_time > 0:
          # Check for equilibrium convergence
          # If relative difference <= 'eqm_tolerance', solution has converged
          # to equilibrium (regardless of absolute tolerance)
          ratio = np.abs(1. - (number_densities[i_time] /
                               number_densities[i_time-1]))
          is_eqm = (ratio <= eqm_tolerance)
          idxs = np.isnan(eqm_times) & is_eqm
          eqm_times[idxs] = solver.t
          if np.all(is_eqm):
            # print(f"Equilibrium convergence after {i_time+1} steps"
            #       f" (t = {current_time:.2e}) [s].")
            number_densities[i_time:] = number_densities[i_time].copy()
            self.number_densities = number_densities[-1]
            if return_eqm_times:
              return number_densities, eqm_times
            else:
              return number_densities

      prev_time = current_time

    self.number_densities = number_densities[-1]
    return number_densities

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
  # Methods for updating attributes
  # ----------------------------------------------------------------------------
  def _update(self):
    # Update complex kinetics matrix
    self.number_densities_dict = {s: self.number_densities[i]
                                  for i, s in enumerate(self.species)}
    self.complex_kinetics_matrix =\
        self.create_complex_kinetics_matrix()

  def update_number_densities_from_abundances(self, abundances: Dict,
                                              gas_density: float):
    from gcrn.helper_functions import number_densities_from_abundances
    self.number_densities = number_densities_from_abundances(abundances,
                                                             gas_density,
                                                             self.species)

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
    if isinstance(value, dict):
      # Convert to ndarray, assuming same index as self.species
      n = np.zeros(shape=len(self.species))
      for i, s in enumerate(self.species):
        n[i] = value[s]
      self._number_densities = n
    elif isinstance(value, np.ndarray):
      self._number_densities = value
    self._update()
