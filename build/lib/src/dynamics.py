from typing import Union, Dict, List
from astropy.units.equivalencies import temperature
import numpy as np
from src.network import Network
from scipy.integrate import ode
import sympy
from numba import jit
import re
from itertools import product
from math import exp  # used in 'eval'


# TODO:
# Clear up distinction between Network and NetworkDynamics
# Where should temperature be represented? How to interface using it?

class NetworkDynamics():
  def __init__(self, network: Network,
               initial_number_densities: Union[Dict, List, np.ndarray],
               temperature=300) -> None:
    self.network = network
    self.species = self.network.species
    self.symbols = [f"n_{s}" for s in self.species]
    if temperature:
      self.network.temperature = temperature

    # Properties
    self._temperature = temperature
    # TODO:
    # Number densities as properties
    self.initial_number_densities =\
        self.setup_initial_number_densities(initial_number_densities)
    self.number_densities = self.initial_number_densities.copy()

    self.rates_vector = self.create_rates_vector(self.number_densities,
                                                 temperature=temperature)
    self.dynamics_vector = self.calculate_dynamics()

    # Rates and Jacobian attributes
    self.rate_dict = self.create_rate_dict()
    self.jacobian_func = self.create_jacobian()

  # TODO:
  # Have number densities and initial number densities as properties?
  def setup_initial_number_densities(self, initial_number_densities: Union[Dict, List, np.ndarray]) -> np.ndarray:
    # Create the array of number densities with the same indexing as the species
    # in the network
    number_densities = np.zeros(len(self.network.species))
    if isinstance(initial_number_densities, dict):
      species = []
      for i, s in enumerate(self.network.species):
        number_densities[i] = initial_number_densities[s]
        species.append(s)
      self.species = species
    elif isinstance(initial_number_densities, List):
      number_densities = np.array(initial_number_densities)
    elif isinstance(initial_number_densities, np.ndarray):
      number_densities = initial_number_densities

    return number_densities

  # ----------------------------------------------------------------------------
  # Methods for Jacobian and rate analysis
  # ----------------------------------------------------------------------------
  def create_rate_dict(self) -> Dict:
    # Create the dictionary describing the time-dependent system of equations
    # d[X]/dt = x_1 + x_2 + ...
    # TODO:
    # Fix double-counting!!!
    # TODO:
    # Potentially build this from just evaluating ZDKExp(Z.TLn(x))?
    # Put in 'n_{key}' for 'x', evaluate RHS symbolically and it should be
    # the rates!
    rate_dict = {}  # keys are species
    for reaction in self.network.reactions:
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

    #   num_species = len(self.species)
    # jacobian = np.zeros((num_species, num_species), dtype=object)
    # # Set up d[A]/dt = a_1 + a_2 + ... for each differential expression with A
    # for i, rate in enumerate(self.rate_dict.values()):
    #   expression = " + ".join(rate)
    #   for j, symbol in enumerate(self.symbols):
    #     differential = sympy.diff(expression, symbol)
    #     jacobian[i, j] = str(differential)

    return jacobian

  # @jit(nopython=True)
  def evaluate_jacobian(self, temperature: float,
                        number_densities: np.ndarray) -> np.ndarray:
    # # Evaluate the 'jacobian_str' using sympy
    # # Note that 'number_densities' must have the same indexing as
    # # network species
    # n_dict = {s: n for s, n in zip(self.symbols, number_densities)}
    # n_dict['Tgas'] = temperature
    # rows, cols = self.jacobian_str.shape
    # jacobian = np.zeros((rows, cols))
    # for i in range(rows):
    #   for j in range(cols):
    #     # Evaluate with temperature and number densities!
    #     jacobian[i, j] = sympy.parse_expr(self.jacobian_str[i, j],
    #                                       evaluate=True, local_dict=n_dict)
    # Evaluate the function Jacobian by calling each element with the provided
    # temperature and number densities
    jacobian = np.zeros_like(self.jacobian_func, dtype=float)
    x, y = jacobian.shape
    for i, j in product(range(x), range(y)):
      jacobian[i, j] = self.jacobian_func[i, j](temperature, number_densities)

    return jacobian

  def create_rates_vector(self, number_densities: np.ndarray,
                          temperature=None, limit_rates=False) -> np.ndarray:
    # Create the vector v(x) = K Exp(Z.T Ln(x)) that includes the stoichiometry
    # into reaction rates
    # 'number_densities' must have the same indexing as species!
    # if (number_densities < 0).any():
    #   print("Error: number densities negative!")
    #   exit(-1)
    Z = self.network.complex_composition_matrix

    # Update kinetics matrix if temperature is provided
    # TODO:
    # Only allow temperature change in Dynamics, which automatically updates
    # all matrices
    if temperature:
      K = self.network.create_complex_kinetics_matrix(limit_rates=limit_rates)
    else:
      K = self.network.complex_kinetics_matrix

    # Check if number densities are below minimum value of 1e-20 and set the
    # rates to zero if so
    # TODO:
    # Check ratio of number densities instead?
    boundary_mask = (number_densities <= 1e-10)
    number_densities[boundary_mask] = 1e-10
    rates_vector = K.dot(np.exp(Z.T.dot(np.log(number_densities))))

    return rates_vector

  def calculate_dynamics(self) -> np.ndarray:
    # Calculate the RHS ZD v(x) = Sv(x)
    S = self.network.stoichiometric_matrix
    dynamics_vector = S.dot(self.rates_vector)

    return dynamics_vector

  def solve(self, timescale: float, initial_number_densities: np.ndarray,
            initial_time=0, create_jacobian=False, jacobian=None,
            limit_rates=False,
            atol=1e-30, rtol=1e-4,
            **solver_kwargs) -> np.ndarray:
    def f(t: float, y: np.ndarray,
          temperature=None, limit_rates=False) -> List[np.ndarray]:
      # Create RHS ZDK Exp(Z.T Ln(x))
      Z = self.network.complex_composition_matrix
      D = self.network.complex_incidence_matrix
      rates_vector = self.create_rates_vector(y, temperature, limit_rates)

      return Z @ D @ rates_vector

    if create_jacobian:
      # Use the analytical Jacobian stored in NetworkDynamics
      jacobian = self.evaluate_jacobian

    # TODO:
    # Experiment with 'min_step' and 'max_step' options

    if jacobian:
      solver = ode(f, jacobian).set_integrator("vode", method='bdf',
                                               atol=atol, rtol=rtol,
                                               **solver_kwargs)
      # TODO:
      # Refactor number densities to be a property and set this as jac param
      solver.set_jac_params(self.temperature, initial_number_densities)
    else:
      solver = ode(f).set_integrator("vode", method='bdf', atol=atol, rtol=rtol,
                                     with_jacobian=True,
                                     **solver_kwargs)

    # Initial values
    solver.set_initial_value(initial_number_densities, initial_time)

    # TODO: Add better time-stepping control
    dt = timescale - initial_time

    # TODO: Check for failure and return codes
    number_densities = []
    while solver.successful() and solver.t < timescale:
      solver.set_f_params(self.temperature, limit_rates)
      number_densities.append(solver.integrate(solver.t + dt))

    return number_densities

  def solve_steady_state(self, initial_number_densities: np.ndarray,
                         create_jacobian=False, jacobian=None,
                         atol=1e-20, rtol=1e-5,
                         eqm_atol=1e-13, eqm_rtol=1e-5,
                         initial_dt=0.1,
                         max_iter=1000) -> Union[np.ndarray, float]:
    # Solves the CRN dynamics for a steady-state given initial conditions and
    # returns the steady-state number densities alongside the time to reach
    # steady-state
    # Solves the system until the change is less than 'tol'
    # TODO:
    # Move this to be a generic method like 'RHS'
    def f(t: float, y: np.ndarray, temperature=None) -> List[np.ndarray]:
      # Create RHS ZDK Exp(Z.T Ln(x))
      Z = self.network.complex_composition_matrix
      D = self.network.complex_incidence_matrix
      v = self.create_rates_vector(y, temperature)

      return Z @ D @ v

    if create_jacobian:
      # Create the Jacobian matrix analytically
      self.create_jacobian()
      jacobian = self.evaluate_jacobian
    if jacobian:
      solver = ode(f, jacobian).set_integrator("vode", method='bdf',
                                               atol=atol, rtol=rtol)
    else:
      solver = ode(f).set_integrator("vode", method='bdf', with_jacobian=True,
                                     atol=atol, rtol=rtol)

    # Initial values
    solver.set_initial_value(initial_number_densities)

    # TODO: Add better time-stepping control
    min_dt = 1e-6
    max_dt = 1e6
    dt = initial_dt

    # TODO: Check for failure and return codes
    y_previous = initial_number_densities.copy()
    done = False
    n_iter = 0

    # TODO:
    # verbosity control
    while not done:
      y = solver.integrate(solver.t + dt)
      absolute_difference = np.abs(y - y_previous)
      relative_difference = np.abs(1 - y / y_previous)

      # Converged (absolute tolerance)
      if (absolute_difference < eqm_atol).all():
        done = True

      # Converged (relative tolerance)
      if (relative_difference < eqm_rtol).all():
        done = True

      if n_iter > max_iter:
        print(f"Max iterations {max_iter} exceeded.")
        print(
            f"Current difference: {absolute_difference}, absolute tolerance: {atol}.")
        print(
            f"Current ratio: {relative_difference}, relative tolerance: {rtol}.")
        done = True

      if (y < 0).any():
        print(f"Error: number densities < 0 at time {solver.t}.")
        done = True

      y_previous = y.copy()

      # Calculate 'dt' from Jacobian linear eigenvalue analysis
      eigenvalues = np.linalg.eigvals(jacobian(self.temperature, y))
      eigenvalues[np.abs(eigenvalues) < 1e-10] = 0
      timescales = 1 / np.abs(np.real(eigenvalues[eigenvalues < 0]))
      # dt = np.sort(timescales)[0]
      dt = np.mean(timescales[:2])

      print(f"{n_iter + 1} / {max_iter}: dt = {dt}")

      if dt < min_dt:
        dt = min_dt
      if dt > max_dt:
        dt = max_dt

      n_iter += 1
    return y, solver.t

  @property
  def temperature(self):
    return self._temperature

  @temperature.setter
  def temperature(self, value: float):
    self._temperature = value
    self.network._temperature = value
