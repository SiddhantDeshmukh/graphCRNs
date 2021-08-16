from typing import Union, Dict, List
import numpy as np
from src.network import Network
from scipy.integrate import ode


# TODO:
# Clear up distinction between Network and NetworkDynamics
# Where should temperature be represented? How to interface using it?

class NetworkDynamics():
  def __init__(self, network: Network,
               initial_number_densities: Union[Dict, List, np.ndarray],
               temperature=300) -> None:
    self.network = network
    self.species = None  # should be indexed tbe same as 'number densities'
    if temperature:
      self.network.temperature = temperature
    self.initial_number_densities =\
        self.setup_initial_number_densities(initial_number_densities)
    self.number_densities = self.initial_number_densities.copy()

    self.rates_vector = self.create_rates_vector(self.number_densities,
                                                 temperature=temperature)
    self.dynamics_vector = self.calculate_dynamics()

    # Properties
    self._temperature = temperature

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

  def create_rates_vector(self, number_densities: np.ndarray,
                          temperature=None) -> np.ndarray:
    # Create the vector v(x) = K Exp(Z.T Ln(x)) that includes the stoichiometry
    # into reaction rates
    # 'number_densities' must have the same indexing as species!
    Z = self.network.complex_composition_matrix

    if temperature:
      K = self.network.create_complex_kinetics_matrix(temperature)
    else:
      K = self.network.complex_kinetics_matrix

    rates_vector = K.dot(np.exp(Z.T.dot(np.log(number_densities))))

    return rates_vector

  def calculate_dynamics(self) -> np.ndarray:
    # Calculate the RHS ZD v(x) = Sv(x)
    S = self.network.stoichiometric_matrix
    dynamics_vector = S.dot(self.rates_vector)

    return dynamics_vector

  def solve(self, timescale: float, initial_number_densities: np.ndarray,
            initial_time=0, create_jacobian=False, jacobian=None,
            atol=1e-30, rtol=1e-4) -> np.ndarray:
    # TODO:
    # Add atol, rtol and other solver params
    def f(t: float, y: np.ndarray, temperature=None) -> List[np.ndarray]:
      # Create RHS ZDK Exp(Z.T Ln(x))
      Z = self.network.complex_composition_matrix
      D = self.network.complex_incidence_matrix
      rates_vector = self.create_rates_vector(y, temperature)

      return Z @ D @ rates_vector

    if create_jacobian:
      # Create the Jacobian matrix analytically
      pass
    if jacobian:
      solver = ode(f, jacobian).set_integrator("vode", method='bdf',
                                               atol=atol, rtol=rtol)
    else:
      solver = ode(f).set_integrator("vode", method='bdf', atol=atol, rtol=rtol,
                                     with_jacobian=True)

    # Initial values
    solver.set_initial_value(initial_number_densities, initial_time)

    # TODO: Add better time-stepping control
    dt = timescale - initial_time

    # TODO: Check for failure and return codes
    number_densities = []
    while solver.successful() and solver.t < timescale:
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
      rates_vector = self.create_rates_vector(y, temperature)

      return Z @ D @ rates_vector

    if create_jacobian:
      # Create the Jacobian matrix analytically
      pass
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
    cfl = 0.9
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
      
      # Calculate 'dt'
      # dt *= cfl * np.min(y / y_previous)
      # print(dt, np.min(y / y_previous))
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
