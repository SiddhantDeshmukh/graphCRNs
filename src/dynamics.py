from typing import Union, Dict, List
import numpy as np
from src.network import Network
from scipy.integrate import ode


class NetworkDynamics():
  def __init__(self, network: Network,
               initial_number_densities: Union[Dict, List, np.ndarray]) -> None:
    self.network = network
    self.initial_number_densities = self.setup_initial_number_densities(
        initial_number_densities)
    self.number_densities = self.initial_number_densities.copy()

    self.rates_vector = self.create_rates_vector(self.number_densities)
    self.dynamics_vector = self.calculate_dynamics()

  def setup_initial_number_densities(self, initial_number_densities: Union[Dict, List, np.ndarray]) -> np.ndarray:
    # Create the array of number densities with the same indexing as the species
    # in the network
    number_densities = np.zeros(len(self.network.species))
    if isinstance(initial_number_densities, dict):
      for i, species in enumerate(self.network.species):
        number_densities[i] = initial_number_densities[species]
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
      solver = ode(f).set_integrator("vode", method='bdf', atol=atol, rtol=rtol)

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
                         tolerance=1e-5,
                         max_iter=1000) -> Union[np.ndarray, float]:
    # Solves the CRN dynamics for a steady-state given initial conditions and
    # returns the steady-state number densities alongside the time to reach
    # steady-state
    # Solves the system until the change is less than 'tol'
    # TODO:
    # Wrap self.solve() using 'atol', 'rtol'?
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
      solver = ode(f, jacobian).set_integrator("vode", method='bdf')
    else:
      solver = ode(f).set_integrator("vode", method='bdf')

    # Initial values
    solver.set_initial_value(initial_number_densities)

    # TODO: Add better time-stepping control
    dt = 0.1

    # TODO: Check for failure and return codes
    y_previous = initial_number_densities.copy()
    done = False
    n_iter = 0

    # TODO:
    # verbosity control
    while not done:
      y = solver.integrate(solver.t + dt)
      difference = np.abs(y - y_previous)
      if (difference < tolerance).all():
        done = True

      if n_iter > max_iter:
        print(f"Max iterations {max_iter} exceeded.")
        print(f"Current difference is {difference} with tolerance {tolerance}.")
        done = True

      y_previous = y.copy()
      n_iter += 1
    return y, solver.t
