# Test sympy parsing vs function-Jacobian creation
from typing import Dict
from numba import jit
from itertools import product
from src.network import Network
from src.dynamics import NetworkDynamics
import numpy as np
import sympy
import re
from math import exp


def create_functional_jacobian(dynamics: NetworkDynamics) -> np.ndarray:
  # Create a Jacobian where each entry is a Callable, taking in number densities
  # as indexed in dynamics.network.species and a temperature
  num_species = len(dynamics.species)
  jacobian = np.zeros((num_species, num_species), dtype=object)
  for i, rate in enumerate(dynamics.rate_dict.values()):
    expression = "+".join(rate)
    for j, symbol in enumerate(dynamics.symbols):
      differential = sympy.diff(expression, symbol)
      rate = str(differential).replace("Tgas", "T")
      pattern = r"_[A-Z0-9]*"
      # TODO:
      # Do the whole replacement with regex!
      rate = re.sub(pattern,
                    lambda s: f"[{dynamics.species.index(s.group()[1:])}]",
                    rate)
      jacobian[i, j] = eval(f"lambda T, n: {rate}")

  return jacobian


def evaluate_functional_jacobian(func_jac: np.ndarray,
                                 T: float, n: np.ndarray) -> np.ndarray:
  # Evaluate the function Jacobian by calling each element with the provided
  # temperature and number densities
  jac = np.zeros_like(func_jac, dtype=float)
  x, y = jac.shape
  for i, j in product(range(x), range(y)):
    jac[i, j] = func_jac[i, j](T, n)
  return jac


def symbolic_RHS(dynamics: NetworkDynamics, temperature: float):
  # Symbolically create the RHS vector of rates ZDK Exp(Z.T Ln(x))
  # WARNING: Does not sanitise input to eval!
  dynamics.temperature = temperature
  Z = dynamics.network.complex_composition_matrix
  D = dynamics.network.complex_incidence_matrix
  K = dynamics.network.complex_kinetics_matrix

  # TODO:
  # Do this symbolically with Sympy. It should only be called ONCE when
  # creating the Jacobian function
  S = Z @ D  # no symbols
  # symbols are in 'number_densities'?
  rates_vector = K.dot(np.exp(Z.T.dot(np.log(dynamics.number_densities))))

  # symbolic vector
  rhs = S.dot(rates_vector)


initial_number_densities = {
    "H": 1e12,
    "H2": 1e-4,
    # "H2": 1e-12,
    "OH": 1e-12,
    # "C": 10**(8.39),  # solar
    # "O": 10**(8.66),  # solar
    "N": 10**(7.83),
    "C": 10**(8.66),  # C-rich
    "O": 10**(8.39),  # C-rich
    "CH": 1e-12,
    "CO": 1e-12,
    "CN": 1e-12,
    "NH": 1e-12,
    "NO": 1e-12,
    "C2": 1e-12,
    "O2": 1e-12,
    "N2": 1e-12,
    "M": 1e11,
}
# network = Network.from_krome_file('../res/catalyst_co.ntw')
network = Network.from_krome_file('../res/solar_co_w05.ntw')
dynamics = NetworkDynamics(network, initial_number_densities, temperature=5700)
print(dynamics.number_densities)
exit()

for temperature in [300, 3000, 5000, 10000, 15000, 20000, 25000, 30000]:
  dynamics.evaluate_jacobian(temperature, dynamics.number_densities)
  for i, time in enumerate(np.logspace(-6, 3, num=50)):
    dynamics.solve(time, dynamics.number_densities, create_jacobian=True)
    print(f"Done time {i + 1} of 50.")
