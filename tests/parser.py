# Test sympy parsing vs function-Jacobian creation
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


initial_number_densities = {
    "H": 1e12,
    "H2": 1e-4,
    "OH": 1e-12,
    "C": 10**(8.39),  # solar
    "O": 10**(8.66),  # solar
    # "C": 10**(8.66),  # C-rich
    # "O": 10**(8.39),  # C-rich
    "CH": 1e-12,
    "CO": 1e-12,
    "M": 1e11,
}

# network = Network.from_krome_file('../res/catalyst_co.ntw')
network = Network.from_krome_file('../res/simplified_co.ntw')
dynamics = NetworkDynamics(network, initial_number_densities, temperature=5700)
jac_func = create_functional_jacobian(dynamics)
# for temperature in [300, 3000, 5000, 10000, 15000, 20000, 25000, 30000]:
#   dynamics.evaluate_jacobian(temperature, dynamics.number_densities)
#   eval_jac = evaluate_functional_jacobian(
#       jac_func, temperature, dynamics.number_densities)

print(network.complex_composition_matrix)
print(network.species)
print(network.complexes)
