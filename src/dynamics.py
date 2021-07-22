from typing import Union, Dict, List
import numpy as np
from src.network import Network


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

  def create_rates_vector(self, number_densities: np.ndarray) -> np.ndarray:
    # Create the vector v(x) = K Exp(Z.T Ln(x)) that includes the stoichiometry
    # into reaction rates
    # 'number_densities' must have the same indexing as species!
    Z = self.network.complex_composition_matrix
    K = self.network.kinetics_matrix
    rates_vector = K.dot(np.exp(Z.T.dot(np.log(number_densities))))

    return rates_vector

  def calculate_dynamics(self) -> np.ndarray:
    # Calculate the RHS ZD v(x) = Sv(x)
    S = self.network.stoichiometric_matrix
    dynamics_vector = S.dot(self.rates_vector)

    return dynamics_vector
