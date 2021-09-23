# Test networks on a density-temperature grid
from itertools import product
from typing import Dict
import numpy as np
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density, log_abundance_to_number_density
from src.network import Network
from src.dynamics import NetworkDynamics


def get_photospheric_conditions(temperature_resolution: int,
                                density_resolution: int) -> np.ndarray:
  min_temperature = 3000
  max_temperature = 9000
  min_density = -9  # log space
  max_density = -6  # log space

  temperature = np.linspace(min_temperature, max_temperature,
                            num=temperature_resolution)
  density = np.logspace(min_density, max_density, num=density_resolution)

  return np.meshgrid(temperature, density)


def setup_number_densities(network: Network,
                           initial_abundances: Dict,
                           gas_density: float) -> np.ndarray:
  # Initialise number densities given gas density
  hydrogen_density = gas_density_to_hydrogen_number_density(gas_density)
  number_densities = np.ndarray(len(network.species))
  for i, s in enumerate(network.species):
    number_densities[i] = log_abundance_to_number_density(
        initial_abundances[s], np.log10(hydrogen_density))

  return number_densities


temperature_resolution, density_resolution = 50, 50
temperature, density = get_photospheric_conditions(temperature_resolution,
                                                   density_resolution)

network_dir = '../res'
network = Network.from_krome_file(f"{network_dir}/react-solar-umist12")

abundance_dict = {
    "H": 1e12,
    "H2": 1e-4,
    "OH": 1e-12,
    "C": 10**(8.39),
    "O": 10**(8.66),
    "CH": 1e-12,
    "CO": 1e-12,
    "M": 1e11,
}

timescale = 1000  # seconds

for x in range(temperature.shape[0]):
  for y in range(temperature.shape[1]):
    T, rho = temperature[x, y], density[x, y]
    initial_densities = setup_number_densities(network, abundance_dict, rho)
    dynamics = NetworkDynamics(network, initial_densities, temperature=T)
    # Unlimited and limited densities
    unlimited_densities = dynamics.solve(timescale, initial_densities)
    limited_densities = dynamics.solve(timescale, initial_densities,
                                       limit_rates=True)
    # Boundary limits
    network.set_reaction_limit('boundary')
    boundary_densities = dynamics.solve(timescale, initial_densities,
                                        limit_rates=True)
    # Reset reaction limits
    network.set_reaction_limit('weak')
