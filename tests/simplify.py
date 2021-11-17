# Simplify a network across a density-temperature regime
# Iterative scheme, we start with the full network and reduce iteratively to
# most important reactions to reproduce results from the full network for all
# important species within a given tolerance
import numpy as np
from gcrn.helper_functions import run_on_density_temperature_grid
from gcrn.network import Network
from gcrn.dynamics import NetworkDynamics


def main():
  # 1. Define initial network (N0)
  network_dir = '../res'
  network_file = f"{network_dir}/solar_co_w05.ntw"
  # network_file = f"{network_dir}/cno.ntw"
  network = Network.from_krome_file(network_file)

  initial_abundances = {
      "H": 1e12,
      "H2": 1e-4,
      "OH": 1e-12,
      "C": 10**(8.39),
      "O": 10**(8.66),
      "N": 10**(7.83),
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

  # User control params
  density_range = (-10, -6)  # log
  temperature_range = (3000, 20000)  # linear

  num_density_points = 50
  num_temperature_points = 50

  # 2. Initialise density-temperature grid
  density = np.logspace(*density_range, num=num_density_points)
  temperature = np.linspace(*temperature_range, num=num_temperature_points)
  timescales = np.logspace(0, 4, num=5)

  # 3. Run dynamics on grid with network N_0
  number_densities = run_on_density_temperature_grid(density, temperature,
                                                     timescales, network,
                                                     initial_abundances)

  print(number_densities.shape)

  # 4. Find shortest paths for every source-target pair of interest

  # 5. Store most important reactions alongside current network

  # 6. Create new network with only most important reactions N_i

  # 7. Run dynamics with new network N_i

  # 8. Compare results, see if tolerance sufficient

  # 9. If good reduction, continue to reduce further

  # 10. If bad reduction, stop.
  return


if __name__ == "__main__":
  main()
