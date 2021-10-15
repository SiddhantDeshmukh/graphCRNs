# Test solver min/max step size to see if it improves stability
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from tests.helper_functions import run_and_plot, setup_number_densities, setup_dynamics, solve_dynamics
from gcrn.network import Network
import numpy as np


initial_number_densities = {
    "H": 1e12,
    "H2": 1e-4,
    # "H2": 1e-12,
    "OH": 1e-12,
    "C": 10**(8.39),  # solar
    "O": 10**(8.66),  # solar
    "N": 10**(7.83),
    # "C": 10**(8.66),  # C-rich
    # "O": 10**(8.39),  # C-rich
    "CH": 1e-12,
    "CO": 1e-12,
    "CN": 1e-12,
    "NH": 1e-12,
    "NO": 1e-12,
    "C2": 1e-12,
    "O2": 1e-12,
    "N2": 1e-12,
    "M": 1e11
}

res_dir = '../res/'
network_file = f"{res_dir}/co_test.ntw"

gas_density = 1e-6  # [g cm^-3]
hydrogen_density = gas_density_to_hydrogen_number_density(gas_density)

network = Network.from_krome_file(network_file)

network.number_densities = setup_number_densities(initial_number_densities,
                                                  hydrogen_density, network)

temperatures = [1000, 5000, 10000]
times = np.logspace(-6, 3, num=10)

run_and_plot(temperatures, times, network, '../out/figs/step_size.png',
             initial_number_densities, hydrogen_density)
