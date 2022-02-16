from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from gcrn.helper_functions import enumerated_product, setup_dynamics
from gcrn.network import Network
import numpy as np
import time


# Dummy network dynamics to profile efficiency
res_dir = '../res'
# network_file = f'{res_dir}/solar_co_w05.ntw'
network_file = f'{res_dir}/cno.ntw'

initial_number_densities = {
    "H": 1e12,
    "H2": 1e-4,
    "OH": 1e-12,
    "C": 10**(8.39),  # solar
    "O": 10**(8.66),  # solar
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

network = Network.from_krome_file(network_file)

temperatures = np.linspace(3000, 15000, num=50)
gas_densities = np.logspace(-12, -6, num=50)
timescales = np.logspace(-6, 6, num=10)
start_time = time.time()
for (i, j), (T, rho) in enumerated_product(temperatures, gas_densities):
  hydrogen_density = gas_density_to_hydrogen_number_density(rho)

  dynamics = setup_dynamics(network, 3000, initial_number_densities,
                            hydrogen_density)
  number_densities = dynamics.solve(timescales,
                                    dynamics.initial_number_densities)
  print(f"Done {i+1} / {len(gas_densities)} densities "
        f"{j+1} / {len(temperatures)} temperatures",
        end="\r")

end_time = time.time()
print(
    f"\n{len(timescales)} timescales,\n"
    f"{len(temperatures)} temperatures,\n"
    f"{len(gas_densities)} densities.\n"
    f"Total time: {(end_time - start_time):.3f} seconds."
)
