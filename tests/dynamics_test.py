from itertools import product
from sadtools.utilities.chemistryUtilities import gas_density_to_hydrogen_number_density
from gcrn.helper_functions import setup_dynamics
from gcrn.network import Network


# Dummy network dynamics to profile efficiency
res_dir = '../res'
network_file = f'{res_dir}/solar_co_w05.ntw'

initial_number_densities = {
    "H": 1e12,
    "H2": 1e-4,
    "OH": 1e-12,
    "C": 10**(8.39),  # solar
    "O": 10**(8.66),  # solar
    "CH": 1e-12,
    "CO": 1e-12,
    "M": 1e11,
}


network = Network.from_krome_file(network_file)

temperatures = [3000, 5000, 10000]
gas_densities = [1e-10, 1e-8, 1e-6]
timescales = [100, 1000, 10000]
for T, rho in product(temperatures, gas_densities):
  print(f"Solving rho = {rho:.2e} [g/cm^3], T = {T} [K]")
  hydrogen_density = gas_density_to_hydrogen_number_density(rho)

  dynamics = setup_dynamics(network, 3000, initial_number_densities,
                            hydrogen_density)
  for timescale in timescales:
    print(f"\tSolving t = {timescale} [s]")
    dynamics.solve(timescale, dynamics.initial_number_densities)
