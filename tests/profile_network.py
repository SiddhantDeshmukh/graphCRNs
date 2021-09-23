# Profile a CRN to investigate species, reactant-product balance, pathfinding
# and timescales
from src.network import Network
network_dir = '../res'
# network_file = f"{network_dir}/react-solar-umist12"
network_file = f"{network_dir}/mp_cno.ntw"
network = Network.from_krome_file(network_file)

species_balance = network.network_species_count()
print("\n".join([f"{key}: {value}" for key, value in species_balance.items()]))
