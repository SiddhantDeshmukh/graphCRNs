# Test network I/O
from gcrn.network import Network
from gcrn.helper_functions import reaction_from_idx
from os import system


res_dir = '../res/'
out_dir = '../out/networks'
network_file = f"{res_dir}/cno.ntw"

system(f"mkdir -p {out_dir}")

network = Network.from_krome_file(network_file)
network.to_krome_format(f"{out_dir}/cno.ntw")
network.to_cobold_format(f"{out_dir}/chem_cno.dat")
cobold_net = network.from_cobold_file(f'{out_dir}/chem_cno.dat')
