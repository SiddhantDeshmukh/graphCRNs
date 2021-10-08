# Test network I/O
from src.network import Network
from os import system


res_dir = '../res/'
out_dir = '../out/networks'
network_file = f"{res_dir}/solar_co_w05.ntw"

system(f"mkdir -p {out_dir}")

network = Network.from_krome_file(network_file)
network.to_krome_format(f"{out_dir}/solar_co_w05.ntw")
network.to_cobold_format(f"{out_dir}/chem.dat")
