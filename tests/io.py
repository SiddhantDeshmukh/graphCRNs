# Test network I/O
from src.network import Network
from os import system


res_dir = '../res/'
out_dir = '../out/test/networks'
network_file = f"{res_dir}/co_test.ntw"

system(f"mkdir -p {out_dir}")

network = Network.from_krome_file(network_file)
network.to_krome_format(f"{out_dir}/test_co.ntw")
network.to_cobold_format(f"{out_dir}/test_co.dat")
