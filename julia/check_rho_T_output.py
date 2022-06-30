#!/home/sdeshmukh/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


# USAGE: ./check_rho_T_output.py <DIRECTORY>


def read_np_output(infile: str, ncols=2) -> np.ndarray:
  arr = np.loadtxt(infile, delimiter=',')
  # Reshape
  # NOTE:
  # - order is extremely important here! [150, 140, 140], [140, 140, 150], etc
  #   all reshape properly from [150 x 140 x 140]
  density, temperature = arr[:, 0], arr[:, 1]
  density = np.reshape(density, (150, 140, 140))
  temperature = np.reshape(temperature, (150, 140, 140))

  return density, temperature


def main():
  # first command-line argument is path to files
  res_dir = sys.argv[1]
  files = [f"{res_dir}/{f}" for f in os.listdir(res_dir)]

  fig, axes = plt.subplots(1, 2)
  axes[0].set_ylabel(r"$\log \rho$")
  axes[1].set_ylabel("T [K]")
  for i, file_ in enumerate(files):
    density, temperature = read_np_output(file_)
    axes[0].plot(np.mean(np.log10(density), axis=(1, 2)))
    axes[1].plot(np.mean(temperature, axis=(1, 2)))

    print(f"Plotted {i+1} of {len(files)}.")
  plt.show()


if __name__ == "__main__":
  main()
