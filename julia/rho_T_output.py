# Create a density-temperature output file for the Julia solver
from itertools import product
from os import system
from typing import Dict, List
import numpy as np
from uio_tools.uio_loader import UIOLoader
import matplotlib.pyplot as plt
import pandas as pd
from sadchem.io.interface.gcrn_uio import load_corresponding_snapshot


def read_np_output(infile: str, ncols=2) -> np.ndarray:
  arr = np.loadtxt(infile, delimiter=',')
  # Reshape
  # NOTE:
  # - order is extremely important here! [150, 140, 140], [140, 140, 150], etc
  #   all reshape properly from [150 x 140 x 140]
  arr = np.reshape(arr, (ncols, 150, 140, 140))
  return arr


def read_number_densities(infile: str) -> Dict:
  df = pd.read_csv(infile, delimiter=',')
  # Reshape to 3D
  return {k: np.reshape(df[k].values, (150, 140, 140)) for k in df.keys()}


def write_array(outfile: str, arr: np.ndarray):
  # Assumes 'arr' is 2D, first column 'density', second column 'temperature'
  np.savetxt(outfile, arr, delimiter=',', header='density,temperature')


def plot_differences(model, gcrn_n):
  # TODO:
  # - convert to abundances!
  fig, axes = plt.subplots(1, 2, sharey=True)
  quc_keys = [f"quc00{i}" for i in range(1, 9)]
  quc_names = [model.get_key_name_units(k)[0] for k in quc_keys]
  log_tau = np.log10(np.mean(model['tau'], axis=(1, 2)))
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colours = prop_cycle.by_key()['color']
  for i, qkey in enumerate(quc_keys):
    species = quc_names[i]
    if species == 'metal':
      species = 'M'
    log_n = np.log10(np.mean(model[qkey], axis=(1, 2)))
    axes[0].plot(log_n, log_tau, c=colours[i], label=species, ls='none',
                 marker='o')
    log_gcrn_n = np.log10(np.mean(gcrn_n[species], axis=(1, 2)))
    axes[0].plot(log_gcrn_n, log_tau, c=colours[i], ls='-', marker='o',
                 mfc='none')

    axes[1].plot(log_n - log_gcrn_n, log_tau, c=colours[i], marker='o',
                 mfc='none')
    # axes[1].axvline(1., c='k', ls=':')

  axes[0].legend()
  axes[0].invert_yaxis()
  axes[0].set_xlabel(r"$\log{\mathrm{n}}$ [cm$^{-3}$]")
  axes[1].set_xlabel("log(CHEM) - log(Julia)")
  axes[1].set_ylabel(r"$\log{\tau}$")

  return fig, axes


def save_subsample_snapshots(loader: UIOLoader, num_snaps_out: int,
                             out_dir: str, snap_out_idxs=None,
                             num_snap_skip=1):
  def reset(loader):
    loader.load_first_model()
    loader.current_model.first_snapshot()

  current_snap_num = 1
  written_files = 0
  total_snaps = loader.num_models * \
      loader.current_model.final_snap_idx - num_snap_skip
  snap_out_frequency = total_snaps // num_snaps_out
  if not snap_out_idxs:
    snap_out_idxs = list(range(num_snap_skip, total_snaps, snap_out_frequency))
  system(f"mkdir -p {out_dir}")
  for snap_num in snap_out_idxs:
    model = load_corresponding_snapshot(loader, snap_num)
    print(f"Writing snap {snap_num}")
    density, temperature = model['rho'], model['temperature']
    arr = np.array([density.flatten(), temperature.flatten()]).T
    current_snap_num = str(snap_num).zfill(3)
    outfile = f"rho_T_{current_snap_num}.csv"
    write_array(f"{out_dir}/{outfile}", arr)
    written_files += 1
    print(f"{written_files}/{num_snaps_out} files written to {out_dir}.")

    if written_files == num_snaps_out:
      reset(loader)
      return

  reset(loader)


def main():
  write_subsample = True
  test_case = True  # equivalent to 'precompile' option in the Julia version
  plot = False

  PROJECT_DIR = "/home/sdeshmukh/Documents/graphCRNs/julia"
  res_dir = f"{PROJECT_DIR}/res"
  out_dir = f"{PROJECT_DIR}/out"
  model_dir = "/media/sdeshmukh/Crucial X6/cobold_runs/chem"
  model_dir += "/d3t63g40mm30c20n20o04chem2"
  loader = UIOLoader(model_dir)
  num_snaps_out = 20  # number of equidistant snapshots to pick
  num_snap_skip = 10  # number of snaps to skip when choosing output

  # To redo!
  am_snap_out_idxs = [45, 73, 108, 136]
  ac_snap_out_idxs = [28, 46, 64, 82, 100, 118]

  if write_subsample:
    save_subsample_snapshots(loader, num_snaps_out,
                             f"{res_dir}/{loader.current_model.id}",
                             snap_out_idxs=ac_snap_out_idxs,
                             num_snap_skip=num_snap_skip)

  if test_case:
    # Write single
    for i in range(3):
      loader.load_next_model()

    model = loader.current_model
    for i in range(5):
      model.next_snapshot()
    # Sample uniform points from model
    # nz, ny, nx = 10, 5, 7
    nz, ny, nx = 30, 10, 14
    original_shape = model['rho'].shape
    idxs = product(*[np.linspace(0, original_shape[i] - 1, num=n, dtype=int)
                     for i, n in enumerate([nz, ny, nx])])
    density = np.zeros((nz * ny * nx))
    temperature = np.zeros((nz * ny * nx))
    for i, p in enumerate(idxs):
      density[i] = model['rho'][p]
      temperature[i] = model['temperature'][p]

    arr = np.array([density, temperature]).T
    outfile = "rho_T_test.csv"
  else:
    # Read full from model
    density, temperature = model['rho'], model['temperature']
    arr = np.array([density.flatten(), temperature.flatten()]).T
    outfile = "rho_T.csv"

  print(f"Array output shape: {arr.shape}")

  write_array(f"{res_dir}/test/{outfile}", arr)
  print(f"Wrote to {res_dir}/test/{outfile}")
  # reshaped_arr = read_np_output(f"{res_dir}/{outfile}")
  # print(f"Array input shape: {reshaped_arr.shape}")

  if plot:
    number_densities = read_number_densities(f"{out_dir}/catalyst.csv")
    fig, axes = plot_differences(model, number_densities)
    plt.show()


if __name__ == "__main__":
  main()
