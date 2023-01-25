# Prototyping for photochemical reaction network, will later be a feature of
# GCRN/JCRN to be able to read the required files and calculate photon fluxes
# Assumes local temperature for photon flux, i.e. Planck function based on local
# grid cell temperature
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import List
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from gcrn.network import Network
from abundances import mm30a04_abundances
from sadchem.postprocessing import calculate_number_densities


class Constants:
  def __init__(self):
    self.h = 6.62607015e-34  # J s
    self.c = 299792458  # m/s
    self.kB = 1.3806503e-23  # m^2 kg s^-2 K-1


const = Constants()  # hidden global


def minmax(arr): return np.min(arr), np.max(arr)


def planck_function(const: Constants, temperature: float, wavelengths):
  a = 2.0*const.h*const.c**2
  b = const.h*const.c/(wavelengths*const.kB*temperature)
  intensity = a/((wavelengths**5) * (np.exp(b) - 1.0))
  floor_val = 1e-99
  floor_mask = (intensity <= floor_val)
  intensity[floor_mask] = 0
  # print(minmax(intensity[~floor_mask]))
  # print(wavelengths[floor_mask])
  # exit()
  if (np.any(np.isnan(intensity))):
    nan_mask = np.isnan(intensity)
    intensity[nan_mask] = 0.
  return intensity


def convert_float(val: str):
  try:
    return float(val)
  except ValueError:
    return val


def parse_row(row: str):
  # Parse strings or floats out of row, assumed to be space-separated
  row_data = re.split(r"\s+", row.strip())
  for i, col in enumerate(row_data):
    row_data[i] = convert_float(col)

  return row_data


def extract_table_data(data: List[str]):
  # Read table data from cross-section file and return it
  table_start_pattern = r"^\s+" + "Lambda"
  for i, line in enumerate(data):
    if re.match(table_start_pattern, line):
      # First line of table data, return from here to the end
      headers = parse_row(line)
      table_data = np.array([parse_row(row) for row in data[i+1:-1]])

  df = pd.DataFrame(data=table_data, columns=headers)
  df.set_index('Lambda', inplace=True)
  data_headers = df.columns.values
  if not "Total" in data_headers:
    # Only happens if there is a single cross-sect, column
    # Create a new 'Total' column that is a copy of the single column
    df["Total"] = df[data_headers[0]]
  return df


def read_cross_sects(filepath: str):
  # Read cross-section files downloaded from PHIDrates
  # All files first have some header text describing references, then the
  # tabular data indexed by wavelength 'Lambda'
  # Wavelengths are in Angstroms, cross-sections are in [cm^2]
  with open(filepath, "r") as infile:
    data = infile.readlines()

  df = extract_table_data(data)
  return df


def get_wavelength_range(df: pd.DataFrame):
  # Wavelengths assumed to be in A
  try:
    return np.min(df['Lambda'], np.max(df['Lambda']))
  except KeyError:  # 'Lambda' is now assumed to be the index
    return np.min(df.index), np.max(df.index)


def add_flux_col(df: pd.DataFrame, temperature: float):
  # Add unattenuated blackbody flux as a column to 'df'
  flux = planck_function(const, temperature,
                         wavelengths=df.index.values * 1e-10)
  df['Flux'] = flux
  return df


def compute_photo_rate(df: pd.DataFrame, branches=["Total"]):
  # From cross-sections and 'Flux' columns, compute the integrated
  # photodissociation rate for the given branches
  rates = {}
  for branch in branches:
    rates[branch] = cumulative_trapezoid(df['Flux'] * (df[branch] * 1e-4),
                                         x=df.index.values,
                                         initial=0)

  return rates


def plot_planck(temperature: float, ax=None, wavelengths=None):
  # SI units, temperature in [K], wavelengths in [m]
  if ax is None:
    _, ax = plt.subplots()

  if wavelengths is None:
    wavelengths = np.linspace(1., 1000.) * 1e-10

  ax.plot(wavelengths / 1e-10, planck_function(const, temperature, wavelengths),
          label=f"T = {temperature:.1f} [K]")
  ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
  ax.set_ylabel("Intensity")
  return ax


def plot_df(df: pd.DataFrame, branch="Total"):
  # Plot photo rates, flux and cross-sections against wavelength
  fig, axes = plt.subplots(3, 1)
  axes[0].plot(df.index, df[branch])
  axes[1].plot(df.index, df['Flux'])
  axes[2].plot(df.index, df[f'Rate_{branch}'])

  # Aesthetics
  for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel(r"Wavelength [$\AA$]")

  axes[0].set_yscale("log")
  axes[2].set_yscale("log")

  axes[0].set_ylabel(f"Cross-section ({branch})" r"[cm$^2$]")
  axes[1].set_ylabel(f"Blackbody Flux")
  axes[2].set_ylabel(f"log rate coefficient" + r"[$s^{-1} \AA^{-1}$]")
  return fig, axes


def determine_branch(file_id: str):
  # Only take the direct dissociation into atomic species branch
  if file_id == "c2":
    branch = "C1D/C1D"
  else:
    branch = '/'.join(s.upper() for s in file_id)
  return branch


def main():
  plt.style.use("standard-scientific")
  # photo_dir = "/home/sdeshmukh/Documents/graphCRNs/res/photo"
  # cross_sect_files = [f"{photo_dir}/{f}" for f in os.listdir(photo_dir)]
  # # cross_sect_files = ["/home/sdeshmukh/Documents/graphCRNs/res/photo/ch.txt"]
  # cross_sect_dfs = [read_cross_sects(f) for f in cross_sect_files]
  temperature = 5772.  # Teff for Sun
  rho = 1e-8
  # # # temperature = 7000.  # Teff

  # for file_, df in zip(cross_sect_files, cross_sect_dfs):
  #   cross_sect_id = file_.split("/")[-1].replace(".txt", "")
  #   branch = determine_branch(cross_sect_id)
  #   print(file_.split("/")[-1])
  #   df = add_flux_col(df, temperature)
  #   df[f"Rate_{branch}"] = compute_photo_rate(df, branches=[branch])[branch]
  #   print(np.mean(df[f"Rate_{branch}"]))

  # fig, axes = plot_df(df, branch=branch)

  # fig, ax = plt.subplots(1, 1)
  # temperatures = [3000., 4500., 6000.]
  # for T in temperatures:
  #   plot_planck(T, ax=ax, wavelengths=None)
  # ax.legend()
  # plt.show()

  # Photochem network comparison
  network_dir = "/home/sdeshmukh/Documents/graphCRNs/res/"
  network_kinetic = Network.from_krome_file(f"{network_dir}/cno_fix.ntw")
  network_photo = Network.from_krome_file(f"{network_dir}/photo.ntw")
  species = ["C", "O", "N", "CH", "OH", "C2", "CO", "CN"]
  species.sort()
  networks = [network_kinetic, network_photo]
  network_labels = ["Kinetic", "Photo"]
  fig, axes = plt.subplots(1, 1)
  # Setup
  times = np.logspace(-8, 3, num=2000)
  for network in [network_kinetic, network_photo]:
    network.temperature = temperature
    network.number_densities = calculate_number_densities(mm30a04_abundances,
                                                          np.log10(rho))
  # Solve photochem network
  print(len(network_photo.reactions))
  n = network_photo.solve(times, n_subtime=1)
  colours = []
  for j, s in enumerate(species):
    # Photo in lines
    l = axes.plot(np.log10(times), np.log10(n[:, j]))
    colours.append(l[0].get_color())

  # Solve kinetic-only network
  print(len(network_kinetic.reactions))
  n = network_kinetic.solve(times, n_subtime=1)
  # Kinetic in points
  for j, s in enumerate(species):
    # Photo in lines
    axes.plot(np.log10(times), np.log10(n[:, j]), label=s, ls='none', marker='o',
              mfc=colours[j], c='k')

  axes.legend()
  plt.show()


if __name__ == "__main__":
  main()
