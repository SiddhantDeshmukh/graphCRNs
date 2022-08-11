import numpy as np
from scipy.interpolate import splrep, splev


def ratio(arr: np.ndarray):
  # Ratios of consecutive elements
  return np.exp(-np.diff(np.log(arr)))


def find_equilibrium(evolution_times: np.ndarray,
                     number_densities: np.ndarray, threshold=1e-6):
  # Compute the time at which each chemical species is in equilibrium, defined
  # by the relative change being less than the (relative) threshold
  # 'evolution_times' must be a 1D array with dimension (time)
  # 'number_densities' must be a 2D array with dimensions (time, species)
  eqm_times = np.empty(shape=number_densities.shape[1])
  eqm_number_densities = np.empty(shape=number_densities.shape[1])
  for i in range(number_densities.shape[1]):  # over species
    try:
      threshold_idx = np.where(
          np.abs(1. - ratio(number_densities[:, i])) >= threshold)[0][-1]
    except IndexError:  # condition not fulfilled; species always in eqm
      threshold_idx = 0
    eqm_times[i] = evolution_times[threshold_idx]
    eqm_number_densities[i] = number_densities[threshold_idx, i]

  return eqm_times, eqm_number_densities


def find_equilibrium_interpolate(evolution_times: np.ndarray,
                                 number_densities: np.ndarray, threshold=1e-6):
  # Compute the time at which each chemical species is in equilibrium, defined
  # by the relative change being less than the (relative) threshold
  # 'evolution_times' must be a 1D array with dimension (time)
  # 'number_densities' must be a 2D array with dimensions (time, species)
  eqm_times = np.empty(shape=number_densities.shape[1])
  eqm_number_densities = np.empty(shape=number_densities.shape[1])
  # We interpolate each species trend first to find a more preciese equilibrium
  # time
  for i in range(number_densities.shape[1]):  # over species
    try:
      # Initial guess is solver time
      threshold_idx = np.where(
          np.abs(1. - ratio(number_densities[:, i])) >= threshold)[0][-1]

      # Sample points around this time

    except IndexError:  # condition not fulfilled; species always in eqm
      threshold_idx = 0
    eqm_times[i] = evolution_times[threshold_idx]
    eqm_number_densities[i] = number_densities[threshold_idx, i]

  return eqm_times, eqm_number_densities
