from itertools import product
from sadtools.utilities.chemistryUtilities import log_abundance_to_number_density
from sadtools.utilities.abu_tools import load_abu, get_abundances
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from gcrn.network import Network
from gcrn.dynamics import NetworkDynamics


def enumerated_product(*args):
  # TODO:
  # Move to utilities and reference
  # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
  yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def setup_number_densities(number_densities_dict: Dict, hydrogen_density: float,
                           network: Network):
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(number_densities_dict[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n

  return number_densities


def setup_dynamics(network: Network, temperature: float,
                   initial_number_densities: Dict, hydrogen_density: float):
  # Initialise number densities
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(initial_number_densities[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n
  # Initialise dynamics at given temperature
  dynamics = NetworkDynamics(
      network, number_densities, temperature=temperature)

  return dynamics


def solve_dynamics(dynamics: NetworkDynamics, times: List,
                   limit_rates=False) -> np.ndarray:
  final_number_densities = []
  # TODO:
  # Refactor 'dynamics' to use dynamics.number_densities instead of passing in
  for i, time in enumerate(times):
    solver_kwargs = {
        'min_step': None,
        # 'max_step': time / 1e9
        'max_step': None
    }

    final = dynamics.solve(time, dynamics.number_densities,
                           create_jacobian=True,
                           limit_rates=limit_rates,
                           **solver_kwargs)[0]
    final_number_densities.append(final)
    print(f"Done {i+1} time of {len(times)}")

  final_number_densities = np.array(final_number_densities).T
  return final_number_densities


def run_and_plot(temperatures: List, times: List, network: Network,
                 filename: str, initial_number_densities: Dict,
                 hydrogen_density: float,
                 limit_rates=False):
  fig, axes = plt.subplots(3, 3, figsize=(10, 12), sharex=True)
  # for i, (density, temperature) in enumerate(product(densities, temperatures)):
  for i, temperature in enumerate(temperatures):
    idx_x = i // 3
    idx_y = i % 3
    dynamics = setup_dynamics(network, temperature,
                              initial_number_densities, hydrogen_density)
    number_densities = solve_dynamics(dynamics, times, limit_rates=limit_rates)

    print(f"Done T = {temperature} K: {i+1} of {len(temperatures)}")

    total_number_density = np.sum(number_densities, axis=0)
    # Total number density
    # axes[idx_x, idx_y].plot(np.log10(times), np.log10(total_number_density),
    #                         label="Total")
    axes[idx_x, idx_y].plot(np.log10(times), total_number_density / total_number_density[0],
                            label="Total")
    # axes[idx_x, idx_y].set_ylim(17, 18)
    # hydrogen_number_density = number_densities[network.species.index('H')]
    for s, n in zip(network.species, number_densities):
      # TODO:
      # Add bool switch to plot either abundance or number densities
      # if s in ['C', 'O', 'H', 'M']:
      #   continue
      # Number density ratio
      axes[idx_x, idx_y].plot(np.log10(times), np.log10(n/total_number_density),
                              label=s, ls='-')
      # axes[idx_x, idx_y].plot(np.log10(times), np.log10(n),
      #                         label=s, ls='-')
      # Abundance
      # abundance = 12 + np.log10(n / hydrogen_number_density)
      # axes[idx_x, idx_y].set_ylim(-13, 13)
      # axes[idx_x, idx_y].set_ylim(-2, 13)
      # axes[idx_x, idx_y].plot(np.log10(times), abundance,
      #                         label=s,  ls='-')
      # Plot problem area where M abundance is higher than 11
      # if s == 'M':
      #   problem_mask = (abundance > 11.005)
      #   if len(times[problem_mask]) > 0:
      #     axes[idx_x, idx_y].axvline(np.log10(times)[problem_mask][0],
      #                                c='k', ls='--')

    axes[idx_x, idx_y].set_title(f"{temperature} K")
    axes[idx_x, idx_y].legend()
    if idx_x == 2:
      axes[idx_x, idx_y].set_xlabel("log time [s]")
    if idx_y == 0:
      axes[idx_x, idx_y].set_ylabel("log number density")
      # axes[idx_x, idx_y].set_ylabel("log number density ratio")
      # axes[idx_x, idx_y].set_ylabel("Abundance")

  plt.savefig(filename, bbox_inches="tight")
  plt.show()

# -----------------------------------------------------------------------------
# Timescale functions
# -----------------------------------------------------------------------------


def jacobian_timescale(jacobian: np.ndarray) -> float:
  # Return the characteristic timescale of the longest mode from the Jacobian
  # eigenvalue decomposition
  # If timescale exceeds bounds, set it to these
  MIN_TIME = 1e-9
  MAX_TIME = 1e6

  # Jacobian analysis
  eigenvalues, eigenvectors = np.linalg.eig(jacobian)

  # Calculate timescales
  neg_evals = [e for e in eigenvalues if e < 0]
  timescales = [1 / np.abs(e) for e in neg_evals]

  # Filter timescales to make sure they are within set bounds
  # Have to do this because otherwise KROME can fail...but why? I think
  # it has to do with computing negative number densities afterwards, but
  # I would have thought there would be some check for this...
  timescales = [time for time in timescales if time >
                MIN_TIME and time < MAX_TIME]
  max_timescale = max(timescales)  # representative for eqm

  return max_timescale


def nullspace(A: np.ndarray, atol=1e-13, rtol=0, return_decomposition=False):
  # Compute the nullspace of a matrix 'A' (at most 2D)
  A = np.atleast_2d(A)
  u, s, vh = np.linalg.svd(A)

  # Filter singular values based on tolerance
  tol = max(atol, rtol*s[0])
  nnz = (s >= tol).sum()
  ns = vh[nnz:].conj().T

  if return_decomposition:
    return ns, (u, s, vh)
  else:
    return ns


def jacobian_svd(jacobian: np.ndarray):
  # Singular Value Decomposition of Jacobian matrix to determine characteristic
  # timescale and the steady-state abundances (null space)
  ns, (u, s, vh) = nullspace(jacobian, return_decomposition=True)

  print(jacobian.shape)
  print(u.shape, s.shape, vh.shape)
  print(ns.shape)

  # Determine timescale from singular values
  print("Nullspace:")
  print(ns)
  print("Checks")
  for col in ns.T:
    print(np.dot(jacobian, col))
  print("Singular values:")
  print(s)  # all positive
  timescales = [1/value for value in s if value > 0]
  print("Timescales:")
  print(timescales)
  # print(np.dot(jacobian, ns))
  avg_timescale = np.mean(np.array(timescales))
  max_timescale = np.max(np.array(timescales))
  print(f"Average: {avg_timescale:.2e}")
  print(f"Max:     {max_timescale:.2e}")

# -----------------------------------------------------------------------------
# Abundance utilities
# -----------------------------------------------------------------------------


def initialise_abundances(abundance_file: str) -> Dict:
  df = load_abu(abundance_file)
  df.set_index('Element', inplace=True)
  abundances = get_abundances(df, ['C', 'H', 'O'])

  # Add molecular and metal abundances
  abundances['CH'] = -8
  abundances['OH'] = -8
  abundances['H2'] = 8
  abundances['CO'] = -8
  abundances['M'] = 11  # representative metal

  return abundances
