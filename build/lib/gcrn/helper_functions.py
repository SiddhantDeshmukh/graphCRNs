from itertools import product
from sadtools.utilities.chemistryUtilities import log_abundance_to_number_density
from sadtools.utilities.abu_tools import load_abu, get_abundances
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from gcrn.network import Network
from gcrn.reaction import Reaction
import re


def sort_dict(d: Dict, reverse=True):
  # Sort dictionary by values
  return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


def reaction_from_complex(source_complex: str, target_complex: str,
                          network: Network) -> Reaction:
  # TODO:
  # network method!
  # Find the first reaction corresponding to 'source_complex -> target_complex'
  # in 'network'. Used in pathfinding to relate paths back to reactions.
  for rxn in network.reactions:
    if rxn.reactant_complex == source_complex and rxn.product_complex == target_complex:
      return rxn

  return None


def reaction_from_idx(idx: int, network: Network) -> Reaction:
  # TODO:
  # network method!
  # Use the reaction index to find the corresponding Reaction in network
  for rxn in network.reactions:
    if rxn.idx == idx:
      return rxn

  return None


def rxn_idxs_from_path(path: str) -> List[str]:
  # From a path e.g. 'C -> 75 -> 244 -> CO' return indices of reactions
  # in the order they appear [75, 244]
  # For a single-path reaction, returns a list of one entry, e.g.
  # 'C -> 76 -> CO': [76]
  return re.findall(r"(?<=-> )\d+(?= ->)", path)


def count_rxns_for_paths(paths: List) -> Dict:
  # Count the occurrences of reactions in the specified dictionary,
  # returning a dictionary {rxn_idx: count}
  counts = {}
  for path in paths:
    rxn_idxs = rxn_idxs_from_path(path)
    for idx in rxn_idxs:
      if not idx in counts:
        counts[idx] = 1
      else:
        counts[idx] += 1

  return counts


# ----------------------------------------------------------------------------
# Functions for counting
# ----------------------------------------------------------------------------
def count_reactant_instances(network, species: str) -> int:
  # For a specified species, count the number of times it appears as a
  # reactant and return the count
  count = 0
  for rxn in network.reactions:
    if species in rxn.reactants:
      count += 1
  return count


def count_product_instances(network, species: str) -> int:
  # For a specified species, count the number of times it appears as a
  # product and return the count
  count = 0
  for rxn in network.reactions:
    if species in rxn.products:
      count += 1
  return count


def network_species_count(network) -> Dict:
  # Count the occurrence of reactant/product occurrences of species in
  # network of reactions. Only counts each species once per reaction,
  # e.g. H + H + H -> H2 yields 1 appearance of H on LHS and 1 appearance
  # of H2 on RHS.
  # TODO:
  # Isn't this just from the adjacency matrix? Try to get it from that
  # key (str, species) to List(int, int; reactant_count, product_count)
  counts = {}
  for s in network.species:
    reactant_count = network.count_reactant_instances(s)
    product_count = network.count_product_instances(s)
    counts[s] = {"R": reactant_count, "P": product_count}

  return counts


def species_counts_from_rxn_counts(rxn_counts: Dict, network: Network) -> Dict:
  # From a dict {rxn_idx: count}, find the associated reaction's reactant
  # species and associate the count of the reaction with the count of the
  # species
  species_counts = {}
  for k, v in rxn_counts.items():
    reactants = reaction_from_idx(k, network).reactants
    for r in reactants:
      if r in species_counts:
        species_counts[r] += reactants.count(r) * v
      else:
        species_counts[r] = reactants.count(r) * v

  return species_counts


def count_rxns_by_pairs(rxn_idx_paths: Dict) -> Dict:
  # Count the occurrences of reactions in the specified dictionary of paths,
  # returning a dictionary of dictionaries {pair: {rxn_idx: count}}
  rxn_counts = {}
  for key, paths in rxn_idx_paths.items():
    rxn_counts[key] = count_rxns_for_paths(paths)

  return rxn_counts


def count_all_rxns(rxn_idx_paths: Dict) -> Dict:
  # Count the occurrences of reactions in the specified dictionary of paths,
  # returning a dictionary {rxn_idx: count}
  rxn_counts = {}
  for key in rxn_idx_paths.keys():
    for path in rxn_idx_paths[key]:
      rxn_idxs = rxn_idxs_from_path(path)
      for idx in rxn_idxs:
        if not idx in rxn_counts:
          rxn_counts[idx] = 1
        else:
          rxn_counts[idx] += 1

  return rxn_counts


def enumerated_product(*args):
  # TODO:
  # Move to utilities and reference
  # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
  yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def setup_number_densities(abundance_dict: Dict, hydrogen_density: float,
                           network: Network):
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(abundance_dict[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n

  return number_densities


# def setup_dynamics(network: Network, temperature: float,
#                    initial_abundances: Dict, hydrogen_density: float):
#   # Initialise number densities
#   number_densities = np.zeros(len(network.species))
#   # Index number densities same as network.species
#   for i, s in enumerate(network.species):
#     n = log_abundance_to_number_density(np.log10(initial_abundances[s]),
#                                         np.log10(hydrogen_density.value))
#     number_densities[i] = n
#   # Initialise dynamics at given temperature
#   dynamics = NetworkDynamics(
#       network, number_densities, temperature=temperature)

#   return dynamics


# def solve_dynamics(dynamics: NetworkDynamics, times: List,
#                    limit_rates=False) -> np.ndarray:
#   # TODO:
#   # Refactor 'dynamics' to use dynamics.number_densities instead of passing in
#   solver_kwargs = {
#       'min_step': None,
#       # 'max_step': time / 1e9
#       'max_step': None
#   }
#   final_number_densities = dynamics.solve(times, dynamics.number_densities,
#                                           create_jacobian=False,
#                                           limit_rates=limit_rates,
#                                           **solver_kwargs)[0]
#   # print(f"Done {i+1} time of {len(times)}")

#   final_number_densities = np.array(final_number_densities).T
#   return final_number_densities


# def run_and_plot(temperatures: List, times: List, network: Network,
#                  filename: str, initial_number_densities: Dict,
#                  hydrogen_density: float, plot_species: List,
#                  limit_rates=False):
#   fig, axes = plt.subplots(3, 3, figsize=(10, 12), sharex=True)
#   # for i, (density, temperature) in enumerate(product(densities, temperatures)):
#   for i, temperature in enumerate(temperatures):
#     idx_x = i // 3
#     idx_y = i % 3
#     dynamics = setup_dynamics(network, temperature,
#                               initial_number_densities, hydrogen_density)
#     number_densities = solve_dynamics(dynamics, times, limit_rates=limit_rates)

#     print(f"Done T = {temperature} K: {i+1} of {len(temperatures)}")

#     total_number_density = np.sum(number_densities, axis=0)
#     # Total number density
#     # axes[idx_x, idx_y].plot(np.log10(times), np.log10(total_number_density),
#     #                         label="Total")
#     # axes[idx_x, idx_y].plot(np.log10(times), total_number_density / total_number_density[0],
#     #                         label="Total")
#     # axes[idx_x, idx_y].set_ylim(17, 18)
#     # hydrogen_number_density = number_densities[network.species.index('H')]
#     for s, n in zip(network.species, number_densities):
#       if s in plot_species:
#         # TODO:
#         # Add bool switch to plot either abundance or number densities
#         # if s in ['C', 'O', 'H', 'M']:
#         #   continue
#         # Number density ratio
#         axes[idx_x, idx_y].plot(np.log10(times), np.log10(n/total_number_density),
#                                 label=s, ls='-')
#         # axes[idx_x, idx_y].plot(np.log10(times), np.log10(n),
#         #                         label=s, ls='-')
#         # Abundance
#         # abundance = 12 + np.log10(n / hydrogen_number_density)
#         # axes[idx_x, idx_y].set_ylim(-13, 13)
#         # axes[idx_x, idx_y].set_ylim(-2, 13)
#         # axes[idx_x, idx_y].plot(np.log10(times), abundance,
#         #                         label=s,  ls='-')
#         # Plot problem area where M abundance is higher than 11
#         # if s == 'M':
#         #   problem_mask = (abundance > 11.005)
#         #   if len(times[problem_mask]) > 0:
#         #     axes[idx_x, idx_y].axvline(np.log10(times)[problem_mask][0],
#         #                                c='k', ls='--')

#     axes[idx_x, idx_y].set_title(f"{temperature} K")
#     axes[idx_x, idx_y].legend()
#     if idx_x == 2:
#       axes[idx_x, idx_y].set_xlabel("log time [s]")
#     if idx_y == 0:
#       axes[idx_x, idx_y].set_ylabel("log number density")
#       # axes[idx_x, idx_y].set_ylabel("log number density ratio")
#       # axes[idx_x, idx_y].set_ylabel("Abundance")

#   plt.savefig(filename, bbox_inches="tight")
#   plt.show()


# def run_on_density_temperature_grid(density: np.ndarray,
#                                     temperature: np.ndarray,
#                                     timescales: List[float], network: Network,
#                                     initial_abundances: Dict):
#   # Run dynamics for specified Network and initial abundances for given
#   # timescale foreach (density, temperature) point.
#   # Returns a 4D array (density, temperature, time, species)
#   species = network.species
#   number_densities = np.zeros(
#       shape=(len(density), len(temperature), len(timescales), len(species)))
#   for (i, j), (rho, T) in enumerated_product(density, temperature):
#     hydrogen_density = gas_density_to_hydrogen_number_density(rho)
#     dynamics = setup_dynamics(network, T, initial_abundances, hydrogen_density)
#     n = solve_dynamics(dynamics, timescales)
#     number_densities[i, j] = n.T.copy()

#   return number_densities


# def density_temperature_grid_with_pathfinding(density: np.ndarray,
#                                               temperature: np.ndarray,
#                                               timescales: List[float],
#                                               network: Network,
#                                               initial_abundances: Dict,
#                                               sources: List, targets: List,
#                                               cutoff=5, max_paths=100):
#   # Run dynamics for specified network and initial abundances foreach
#   # (density, temperature, timescale) point and compute PointPaths instances
#   # at each point
#   species = network.species
#   number_densities = np.zeros(shape=(len(density), len(temperature),
#                                      len(timescales), len(species)))
#   all_point_paths: np.ndarray[PointPaths] = np.empty(shape=(len(density),
#                                                             len(temperature),
#                                                             len(timescales)),
#                                                      dtype=object)
#   for (i, j, k), (rho, T, t) in enumerated_product(density, temperature, timescales):
#     hydrogen_density = gas_density_to_hydrogen_number_density(rho)
#     dynamics = setup_dynamics(network, T, initial_abundances, hydrogen_density)
#     n = dynamics.solve(t, dynamics.number_densities,
#                        create_jacobian=False,
#                        limit_rates=False)[0]
#     number_densities[i, j, k] = n.T.copy()
#     all_point_paths[i, j, k] = deepcopy(compute_point_paths(network, sources, targets,
#                                                             rho, T, t,
#                                                             cutoff=cutoff,
#                                                             max_paths=max_paths))

#   return number_densities, all_point_paths

# ----------------------------------------------------------- ------------------
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
