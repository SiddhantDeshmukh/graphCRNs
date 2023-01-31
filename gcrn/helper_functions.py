from __future__ import annotations
from itertools import product
from sadtools.utilities.chemistryUtilities import log_abundance_to_number_density
from sadtools.utilities.abu_tools import load_abu, get_abundances
from typing import Dict, List
import numpy as np
from gcrn.network import Network
from gcrn.reaction import Reaction
import re

mass_hydrogen = 1.67262171e-24  # g


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


def setup_number_densities(abundance_dict: Dict,
                           hydrogen_density: float, network: Network):
  number_densities = np.zeros(len(network.species))
  # Index number densities same as network.species
  for i, s in enumerate(network.species):
    n = log_abundance_to_number_density(np.log10(abundance_dict[s]),
                                        np.log10(hydrogen_density.value))
    number_densities[i] = n

  return number_densities


def number_densities_from_abundances(abundances: Dict,
                                     gas_density: float, species: List[str]):
  # Determine hydrogen density
  abundance_values = np.array([v for v in abundances.values()])
  percentage_hydrogen = 10**abundances['H'] / np.sum(10**abundance_values)
  hydrogen_density = gas_density / (percentage_hydrogen * mass_hydrogen)
  number_densities = np.zeros(len(species))
  # Index number densities same as network.species
  for i, s in enumerate(species):
    number_densities[i] = 10**(np.log10(hydrogen_density) + abundances[s] - 12)

  return number_densities


# ----------------------------------------------------------- ------------------
# Timescale functions
# -----------------------------------------------------------------------------


def jacobian_timescales(jacobian: np.ndarray) -> float:
  # Return the characteristic timescale of the longest mode from the Jacobian
  # eigenvalue decomposition
  # If timescale exceeds bounds, set it to these
  # MIN_TIME = 1e-9
  # MAX_TIME = 1e6

  # Jacobian analysis
  eigenvalues, eigenvectors = np.linalg.eig(jacobian)

  # Calculate timescales
  timescales = [1 / np.abs(e) if e < 0 else 0 for e in eigenvalues]
  # neg_evals = [e for e in eigenvalues if e < 0]
  # timescales = [1 / np.abs(e) for e in neg_evals]

  # Filter timescales to make sure they are within set bounds
  # Have to do this because otherwise KROME can fail...but why? I think
  # it has to do with computing negative number densities afterwards, but
  # I would have thought there would be some check for this...
  # timescales = [time for time in timescales if time >
  #               MIN_TIME and time < MAX_TIME]
  # max_timescale = max(timescales)  # representative for eqm
  # print(", ".join([f"{t:1.2e}" for t in timescales]))
  # print(", ".join([f"{e:1.2e}" for e in eigenvalues]))

  return timescales


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
