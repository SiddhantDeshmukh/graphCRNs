# Collection of functions to profile network for balance and complexity
from typing import List
from gcrn.helper_functions import sort_dict, species_counts_from_rxn_counts
from gcrn.network import Network

from gcrn.pathfinding import PointPaths


def most_travelled_pair_paths(points_list: List[PointPaths],
                              source_target_pairs: List):
  # NOTE:
  # Can we recover the source target pairs list from the PointPaths? Keys are
  # the same!
  # Find the most travelled pathways across entire regime of PointPaths
  # instances by source-target pair
  # i.e. for each source-target pair, which pathways were the most travelled
  # across the entire density-temperature-time regime
  # (parametrised by PointPaths)?
  key_paths = {p: {} for p in source_target_pairs}
  for pair in source_target_pairs:
    for point in points_list:
      for path in point.unique_paths[pair]:
        try:
          key_paths[pair][path] += 1
        except KeyError:
          key_paths[pair][path] = 1

  return key_paths


def most_travelled_pair_idx_paths(points_list: List[PointPaths],
                                  source_target_pairs: List):
  # NOTE:
  # Can we recover the source target pairs list from the PointPaths? Keys are
  # the same!
  # Find the most travelled pathways across entire regime of PointPaths
  # instances by source-target pair
  # i.e. for each source-target pair, which pathways were the most travelled
  # across the entire density-temperature-time regime
  # (parametrised by PointPaths)?
  key_paths = {p: {} for p in source_target_pairs}
  for pair in source_target_pairs:
    for point in points_list:
      for path in point.rxn_idx_paths[pair]:
        try:
          key_paths[pair][path] += 1
        except KeyError:
          key_paths[pair][path] = 1

  return key_paths


def total_counts_across_paths(points_list: List[PointPaths], network: Network):
  # Count reactions (and hence species) across all paths to determine the most
  # important ones
  total_counts = {}
  for r in points_list:
    counts = {}
    for k, v in r.all_rxn_counts.items():
      try:
        counts[k] += v
      except KeyError:
        counts[k] = v

    for k, v in counts.items():
      try:
        total_counts[k] += v
      except KeyError:
        total_counts[k] = v

  # Most important species by looking up reactants of most important reactions
  species_counts = species_counts_from_rxn_counts(total_counts, network)
  # Sort by values
  total_counts = sort_dict(total_counts)
  species_counts = sort_dict(species_counts)

  return total_counts, species_counts


def most_important_species_by_pairs(points_list: List[PointPaths],
                                    source_target_pairs: List, network: Network):
  # Count most important species by source-target pair across entire
  # points_list
  pair_counts = {p: {} for p in source_target_pairs}
  for pair in source_target_pairs:
    for point in points_list:
      for k, v in point.pair_rxn_counts[pair].items():
        try:
          pair_counts[pair][k] += v
        except KeyError:
          pair_counts[pair][k] = v

  # Species count by pairs
  pair_species_counts = {k: sort_dict(species_counts_from_rxn_counts(v, network))
                         for k, v in pair_counts.items()}

  return pair_species_counts
