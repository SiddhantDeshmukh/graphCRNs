from src.reaction import Reaction
from typing import List
import numpy as np


# -------------------------------------------------------------------------
# Creating chemical matrices
# -------------------------------------------------------------------------
def laplacian_matrix(incidence_matrix: np.ndarray,
                     kinetics_matrix: np.ndarray) -> np.ndarray:
  # (m x r) incidence (D) and (r x m) kinetics (K) -> L := -DK (m x m)
  return -incidence_matrix @ kinetics_matrix


def complex_incidence_matrix(complexes: List,
                             reactions: List[Reaction]) -> np.ndarray:
  # Create (c x r) incidence matrix linking complexes to reactions
  complex_incidence_matrix = np.zeros((len(complexes), len(reactions)))

  for i, complex in enumerate(complexes):
    for j, rxn in enumerate(reactions):
      if complex == rxn.reactant_complex:
        complex_incidence_matrix[i, j] = -1
      if complex == rxn.product_complex:
        complex_incidence_matrix[i, j] = 1

  return complex_incidence_matrix


def complex_composition_matrix(species: List,
                               complexes: List) -> np.ndarray:
  # Create matrix describing the composition of each complex 'c' based on its
  # constituent metabolites 'm' (m x c)
  complex_composition_matrix = np.zeros((len(species, len(complexes))))
  for i, species in enumerate(species):
    for j, complex in enumerate(complexes):
      split_complex = [c.strip() for c in complex.split("+")]
      if species in split_complex:
        complex_composition_matrix[i, j] = 1

  return complex_composition_matrix
