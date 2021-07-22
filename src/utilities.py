import numpy as np


def incidence_to_adjacency(incidence_matrix: np.ndarray,
                           remove_self_loops=False) -> np.ndarray:
  # From an incidence matrix, create and return the respective adjacency matrix
  # Useful since networkx graphs can be created with adjacency matrices,
  # but not incidence matrices
  adjacency_matrix = (np.dot(incidence_matrix,
                             incidence_matrix.T) > 0).astype(int)
  if remove_self_loops:
    np.fill_diagonal(adjacency_matrix, 0)

  return adjacency_matrix
