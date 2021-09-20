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


def normalise_2d(array: np.ndarray) -> np.ndarray:
  # Requires 2D array input
  # Normalises the array between [0, 1]
  norm = np.linalg.norm(array)
  return array / norm


def cofactor_matrix(matrix: np.ndarray) -> np.ndarray:
  C = np.zeros(matrix.shape)
  nrows, ncols = C.shape
  for row in range(nrows):
    for col in range(ncols):
      minor = matrix[np.array(list(range(row))+list(range(row+1, nrows)))[:, np.newaxis],
                     np.array(list(range(col))+list(range(col+1, ncols)))]
      C[row, col] = (-1)**(row+col) * np.linalg.det(minor)
  return C


# -------------------------------------------------------------------------
# Laplacian matrix utilities
# -------------------------------------------------------------------------
def compute_balance(laplacian: np.ndarray) -> np.ndarray:
  # Using Kirchhoff's Matrix Tree theorem, compute the kernel of the Laplacian
  # corresponding to a positive, complex-balanced equilibrium
  # Only need first row of cofactor matrix!
  rho = cofactor_matrix(laplacian)[0]
  return rho
