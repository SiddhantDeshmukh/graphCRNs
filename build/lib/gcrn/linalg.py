# Linear algebra routines
import numpy as np


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
