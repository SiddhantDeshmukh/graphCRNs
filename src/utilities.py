from typing import List
from src.reaction import Reaction
import numpy as np
import re


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


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------
def determine_krome_format(reaction: Reaction) -> str:
  # Determine the KROME format "@format:..." string for a given reaction
  format_str = "@format:"
  if reaction.idx:
    format_str += "idx,"

  format_str += ','.join(["R" for r in reaction.reactants]) + ","
  format_str += ','.join(["P" for p in reaction.products]) + ","
  format_str += "rate,"

  if reaction.min_temperature or reaction.max_temperature:
    format_str += "Tmin,Tmax,limit,"

  # Remove trailing comma
  return format_str.rstrip(',')


def list_to_krome_format(reactions: List[Reaction]) -> str:
  # Write a list of reactions to a KROME-readable string
  output = ""
  old_format = ""
  for rxn in reactions:
    format_str = determine_krome_format(rxn)
    if old_format != format_str:  # new format
      old_format = format_str
      output += f"\n## {len(rxn.reactants)} reactants, {len(rxn.products)} products\n"
      output += f"{format_str}\n"

    output += f"{rxn.krome_str()}\n"

  return output


def to_fortran_str(quantity: float, fmt='.2e') -> str:
  output_quantity = f"{quantity:{fmt}}"
  output_quantity = output_quantity.replace('e', 'd').replace('+', '')

  return output_quantity


def constants_from_rate(rate: str) -> List:
  # Read Arrhenius rate and return alpha, beta, gamma, i.e.
  # r(T) = alpha * (Tgas/300)**beta * exp(-gamma / Tgas)
  def constants_from_part(part: str):
    alpha, beta, gamma = None, None, None
    if part.strip()[0].isdigit():
      print(part.strip()[0])
      alpha = part
    elif '^' in part:
      print(part)
      beta = re.findall(r'[\(0-9\)]', part.split('^')[-1])
    elif part.strip().startswith('exp'):
      print(part)
      gamma = float(part.split('/')[0].strip()[4:])

    return alpha, beta, gamma

  alpha, beta, gamma = None, None, None
  # Change '**' to '^' for a unique identifier
  rate = rate.replace('**', '^')
  if '*' in rate:
    # TODO:
    # This won't work since it'll split on '**'
    parts = rate.split("*")
    print(parts)
    for part in parts:
      alpha, beta, gamma = constants_from_part(part)
  else:
    alpha, beta, gamma = constants_from_part(rate)

  print(alpha, beta, gamma)

  return alpha, beta, gamma

# -------------------------------------------------------------------------
# List utilities
# -------------------------------------------------------------------------


class PaddedList(list):
  def ljust(self, n, fill_value=''):
    return self + [fill_value] * (n - len(self))


def pad_list(lst: List, n: int, fill_value='') -> PaddedList:
  return PaddedList(lst).ljust(n, fill_value=fill_value)
