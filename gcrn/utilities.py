from typing import Dict, List
from gcrn.reaction import Reaction
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
def compute_complex_balance(laplacian: np.ndarray) -> np.ndarray:
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
    format_str += "Tmin,Tmax,limit,ref"

  # Remove trailing comma
  return format_str.rstrip(',')


def group_rxns(reactions: List[Reaction], krome_str=False) -> Dict:
  # Group reactions based on number of reactants and products using the KROME
  # format string of each reaction
  # if 'str' is 'rxn', save str(rxn)
  # if 'str' is 'krome', save rxn.krome_str()
  format_dict = {}  # key is format_str, value is list of matching rxns
  for rxn in reactions:
    format_str = determine_krome_format(rxn)
    value_str = rxn.krome_str() if krome_str else str(rxn)
    if format_str in format_dict.keys():
      format_dict[format_str].append(value_str)
    else:
      format_dict[format_str] = [value_str]

  return format_dict


def list_to_krome_format(reactions: List[Reaction]) -> str:
  # Write a list of reactions to a KROME-readable string
  output = ""
  format_dict = group_rxns(reactions, krome_str=True)

  # Write out dictionary
  # TODO:
  # Write out sequentially based on index
  for i, (format_str, krome_strs) in enumerate(format_dict.items()):
    output += f'## {list(format_str).count("R")} reactants, {list(format_str).count("P")} products\n'
    output += f'{format_str}\n'
    output += '\n'.join(rxn for rxn in krome_strs) + "\n"

  return output


def to_fortran_str(quantity: float, fmt='.2e') -> str:
  output_quantity = f"{quantity:{fmt}}"
  output_quantity = output_quantity.replace('e', 'd').replace('+', '')

  return output_quantity


def constants_from_rate(rate: str) -> List:
  # Read Arrhenius rate and return alpha, beta, gamma, i.e.
  # r(T) = alpha * (Tgas/300)**beta * exp(-gamma / Tgas)
  def constants_from_part(part: str, alpha=0., beta=0., gamma=0.):
    if part.strip()[0].isdigit():
      alpha = float(part)
    elif '^' in part:
      beta = float(part.split('^')[-1].replace('(', '').replace(')', ''))
    elif part.strip().startswith('exp'):
      # Note minus sign: 'rate' has '-gamma' in it because of Arrhenius form
      gamma = -float(part.split('/')[0].strip()[4:])

    return alpha, beta, gamma

  alpha, beta, gamma = 0., 0., 0.
  # Change '**' to '^' for a unique identifier
  rate = rate.replace('**', '^')
  if '*' in rate:
    parts = rate.split("*")
    for part in parts:
      alpha, beta, gamma = constants_from_part(
          part, alpha=alpha, beta=beta, gamma=gamma)
  else:
    alpha, beta, gamma = constants_from_part(
        rate, alpha=alpha, beta=beta, gamma=gamma)

  # TODO:
  # Use a regex pattern match to extract alpha, beta, gamma from the string
  # instead of this nested-function-inheritance confusion
  return alpha, beta, gamma

# -------------------------------------------------------------------------
# List utilities
# -------------------------------------------------------------------------


class PaddedList(list):
  def ljust(self, n, fill_value=''):
    return self + [fill_value] * (n - len(self))


def pad_list(lst: List, n: int, fill_value='') -> PaddedList:
  return PaddedList(lst).ljust(n, fill_value=fill_value)
