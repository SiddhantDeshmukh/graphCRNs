from src.utilities import cofactor_matrix
from src.dynamics import NetworkDynamics
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pylab import draw, draw_networkx_edge_labels
from networkx.drawing.nx_pydot import to_pydot
from src.network import Network
import pydot
import numpy as np
from scipy.linalg import null_space

# # Simple network of a reversible reaction
# # H + CH <-> C + H2
# # k_forward = 5
# # k_backward = 3
# reactants = ['H', 'CH']
# products = ['C', 'H2']
# forward_rate = 5
# reverse_rate = 3

# forward_reaction = Reaction(
#     reactants, products, rate_expression=f"{forward_rate}")
# reverse_reaction = Reaction(
#     products, reactants, rate_expression=f"{reverse_rate}")

# network = Network([forward_reaction, reverse_reaction])
# # Move kinetics matrix and Laplacian matrix into 'Network'
# kinetics_matrix = np.zeros((2, 2))
# for i, rxn in enumerate(network.reactions):
#   for j, complex in enumerate(network.complexes):
#     if complex == rxn.reactant_complex:
#       kinetics_matrix[i, j] = rxn.evaluate_rate_expression(300)
#     else:
#       kinetics_matrix[i, j] = 0

# laplacian_matrix = -network.complex_incidence_matrix @ kinetics_matrix

# # Determine RHS vector Exp(Z.T Ln(x)), where 'x' is number densities
# x = [1., 3., 2., 4.]  # strictly non-negative
# rhs = np.exp(network.complex_composition_matrix.T.dot(np.log(x)))
# Z, D, K = network.complex_composition_matrix, network.complex_incidence_matrix, kinetics_matrix
# dynamics_rhs = Z @ D @ K.dot(rhs)

# print(x)
# print(network.species)
# print(dynamics_rhs)

# Check this by hand!
# Also make sure the list indices line up as expected with the array

# print(network.species_incidence_matrix.shape)
# print(kinetics_matrix)
# print(network.complex_incidence_matrix)
# print(laplacian_matrix)


def is_square(matrix: np.ndarray) -> bool:
  # Checks if a matrix is square. Must be a 2D array
  return matrix.shape[0] == matrix.shape[1]


def is_identity(matrix: np.ndarray) -> bool:
  # Checks if a matrix is the identity matrix. Must be a 2D array
  return (matrix == np.eye(matrix.shape[0])).all()


def is_orthogonal(matrix: np.ndarray) -> bool:
  # Checks if a matrix is orthogonal. Must be a 2D array
  return is_identity(matrix @ matrix.T)


if __name__ == "__main__":
  # Initialising network
  # krome_file = '../res/react-co-solar-umist12'
  # krome_file = '../res/ring-reaction'
  # krome_file = '../res/quad-ring-reaction'
  # krome_file = '../res/T-network'
  krome_file = '../res/L-network'
  # krome_file = '../res/diamond-network'
  network = Network.from_krome_file(krome_file)

  print(f"{len(network.species)} Species")
  # print(network.species)
  print(f"{len(network.complexes)} Complexes")
  # print(network.complexes)
  print(f"{len(network.reactions)} Reactions")
  # print("Rates")
  # for rxn in network.reactions:
  #   print(rxn.rate)

  to_pydot(network.species_graph).write_png("./species.png")
  to_pydot(network.complex_graph).write_png("./complex.png")

  # Check matrices
  print("Adjacency matrices")
  print(f"Species: {nx.to_numpy_array(network.species_graph).shape}")
  print(f"Complex: {nx.to_numpy_array(network.complex_graph).shape}")

  print("Incidence matrices")
  print(f"Species:        {network.species_incidence_matrix.shape}")
  print(f"Complex:        {network.complex_incidence_matrix.shape}")
  print(f"Composition:    {network.complex_composition_matrix.shape}")
  print(f"Stoichiometric: {network.stoichiometric_matrix.shape}")
  print(f"Kinetics:       {network.complex_kinetics_matrix.shape}")
  print(f"Laplacian:      {network.complex_laplacian.shape}")

  # Dynamics
  # initial_number_densities = {
  #     "H": 1e12,
  #     "H2": 1e-4,
  #     "OH": 1e-12,
  #     "C": 10**(8.39),
  #     "O": 10**(8.66),
  #     "CH": 1e-12,
  #     "CO": 1e-12,
  #     "M": 1e11,
  # }

  # initial_number_densities = {
  #     "A": 2,
  #     "B": 3,
  #     "C": 4,
  #     "D": 5
  # }

  # dynamics = NetworkDynamics(network, initial_number_densities)
  # print("Dynamics")
  # # print(dynamics.network.species)
  # print(f"x_0   = {dynamics.initial_number_densities}")
  # print(f"x_dot = {dynamics.dynamics_vector}")

  # Equilibrium
  # Complex-balanced equilibrium exists if Dv(x*) = 0
  # Species-balanced equilibrium exists if Sv(x*) = 0 (?)
  # Since Dv(x*) = -LExp(...), we can check nullspace of L for complex-balanced
  # and nullspace of SK for species-balanced
  # This gives negative answers, reckon I should multiply by -1!
  # I think it's because the dynamics equation will have -L, not L, so we need
  # to multiply by -1 to ensure all concentrations >= 0
  complex_nullspace = -null_space(network.complex_laplacian.T)

  species_laplacian = -network.stoichiometric_matrix @ network.complex_kinetics_matrix
  species_nullspace = -null_space(species_laplacian.T)

  print("Equilibria nullspaces")
  print(f"Complex: {complex_nullspace.shape}")
  # print(network.laplacian_matrix)
  print(f"Species: {species_nullspace.shape}")
  # print(species_laplacian)
  print(species_nullspace)

  # Nullspace is Exp(Z^T Ln(x)) := y
  # Need to invert this relationship to get 'x'
  # x = EXP((Z^T)^-1 Ln(y))
  # First check if 'Z' is orthogonal
  y = complex_nullspace
  Z = network.complex_composition_matrix
  y = np.log(y)

  # Compute complex balance from Matrix Tree theorem
  print(network.complex_laplacian.shape)
  print(network.species_laplacian.shape)
  # print(network.compute_complex_balance(300))
  # print(network.compute_species_balance(300))
  print(cofactor_matrix(network.complex_laplacian)[0])
  print(cofactor_matrix(network.species_laplacian)[0])
  exit()
  # if is_orthogonal(Z):
  #   y = Z @ y
  # else:
  #   # Costly! Can't invert a non-square matrix!!!
  #   y = np.linalg.inv(Z.T) @ y
  # x = np.exp(y)

  # print(x)

  # This is the same as 'x' here because the species are the same as the
  # complexes
  # However, won't this generalise to species matrix well? Since the complex

  # Check network dynamics to see complex-balanced eqm and steady-states

  # Normalising kinetics
  # normalised_kinetics = network.create_kinetics_matrix(300, True)

  # Pathfinding
  # Find shortest path and 'k' shortest path between two species
  source = 'C'
  target = 'CO'
  # source = 'A'
  # target = 'C'
  cutoff = 4
  shortest_paths = nx.all_simple_paths(network.species_graph, source, target,
                                       cutoff=cutoff)

  unique_paths = []
  unique_lengths = []

  print("Paths and lengths")
  count = 0
  for path in shortest_paths:
    total_length = 0
    for i in range(len(path) - 1):
      source, target = path[i], path[i+1]
      edge = network.species_graph[source][target][0]
      length = edge['weight']
      total_length += length
    # print(path, total_length)

    string_path = ','.join(path)
    if not string_path in unique_paths:
      unique_paths.append(string_path)
      unique_lengths.append(total_length)

    count += 1
    if count >= 1000:
      break

  print(count)
  for path, length in zip(unique_paths, unique_lengths):
    print(path, length)
