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

if __name__ == "__main__":
  # Initialising network
  # krome_file = '../res/react-co-solar-umist12'
  krome_file = '../res/ring-reaction'
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
  print(f"Kinetics:       {network.kinetics_matrix.shape}")
  print(f"Laplacian:      {network.laplacian_matrix.shape}")

  # # Dynamics
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

  # dynamics = NetworkDynamics(network, initial_number_densities)
  # print(dynamics.network.species)
  # print(dynamics.initial_number_densities)
  # print(dynamics.dynamics_vector)

  # Equilibrium
  # Complex-balanced equilibrium exists if Dv(x*) = 0
  # Species-balanced equilibrium exists if Sv(x*) = 0 (?)
  # Since Dv(x*) = -LExp(...), we can check nullspace of L for complex-balanced
  # and nullspace of SK for species-balanced
  complex_nullspace = null_space(network.laplacian_matrix)

  species_laplacian = network.stoichiometric_matrix @ network.kinetics_matrix
  species_nullspace = null_space(species_laplacian)

  print("Equilibria nullspaces")
  print(f"Complex: {complex_nullspace}")
  print(network.laplacian_matrix)
  print(f"Species: {species_nullspace}")
  print(species_laplacian)

  # Pathfinding
