# Prototyping for a Graph NN to solve chemistry
# Inspiration taken from https://github.com/andreuva/graphnet_nlte
# (Vicente +21)
import numpy as np
import tf_geometric as tfg
import tensorflow as tf

from gcrn.graphNet.sampling import sample_nd


def tutorial():
  # (from https://tf-geometric.readthedocs.io/en/latest/index.html)
  # Construct a graph and apply a multi-head graph attention network (GAT) to it
  graph = tfg.Graph(
      x=np.random.randn(5, 20),
      edge_index=[[0, 0, 1, 3],
                  [1, 2, 2, 1]]  # 4 undirected edges
  )

  print("Graph Description:")
  print(graph)
  # Convert to directed graph automatically
  graph.convert_edge_to_directed()
  print("Processed Graph Description:")
  print(graph)
  print("Processed Edge Index:")
  print(graph.edge_index)

  # Multi-head GAT
  gat_layer = tfg.layers.GAT(units=4, num_heads=4, activation=tf.nn.relu)
  output = gat_layer([graph.x, graph.edge_index])
  print("Output:")
  print(output)


def write_samples(samples: np.ndarray, outfile: str):
  np.savetxt(outfile, samples, delimiter=",",
             header="density,temperature,tspan")


def main():
  # tutorial()
  # graph_cnn()
  # (rho, T, tspan)
  num_points = 1000
  # x & z uniform in log-space, y uniform in linear space
  bounds = [(-10, -6), (3000., 30000.), (-6, 6)]
  samples = sample_nd(bounds, num_points)
  samples[:, 0] = 10**samples[:, 0]
  samples[:, 2] = 10**samples[:, 2]
  write_samples(samples, "./samples.csv")


if __name__ == "__main__":
  main()
