# Prototyping for a Graph NN to solve chemistry
# Inspiration taken from https://github.com/andreuva/graphnet_nlte
# (Vicente +21)
import numpy as np
import tf_geometric as tfg
import tensorflow as tf


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


def main():
  tutorial()
  return


if __name__ == "__main__":
  main()
