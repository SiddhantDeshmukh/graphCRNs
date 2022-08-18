import networkx as nx
import matplotlib.pyplot as plt
from gcrn.network import Network
from gcrn.graphs import create_species_graph


def main():
  PROJECT_DIR = "/home/sdeshmukh/Documents/graphCRNs"
  RES_DIR = f"{PROJECT_DIR}/res"
  FIG_DIR = f"{PROJECT_DIR}/out/figs"
  network = Network.from_krome_file(f"{RES_DIR}/ch_oh_co.ntw")

  G = create_species_graph(network)
  fig, ax = plt.subplots()
  # Layout
  options = {
      # "node_color": "orange",
      "node_size": 500,
      # "with_labels": True,
      # 'font_size': 11,
      'linewidths': 1,
      # "edgecolors": "black",
      # 'width': 1
  }
  G.remove_node("H")
  pos = {
      'C': (10., 10.),
      'O': (10., -10.),
      'CH': (20., 10.),
      'OH': (20., -10.),
      'CO': (25., 0.),
  }
  arrowsize = 15
  # General labels
  nx.draw_networkx_labels(G, pos, ax=ax)
  # Atoms
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=["C", "O"],
                         node_color="tomato", **options)
  # Molecules
  nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=["OH", "CO", "CH"],
                         node_color="tab:cyan", **options)
  # C -> CH -> CO pathway (P1)
  nx.draw_networkx_edges(G, pos, edgelist=[("C", "CH"), ("CH", "CO"),
                                           ("CH", "C"), ("CO", "CH")],
                         edge_color="forestgreen", ax=ax, arrowsize=arrowsize)
  # C -> CO direct pathway (P2)
  nx.draw_networkx_edges(G, pos, edgelist=[("C", "CO"), ("CO", "C")],
                         edge_color="grey", ax=ax, arrowsize=arrowsize)
  # O -> CO direct pathway (P2)
  nx.draw_networkx_edges(G, pos, edgelist=[("O", "CO"), ("CO", "O")],
                         edge_color="grey", ax=ax, arrowsize=arrowsize)
  # O -> OH -> CO pathway (P3)
  nx.draw_networkx_edges(G, pos, edgelist=[("O", "OH"), ("OH", "CO"),
                                           ("OH", "O"), ("CO", "OH")],
                         edge_color="forestgreen", ax=ax, arrowsize=arrowsize)
  # Edge labels
  nx.draw_networkx_edge_labels(G, pos, edge_labels={
      ('C', 'CH'): r"$\mathbf{P_1}$: C + H $\Longleftrightarrow$ CH",
      ('CH', 'CO'): r"$\mathbf{P_1}$: CH + O $\Longleftrightarrow$ CO + H",
      ('C', 'CO'): r"$\mathbf{P_2}$: C + O $\Longleftrightarrow$ CO",
      ('O', 'CO'): r"$\mathbf{P_2}$: O + C $\Longleftrightarrow$ CO",
      ('O', 'OH'): r"$\mathbf{P_3}$: O + H $\Longleftrightarrow$ OH",
      ('OH', 'CO'): r"$\mathbf{P_3}$: OH + C $\Longleftrightarrow$  CO + H",
  })
  # Plot aesthetics
  ax.axis("off")
  plt.tight_layout()
  plt.savefig(f"{FIG_DIR}/ch_oh_co_diagram.png")
  plt.show()


if __name__ == "__main__":
  main()
