import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


species = ['C', 'C2', 'CH', 'CN', 'CO', 'H', 'H2',
           'N', 'N2', 'NH', 'NO', 'O', 'O2', 'OH']


plt.figure()
options = {
    "node_color": "white",
    "with_labels": True,
    'font_size': 10,
    # 'arrows': FalseSun,
    # 'linewidths': 0,
    # 'width': 0

}

edges = [(s, s1) for s in species for s1 in species if not s == s1
         if np.random.random() <= 0.3]
G = nx.MultiDiGraph(edges)
nx.draw_shell(G,  **options)
plt.show()
