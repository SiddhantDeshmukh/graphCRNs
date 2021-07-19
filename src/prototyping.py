import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pylab import draw, draw_networkx_edge_labels
from networkx.drawing.nx_pydot import to_pydot
from src.reactions import Reaction
import pydot


reactants = ['H', 'CH']
products = ['C', 'H2']
forward_rate = 5
reverse_rate = 3

forward_reaction = Reaction(
    reactants, products, rate_expression=f"{forward_rate}")
reverse_reaction = Reaction(
    products, reactants, rate_expression=f"{reverse_rate}")

# Create graph of species
species_graph = nx.MultiDiGraph()
for r in forward_reaction.reactants:
  for p in forward_reaction.products:
    species_graph.add_edge(r, p, weight=forward_rate)
for r in reverse_reaction.reactants:
  for p in reverse_reaction.products:
    species_graph.add_edge(r, p, weight=reverse_rate)

plt.figure()
pos = nx.spring_layout(species_graph)
draw(species_graph, pos=pos, with_labels=True)
draw_networkx_edge_labels(species_graph, pos=pos)

# Create graph of complexes
complex_graph = nx.MultiDiGraph()
complex_graph.add_edge(forward_reaction.reactant_complex,
                       forward_reaction.product_complex, weight=forward_rate)
complex_graph.add_edge(forward_reaction.product_complex,
                       forward_reaction.reactant_complex, weight=reverse_rate)

plt.figure()
pos = nx.spring_layout(complex_graph)
draw(complex_graph, pos=pos, with_labels=True)
draw_networkx_edge_labels(complex_graph, pos=pos)

print(complex_graph.number_of_edges())
# plt.show()

pydot_species_graph = to_pydot(species_graph)
pydot_complex_graph = to_pydot(complex_graph)

pydot_species_graph.write_png("./species.png")
pydot_complex_graph.write_png("./complex.png")
