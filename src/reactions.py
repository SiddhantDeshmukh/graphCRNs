# First we have to write something to read in reactions
# Need some kind of 'intelligent' reader, the idea is to use KROME format so we
# read to find the 'format', then the lines after that have to follow this
# format. Need a Reaction class and a Network class for sure
from networkx.drawing.nx_pydot import to_pydot
from typing import List
import networkx as nx
from itertools import product
from math import exp  # used in 'eval'


def create_complex(lst): return ' + '.join(lst)


class Reaction:
  def __init__(self, reactants: List, products: List, rate_expression: str,
               idx=None) -> None:
    self.reactants = reactants
    self.products = products

    # Remove empty reactants and products
    self.reactants = [reactant for reactant in self.reactants if reactant]
    self.products = [product for product in self.products if product]

    self.reactant_complex = create_complex(self.reactants)
    self.product_complex = create_complex(self.products)

    # Need to evaluate this rate expression for temperature, so keep it as
    # something we can 'eval'
    self.rate_expression = rate_expression
    # Fortran -> Python format
    self.rate_expression = rate_expression.replace('d', 'e')

    self.rate = self.evaluate_rate_expression(300)  # default

    if idx:
      self.idx = idx

  def __str__(self) -> str:
    output = f"{create_complex(self.reactants)} -> {create_complex(self.products)}"
    output += f"\tRate: {self.rate_expression}"
    if self.idx:
      output += f"\tIndex = {self.idx}"

    return output

  def evaluate_rate_expression(self, temperature=None):
    # Evaluate (potentially temperature-dependent) rate expression
    # WARNING: Eval is evil!
    expression = self.rate_expression.replace("Tgas", str(temperature))
    rate = eval(expression)
    return rate


class Network:
  def __init__(self, reactions: List[Reaction]) -> None:
    self.reactions = reactions

    # Potential space saving with clever list comprehensions?
    # Doing species and complexes in this way is redundant, should do complexes
    # first and then reduce that to get the species
    species = []
    for rxn in reactions:
      species.extend(rxn.reactants + rxn.products)
    self.species = list(set(species))

    # TODO: Additional constraint: combinations, not permutations!
    # e.g. H + H + H2 == H2 + H + H
    # Also need to check this when creating the graph!
    complexes = []
    for rxn in reactions:
      complexes.append(rxn.reactant_complex)
      complexes.append(rxn.product_complex)
    self.complexes = list(set(complexes))

    # Create MultiDiGraphs for species and complexes using networkx
    self.species_graph = self.create_species_graph()
    self.complex_graph = self.create_complex_graph()

  def create_species_graph(self, temperature=None) -> nx.MultiDiGraph:
    # Create graph of species
    species_graph = nx.MultiDiGraph()
    for rxn in self.reactions:
      # TODO: Optimise with itertools.product()
      for r in rxn.reactants:
        for p in rxn.products:
          weight = rxn.evaluate_rate_expression(temperature) \
              if temperature else rxn.rate
          species_graph.add_edge(r, p, weight=weight)

    return species_graph

  def create_complex_graph(self, temperature=None) -> nx.MultiDiGraph:
    # Create graph of complexes
    complex_graph = nx.MultiDiGraph()
    for rxn in self.reactions:
      weight = rxn.evaluate_rate_expression(temperature) \
          if temperature else rxn.rate
      complex_graph.add_edge(rxn.reactant_complex,
                             rxn.product_complex, weight=weight)

    return complex_graph

  def update_species_graph(self, temperature: float):
    # Given a certain temperature, update the species Graph (weights)
    self.species_graph = self.create_species_graph(temperature=temperature)

  def update_complex_graph(self, temperature: float):
    # Given a certain temperature, update the species Graph (weights)
    self.complex_graph = self.create_complex_graph(temperature=temperature)


# cls method for Network!
def read_krome_file(filepath: str) -> Network:
  # Reads in a KROME rxn network as a Network
  rxn_format = None

  # Store all as list in case there are duplicates; should only have one
  # 'rxn_idx' and 'rate_expression', so we just pass in the first index of the
  # list as Reaction init
  format_dict = {
      'idx': [],
      'R': [],
      'P': [],
      'rate': []
  }

  reactions = []
  with open(filepath, 'r', encoding='utf-8') as infile:
    while True:
      line = infile.readline().strip()
      if not line:
        break

      # Check for 'format'
      if line.startswith('@format:'):
        # Reaction usually made up of 'idx', 'R', 'P', 'rate'
        rxn_format = line.replace('@format:', '').split(',')

      else:
        split_line = line.split(',')
        for i, item in enumerate(rxn_format):
          format_dict[item].append(split_line[i])

        reactions.append(Reaction(format_dict['R'], format_dict['P'],
                                  format_dict['rate'][0],
                                  idx=format_dict['idx'][0]))

      # Reset quantities
      for key in format_dict.keys():
        format_dict[key] = []

  return Network(reactions)


if __name__ == "__main__":
  krome_file = '../res/react-co-solar-umist12'
  network = read_krome_file(krome_file)

  print("Species")
  print(network.species)
  print("Complexes")
  print(network.complexes)

  to_pydot(network.species_graph).write_png("./species.png")
  to_pydot(network.complex_graph).write_png("./complex.png")
