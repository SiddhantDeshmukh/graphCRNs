# First we have to write something to read in reactions
# Need some kind of 'intelligent' reader, the idea is to use KROME format so we
# read to find the 'format', then the lines after that have to follow this
# format. Need a Reaction class and a Network class for sure
from typing import List


def create_complex(lst): return ' + '.join(lst)


class Reaction:
  def __init__(self, reactants: List, products: List, rate_expression: str,
               idx=None) -> None:
    self.reactants = reactants
    self.products = products

    # Remove empty reactants and products
    self.reactants = [reactant for reactant in self.reactants if reactant]
    self.products = [product for product in self.products if product]

    # Need to evaluate this rate expression for temperature, so keep it as
    # something we can 'eval'
    self.rate_expression = rate_expression
    # Fortran -> Python format
    self.rate_expression = rate_expression.replace('d', 'e')

    if idx:
      self.idx = idx

  def __str__(self) -> str:
    output = f"{create_complex(self.reactants)} -> {create_complex(self.products)}"
    output += f"\tRate: {self.rate_expression}"
    if self.idx:
      output += f"\tIndex = {self.idx}"

    return output


class Network:
  def __init__(self, reactions: List[Reaction]) -> None:
    self.reactions = reactions

    species = []
    for rxn in reactions:
      species.extend(rxn.reactants + rxn.products)
    self.species = list(set(species))

    # Also initialise incidence matrix, adjacency matrix, etc?


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

  for rxn in network.reactions:
    print(rxn)

  print(network.species)
