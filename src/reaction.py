# First we have to write something to read in reactions
# Need some kind of 'intelligent' reader, the idea is to use KROME format so we
# read to find the 'format', then the lines after that have to follow this
# format. Need a Reaction class and a Network class for sure
from typing import List
from math import exp  # used in 'eval'


# Order list alphabetically so that complex combinations work out
# e.g. H + CH == CH + H
# TODO: Do it properly when creating the set of complexes!
def create_complex(lst): return ' + '.join(sorted(lst))


class Reaction:
  def __init__(self, reactants: List, products: List, rate_expression: str,
               idx=None, min_temperature=None, max_temperature=None) -> None:
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

    self.idx = idx if idx else None
    self.min_temperature = min_temperature if min_temperature else None
    self.max_temperature = max_temperature if max_temperature else None

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
