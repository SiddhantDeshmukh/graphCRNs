# First we have to write something to read in reactions
# Need some kind of 'intelligent' reader, the idea is to use KROME format so we
# read to find the 'format', then the lines after that have to follow this
# format. Need a Reaction class and a Network class for sure
from typing import Dict, List
from math import exp  # used in 'eval'
from .limits import limit_dict, scale_dict


# Order list alphabetically so that complex combinations work out
# e.g. H + CH == CH + H
# TODO: Do it properly when creating the set of complexes!
def create_complex(lst): return ' + '.join(sorted(lst))


class Reaction:
  def __init__(self, reactants: List, products: List, rate_expression: str,
               idx=None, min_temperature=None, max_temperature=None,
               limit=None) -> None:
    self.reactants = reactants
    self.products = products

    # Remove empty reactants and products
    self.reactants = [reactant for reactant in self.reactants if reactant]
    self.products = [product for product in self.products if product]

    # Create stoichiometry
    self.stoichiometry = self.calcluate_stoichiometry()

    self.reactant_complex = create_complex(self.reactants)
    self.product_complex = create_complex(self.products)

    # Need to evaluate this rate expression for temperature, so keep it as
    # something we can 'eval'
    self.rate_expression = rate_expression
    # Fortran -> Python format
    self.rate_expression = rate_expression.replace('d', 'e')

    self.rate = self.evaluate_rate_expression(300)  # default
    self.mass_action_rate_expression = self.determine_mass_action_rate()

    self.idx = idx if idx else None
    self.min_temperature = min_temperature if min_temperature else None
    self.max_temperature = max_temperature if max_temperature else None
    self.limit = limit if limit else None

  def __str__(self) -> str:
    output = f"{create_complex(self.reactants)} -> {create_complex(self.products)}"
    output += f"\tRate: {self.rate_expression}"
    if self.idx:
      output += f"\tIndex = {self.idx}"

    return output

  def __call__(self, temperature: float) -> float:
    return self.evaluate_rate_expression(temperature)

  def calcluate_stoichiometry(self) -> Dict:
    # Write out stoichiometry for the reaction
    # Convention: reactants are negative
    reactant_stoichiometry = {}
    product_stoichiometry = {}

    for reactant in self.reactants:
      if reactant in reactant_stoichiometry.keys():
        reactant_stoichiometry[reactant] -= 1
      else:
        reactant_stoichiometry[reactant] = -1

    for product in self.products:
      if product in product_stoichiometry.keys():
        product_stoichiometry[product] += 1
      else:
        product_stoichiometry[product] = 1

    return reactant_stoichiometry, product_stoichiometry

  def evaluate_rate_expression(self, temperature=None):
    # Evaluate (potentially temperature-dependent) rate expression
    # WARNING: Eval is evil!
    # TODO: Sanitise input
    expression = self.rate_expression.replace("Tgas", str(temperature))
    rate = eval(expression)
    if self.limit:
      rate = limit_dict[self.limit](rate, temperature, self.min_temperature,
                                    self.max_temperature,
                                    scale_dict[self.limit],
                                    scale_dict[self.limit])
    return rate

  def determine_mass_action_rate(self):
    # String expression of the reaction rate v(x)
    reactants = self.stoichiometry[0]
    rate = f"({self.rate_expression})"
    for key, value in reactants.items():
      rate += f" * n_{key}"
      if abs(value) > 1:
        rate += f"^{abs(value)}"

    return rate
