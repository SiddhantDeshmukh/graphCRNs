# First we have to write something to read in reactions
# Need some kind of 'intelligent' reader, the idea is to use KROME format so we
# read to find the 'format', then the lines after that have to follow this
# format. Need a Reaction class and a Network class for sure
from typing import Dict, List, Union
from math import exp  # used in 'eval'
from .limits import limit_dict, scale_dict
import numpy as np
import re


# Order list alphabetically so that complex combinations work out
# e.g. H + CH == CH + H
# TODO: Do it properly when creating the set of complexes!
def create_complex(lst): return ' + '.join(sorted(lst))


class Reaction:
  def __init__(self, reactants: List, products: List, rate_expression: str,
               idx=None, min_temperature=None, max_temperature=None,
               limit=None, reference=None) -> None:
    self.reactants = reactants
    self.products = products

    # Remove empty reactants and products
    self.reactants = [reactant for reactant in self.reactants if reactant]
    self.products = [product for product in self.products if product]

    # Create stoichiometry
    self.stoichiometry = self.calcluate_stoichiometry()

    self.reactant_complex = create_complex(self.reactants)
    self.product_complex = create_complex(self.products)

    self.idx = idx if idx else None
    self.min_temperature = float(min_temperature)\
        if min_temperature else None
    self.max_temperature = float(max_temperature)\
        if max_temperature else None
    self.limit = limit if limit else None

    # Need to evaluate this rate expression for temperature, so keep it as
    # something we can 'eval'
    self.rate_expression = rate_expression
    # Reference default is 'NONE'
    self.reference = reference if reference else 'NONE'
    # Fortran -> Python format
    self.rate_expression = rate_expression.replace('d', 'e')
    self.rate = self.evaluate_rate_expression(300)  # default
    self.mass_action_rate_expression = self.determine_mass_action_rate()

  def __str__(self) -> str:
    return f"{create_complex(self.reactants)} -> {create_complex(self.products)}"

  def description(self) -> str:
    output = ""
    if self.idx:
      output += f"{self.idx}: "

    output += f"{create_complex(self.reactants)} -> {create_complex(self.products)}"
    output += f"\tRate: {self.rate_expression}"

    return output

  def krome_str(self) -> str:
    from gcrn.utilities import to_fortran_str
    # Comma-separated string as expected by KROME format
    rxn_str = ""
    if self.idx:
      rxn_str += f"{self.idx},"

    rxn_str += ','.join([r for r in self.reactants]) + ","
    rxn_str += ','.join([p for p in self.products]) + ","
    # TODO:
    # Find a better way to replace 'e' -> 'd' while retaining 'exp'
    rxn_str += f"{self.rate_expression.replace('e', 'd').replace('dxp', 'exp')},"

    if self.min_temperature or self.max_temperature:
      rxn_str += f"{to_fortran_str(self.min_temperature, '.3e')},"
      rxn_str += f"{to_fortran_str(self.max_temperature, '.3e')},"
      rxn_str += f"{self.limit},"

    rxn_str += self.reference

    # Remove trailing comma
    return rxn_str.rstrip(',')

  def __call__(self, temperature: float, use_limit=True) -> float:
    return self.evaluate_rate_expression(temperature, use_limit=use_limit)

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

  def evaluate_rate_expression(self, temperature=None, use_limit=True):
    # Evaluate (potentially temperature-dependent) rate expression
    # WARNING: Eval is evil!
    # TODO: Sanitise input
    expression = self.rate_expression.replace("Tgas", str(temperature))
    rate = eval(expression)

    # Check if rate should be limited
    if use_limit and self.limit:
      scale_limits = ['weak', 'medium', 'strong']
      if self.limit in scale_limits:  # 'limit' uses 'scale'
        rate = limit_dict[self.limit](rate, temperature, self.min_temperature,
                                      self.max_temperature,
                                      scale_dict[self.limit],
                                      scale_dict[self.limit])
      elif self.limit == 'boundary':
        if temperature < self.min_temperature:
          temperature = self.min_temperature
        elif temperature > self.max_temperature:
          temperature = self.max_temperature

        expression = self.rate_expression.replace("Tgas", str(temperature))
        rate = eval(expression)

      elif self.limit == 'sharp':
        if temperature < self.min_temperature or temperature > self.max_temperature:
          rate = 0
      else:
        rate = limit_dict[self.limit](rate, temperature,
                                      self.min_temperature,
                                      self.max_temperature)

    return rate

  def determine_mass_action_rate(self):
    # String expression of the reaction rate v(x)
    reactants = self.stoichiometry[0]
    rate = f"({self.rate_expression})"
    for key, value in reactants.items():
      rate += f" * n_{key}"
      if abs(value) > 1:
        rate += f"**{abs(value)}"

    return rate

  def evaluate_mass_action_rate(self, temperature: float,
                                number_densities: Dict) -> float:
    # Provided a temperature and number densities for the reactants (dictionary
    # containing all reactants or numpy array indexed the same as the reactants)
    # compute the mass action rate
    rate = self.determine_mass_action_rate().replace("Tgas", str(temperature))
    # TODO:
    # Regex replacement for 'n_*'! Otherwise 'n_O2' can get replaced by
    # 'n_O', etc
    pattern = r"n_[A-Z0-9]*"
    # if isinstance(number_densities, dict):
    for r in self.reactants:
      rate = re.sub(pattern,
                    lambda s: f"{number_densities[s.group()[2:]]}",
                    rate)
    # elif isinstance(number_densities, np.ndarray):
    #   for i, r in enumerate(self.reactants):
    #     rate = re.sub(pattern,
    #                   lambda s: f"{number_densities[self.reactants.index(s.group()[2:])]}",
    #                   rate)

    return eval(rate)
