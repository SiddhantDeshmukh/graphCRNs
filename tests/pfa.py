# A (prototypical) implementation of path flux analysis (PFA) from
# Sun et al (2010) [Combustion and Flame 157 (2010) 1298-1307]
from typing import List
from gcrn.network import Network


# The goal is to identify the species most important to the target species
# using the 'interaction coefficient'


def interaction_coefficient(network: Network, source: str, target: str):
  pass


def interaction_coefficients(network: Network, sources: List[str],
                             targets: List[str]):
  interaction_coefficients = []
