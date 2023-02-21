import numpy as np
from scipy.interpolate import splrep, splev
from gcrn.network import Network
from typing import List, Dict
from itertools import compress


def ratio(arr: np.ndarray):
  # Ratios of consecutive elements
  return np.exp(-np.diff(np.log(arr)))


def find_equilibrium(evolution_times: np.ndarray,
                     number_densities: np.ndarray, threshold=1e-6):
  # Compute the time at which each chemical species is in equilibrium, defined
  # by the relative change being less than the (relative) threshold
  # 'evolution_times' must be a 1D array with dimension (time)
  # 'number_densities' must be a 2D array with dimensions (time, species)
  eqm_times = np.empty(shape=number_densities.shape[1])
  eqm_number_densities = np.empty(shape=number_densities.shape[1])
  for i in range(number_densities.shape[1]):  # over species
    try:
      threshold_idx = np.where(
          np.abs(1. - ratio(number_densities[:, i])) >= threshold)[0][-1]
    except IndexError:  # condition not fulfilled; species always in eqm
      threshold_idx = 0
    eqm_times[i] = evolution_times[threshold_idx]
    eqm_number_densities[i] = number_densities[threshold_idx, i]

  return eqm_times, eqm_number_densities


def find_equilibrium_interpolate(evolution_times: np.ndarray,
                                 number_densities: np.ndarray, threshold=1e-6):
  # Compute the time at which each chemical species is in equilibrium, defined
  # by the relative change being less than the (relative) threshold
  # 'evolution_times' must be a 1D array with dimension (time)
  # 'number_densities' must be a 2D array with dimensions (time, species)
  eqm_times = np.empty(shape=number_densities.shape[1])
  eqm_number_densities = np.empty(shape=number_densities.shape[1])
  # We interpolate each species trend first to find a more preciese equilibrium
  # time
  for i in range(number_densities.shape[1]):  # over species
    try:
      # Initial guess is solver time
      threshold_idx = np.where(
          np.abs(1. - ratio(number_densities[:, i])) >= threshold)[0][-1]

      # Sample points around this time

    except IndexError:  # condition not fulfilled; species always in eqm
      threshold_idx = 0
    eqm_times[i] = evolution_times[threshold_idx]
    eqm_number_densities[i] = number_densities[threshold_idx, i]

  return eqm_times, eqm_number_densities


# ----------------------------------------------------------- ------------------
# Timescale functions
# -----------------------------------------------------------------------------
def iets(jacobian: np.ndarray):
  # Return the inverse-eigenvalue timescales without filtering
  # zero-eigenvalues are given zero-value timescales
  # Jacobian analysis
  eigenvalues, eigenvectors = np.linalg.eig(jacobian)

  # Calculate timescales
  timescales = [1 / np.abs(e) if e < 0 else 0 for e in eigenvalues]
  return timescales


def evts(jacobian: np.ndarray):
  # CTS-ID from Caudal+12
  pass


def network_ofts(network: Network):
  # OpenFOAM timescale (Golovitchev & Nordin 2001)
  # Timescale is the sum over reactions of number density divided by
  # sum of production rate * stoichiometric coefficient of product species
  denominator = 0.
  rates = network.calculate_dynamics()
  for rxn in network.reactions:
    _, product_stoich = rxn.calculate_stoichiometry()
    denominator = 0.
    for i, s in enumerate(network.species):
      if s in product_stoich.keys():
        denominator += product_stoich[s] * rates[i]

  n_total = network.number_densities.sum()
  timescale = n_total / denominator

  return timescale


def network_ets(network: Network,
                relevant_species=["CH", "CO", "OH", "CN", "C2"], min_rate=1e-16):
  # Evans timescale (Evans+2019)
  # Max of (number denisty / rate) for all _relevant_ species
  timescales = {}
  n = network.number_densities
  rates = network.calculate_dynamics()
  for i, s in enumerate(network.species):
    if s in relevant_species:
      if rates[i] < min_rate:
        timescales[i] = np.nan
      timescales[s] = n[i] / np.abs(rates[i])

      # print(f"{s}: {timescales[s]:1.2e} [s]")

  # print(timescales.values())
  max_value = np.nanmax([v for v in timescales.values()])
  return max_value


def network_evts(network: Network):
  # Analyse network at current state (assumes Jacobian initialised) and compute
  # EVTS
  eigenvalues, eigenvectors = np.linalg.eig(network.evaluate_jacobian(network.temperature,
                                                                      network.number_densities))
  rates = network.calculate_dynamics()
  print(eigenvalues.shape)
  print(eigenvectors.shape)
  for s, rate in zip(network.species, rates):
    print(f"{s}: {rate:1.2e}")


def network_irrts(network: Network, use_relevant_species=False,
                  relevant_species=["CO", "OH", "CH", "CN", "C2"],
                  verbose=False, return_all=False):
  # Compute Inverse Reaction Rate Timescale for the provided network
  rates = network.eval_rates_vector()

  if use_relevant_species:
    # Only include those that produce 'relevant_products'
    for i, rxn in enumerate(network.reactions):
      rxn_species = list(set(rxn.products + rxn.reactants))
      if not any(x in relevant_species for x in rxn_species):
        rates[i] = np.nan

  timescales = 1 / rates

  if verbose:
    for i, rxn in enumerate(network.reactions):
      if not np.isnan(timescales[i]):
        print(f"{i}: {rxn}: {timescales[i]:1.2e} [s]")
    longest_ts = np.nanmax(timescales)
    ts_idx = np.where(timescales == longest_ts)[0][0]
    print(
        f"Longest timescale: {network.reactions[ts_idx]}; {longest_ts:1.2e} [s]")

  if return_all:
    return timescales
  else:
    return np.nanmax(timescales)


def network_firrts(network: Network, verbose=False, return_all=False,
                   weights_threshold=0.05, production_filter_species=[],
                   reactant_blacklist_species=[]):
  # Custom version of inverse reaction rate timescale that filters rates based
  # on production rates of species
  # 'Unimportant' reactions (low yield) are not used to calculate a timescale
  # 'production_filter_species' is the list of species to consider on the RHS
  # (those produced) to filter rates. This lets the user neglect unimportant
  # species in the network if so desired.
  # If no species chosen for production filter, use all species
  def weight_rxns(s: str, rxns: List, weights_threshold=0.05):
    # Compute weights for each species based on production rate of each rxn
    rates = np.zeros(shape=len(rxns))
    for j, rxn in enumerate(rxns):
      reactants, products = rxn.reactants, rxn.products
      has_useful_product = any([s_ in production_filter_species
                                for s_ in products])
      if s in reactant_blacklist_species or not has_useful_product:
        continue
      if not rxn.idx in done_rxn_idxs:
        original_rate = rxn.evaluate_mass_action_rate(T, n)
        adjusted_rate = original_rate / n[s]
        # rates[j] = original_rate
        rates[j] = adjusted_rate
        if verbose:
          print(f"{rxn.idx}: {rxn}, dividing by {s} ({n[s]:1.2e})")
          print(
              f"Original: {original_rate:1.2e}, Adjusted: {adjusted_rate:1.2e}")
    total_production_rate = np.sum(rates)
    weights = rates / total_production_rate

    # Keep reactions that have weights higher than threshold
    weight_mask = (weights > weights_threshold)
    # weight_mask = (weights > 0.)  # no filtering!
    filtered_rates = rates[weight_mask]
    filtered_rxns = list(compress(rxns, weight_mask))

    return filtered_rates, filtered_rxns, weights[weight_mask]

  if not production_filter_species:
    production_filter_species = network.species
  # reaction rates to use for analysis
  useful_rates, useful_rxns, useful_weights = [], [], []
  done_rxn_idxs = []
  rhs_dict = network.rhs_rxns()
  T, n = network.temperature, network.number_densities_dict
  for s in rhs_dict.keys():
    rxns_to_consider = [rhs_dict[s]["source"] + rhs_dict[s]["sink"]]
    for rxns in rxns_to_consider:
      filtered_rates, filtered_rxns, weights = weight_rxns(s, rxns,
                                                           weights_threshold=weights_threshold)
      useful_rates.extend(filtered_rates)
      useful_rxns.extend(filtered_rxns)
      useful_weights.extend(weights)

  useful_rates = np.array(useful_rates)
  timescales = 1 / useful_rates

  if verbose:
    print(f"T = {T} [K]. {len(timescales)} relevant timescales.")
    for i in range(len(useful_rates)):
      print(
          f"{i}/{useful_rxns[i].idx}: {useful_rxns[i]}; {timescales[i]:1.2e} [s], rate = {1/timescales[i]:1.2e}, weight = {useful_weights[i]:.2f}")
    longest_ts = np.nanmax(timescales)
    ts_idx = np.where(timescales == longest_ts)[0][0]
    print(
        f"Longest timescale: ({useful_rxns[ts_idx].idx}) {useful_rxns[ts_idx]}; {longest_ts:1.2e} [s]; {1 / longest_ts:1.2e} [1/s]")

  if return_all:
    return timescales
  else:
    return np.nanmax(timescales)


def network_iets(network: Network):
  return iets(network.evaluate_jacobian(network.temperature,
                                        network.number_densities))


def compute_timescales(network: Network, method="firrts", **kwargs):
  timescale_func = {
      "iets": network_iets,
      "ets": network_ets,
      "evts": network_evts,
      "ofts": network_ofts,
      "irrts": network_irrts,
      "firrts": network_firrts,
  }[method]

  return timescale_func(network, **kwargs)
