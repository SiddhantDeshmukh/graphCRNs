# Figure out the best chemical timescales to use
# Compare simple inverse reaction rates and the Jacobian eigenvalues
from gcrn.network import Network
from gcrn.timescales import iets
from sadchem.postprocessing import calculate_number_densities
from abundances import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import compress


def to_arr(*args):
  return [np.array(arg) for arg in args]


def evts():
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
  if not production_filter_species:
    production_filter_species = network.species
  useful_rates, useful_rxns = [], []  # reaction rates to use for analysis
  done_rxn_idxs = []
  rhs_dict = network.rhs_rxns()
  T, n = network.temperature, network.number_densities_dict
  for s in rhs_dict.keys():
    # Compute weights for each species based on production rate
    rxns = rhs_dict[s]["source"] + rhs_dict[s]["sink"]
    rates = np.zeros(shape=len(rxns))
    for j, rxn in enumerate(rxns):
      reactants, products = rxn.reactants, rxn.products
      has_useful_product = any([s_ in production_filter_species
                                for s_ in products])
      if s in reactant_blacklist_species or not has_useful_product:
        continue
      if not rxn.idx in done_rxn_idxs:
        rates[j] = rxn.evaluate_mass_action_rate(T, n)
        done_rxn_idxs.append(rxn.idx)
    total_production_rate = np.sum(rates)
    weights = rates / total_production_rate

    weight_mask = (weights > weights_threshold)
    # weight_mask = (weights > 0.)  # no filtering!
    filtered_rates = rates[weight_mask]
    filtered_rxns = list(compress(rxns, weight_mask))
    # Add reactions that have weights higher than threshold
    useful_rates.extend(filtered_rates)
    useful_rxns.extend(filtered_rxns)

  useful_rates = np.array(useful_rates)
  timescales = 1 / useful_rates
  # print(weights)

  if verbose:
    print(f"T = {T} [K]. {len(timescales)} relevant timescales.")
    for i in range(len(useful_rates)):
      print(f"{i}: {useful_rxns[i]}; {timescales[i]:1.2e} [s]")
    longest_ts = np.nanmax(timescales)
    ts_idx = np.where(timescales == longest_ts)[0][0]
    print(f"Longest timescale: {useful_rxns[ts_idx]}; {longest_ts:1.2e} [s]")

  if return_all:
    return timescales
  else:
    return np.nanmax(timescales)


def network_iets(network: Network):
  return iets(network.evaluate_jacobian(network.temperature,
                                        network.number_densities))


def main():
  PROJECT_DIR = "/home/sdeshmukh/Documents/graphCRNs"
  RES_DIR = f"{PROJECT_DIR}/res"
  # network_id = "solar_co_w05.ntw"
  network_id = "cno_fix.ntw"
  # network_id = "cno_extra.ntw"
  network = Network.from_krome_file(f"{RES_DIR}/{network_id}",
                                    initialise_jacobian=True)
  # temperature = 5700.
  # rho = 1e-8
  temperature = 3500.
  rho = 1e-11
  # abundances = mm00_abundances
  abundances = mm30a04_abundances
  n = calculate_number_densities(abundances, np.log10(rho))
  network.number_densities = n
  network.temperature = temperature
  # Step through network, keeping track of various timescales
  initial_time = 1e-8
  times = np.logspace(-4, 8, num=50)
  timescales = {
      "IRRTS": [network_irrts(network)],
      "IRRTS FILTER": [network_irrts(network, use_relevant_species=True)],
      "FIRRTS": [network_firrts(network)],
      "ETS": [network_ets(network)],
      "OFTS": [network_ofts(network)],
      # "IETS": [],
  }
  # timescales["IETS"] = [iets(network.evaluate_jacobian(temperature,
  #                                              network.number_densities))]
  number_densities = [network.number_densities]
  prev_time = 0.
  for i, time in enumerate(times):
    # solve, then evaluate timescale
    network.solve([time], initial_time=prev_time, n_subtime=1)
    number_densities.append(deepcopy(network.number_densities))

    # Timescales
    timescales["IRRTS"].append(network_irrts(network))
    timescales["IRRTS FILTER"].append(network_irrts(network,
                                                    use_relevant_species=True))
    timescales["FIRRTS"].append(network_firrts(network))
    timescales["ETS"].append(network_ets(network,
                                         relevant_species=["CH", "CO", "OH", "CN", "C2"]))
    timescales["OFTS"].append(network_ofts(network))
    # timescales["IETS"].append(iets(network.evaluate_jacobian(temperature,
    #                                                           network.number_densities)))

    prev_time = time

  # network_evts(network)
  # network_irrts(network)
  # exit()
  all_irrts_timescales = network_irrts(network, verbose=True, return_all=True)
  firrts_timescales_all = network_firrts(network, verbose=True, return_all=True)
  firrts_timescales_sample = network_firrts(network, verbose=True,
                                            return_all=True,
                                            production_filter_species=[
                                                "C", "O", "CO", "OH", "CH", "CN", "C2"],
                                            reactant_blacklist_species=["NO", "NH", "N2", "O2", "H", "H2", "M"])

  fig, ax = plt.subplots()
  ax.hist(np.log10(all_irrts_timescales), bins=15, label="IRRTS")
  ax.hist(np.log10(firrts_timescales_all), bins=15, label="FIRRTS All")
  ax.hist(np.log10(firrts_timescales_sample), bins=15, label="FIRRTS Sample")
  ax.legend()
  plt.show()
  exit()

  times = np.array([initial_time, *times])
  number_densities = np.array(number_densities)
  number_densities[0] = number_densities[1]
  fig, axes = plt.subplots(3, 1)
  species = ["CH", "CO", "OH", "CN", "C2"]
  # species = network.species
  # print(network.species)
  for i, s in enumerate(network.species):
    # Only plot relevant species, but enumerate over all bc of indices
    if s in species:
      # evolution
      axes[0].plot(times, number_densities[:, i],
                   marker='o', label=s)

  for t_key in timescales.keys():
    # timescales
    timescales[t_key][0] = timescales[t_key][1]
    axes[1].plot(times, timescales[t_key], marker='o', label=t_key)
    # # Change in timescales
    # axes[2].plot(times, np.diff(timescales[:, i], prepend=0) / timescales[:, i],
    #             marker='o', label=t_key)

  # Aesthetics
  for ax in axes:
    ax.set_xlabel("log time [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")

  axes[0].set_ylabel(r"log $n$ [cm$^{-3}$]")
  axes[1].set_ylabel(r"log $\tau$ [s]")
  axes[2].set_ylabel(r"$\Delta \tau / \tau$ [s]")
  axes[0].legend()
  axes[1].legend()
  fig.suptitle(rf"T = {temperature:.0f} [K], $\rho$ = {rho:1.1e}")
  plt.show()


if __name__ == "__main__":
  main()
