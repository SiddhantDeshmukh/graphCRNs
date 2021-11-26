# Simplify a network across a density-temperature regime
# Iterative scheme, we start with the full network and reduce iteratively to
# most important reactions to reproduce results from the full network for all
# important species within a given tolerance
from copy import deepcopy
import numpy as np
from gcrn.helper_functions import density_temperature_grid_with_pathfinding, reaction_from_idx, rxn_idxs_from_path
from gcrn.network import Network
from gcrn.profiling import most_travelled_pair_idx_paths
import matplotlib.pyplot as plt


def plot_instance(axes, number_densities, network, network_key):
  # Plot one grid cell (density, temperature) over time
  print(network.species)
  for i, s in enumerate(network.species):
    x, y = i // 3, i % 3
    axes[x, y].plot(np.log10(number_densities[2, 3, :, i]), label=network_key)
    axes[x, y].set_title(s)
    axes[x, y].legend()


def main():
  # 1. Define initial network (N0)
  network_dir = '../res'
  network_file = f"{network_dir}/solar_co_w05.ntw"
  # network_file = f"{network_dir}/cno.ntw"
  network = Network.from_krome_file(network_file)

  source_targets = [
      ('C', 'CO'),
      ('O', 'CO'),
      ('C', 'CH'),
      ('O', 'OH'),
      ('H', 'H2'),
      # ('C', 'CN'),
  ]
  # reverse paths
  source_targets += [(t, s) for s, t in source_targets]
  source_target_pairs = [f'{s}-{t}' for s, t in source_targets]
  sources, targets = [l[0] for l in source_targets], [l[1]
                                                      for l in source_targets]

  initial_abundances = {
      "H": 1e12,
      "H2": 1e-4,
      "OH": 1e-12,
      "C": 10**(8.39),
      "O": 10**(8.66),
      "N": 10**(7.83),
      "CH": 1e-12,
      "CO": 1e-12,
      "CN": 1e-12,
      "NH": 1e-12,
      "NO": 1e-12,
      "C2": 1e-12,
      "O2": 1e-12,
      "N2": 1e-12,
      "M": 1e11,
  }

  simplification_dict = {}

  # User control params
  density_range = (-10, -6)  # log
  temperature_range = (3000, 20000)  # linear

  num_density_points = 5
  num_temperature_points = 5

  # 2. Initialise density-temperature grid
  density = np.logspace(*density_range, num=num_density_points)
  temperature = np.linspace(*temperature_range, num=num_temperature_points)
  timescales = np.logspace(-6, 4, num=5)

  # 3. Run dynamics on grid with network N_0
  network_key = 'N0'
  # 4. Find shortest paths for every source-target pair of interest
  # Foreach (density, temperature, time), find the most important reactions
  number_densities, all_point_paths = \
      density_temperature_grid_with_pathfinding(density, temperature,
                                                timescales, network,
                                                initial_abundances,
                                                sources, targets,
                                                cutoff=10, max_paths=4)

  # # Plot number densities of original network
  fig, axes = plt.subplots(3, 3, figsize=(8, 8))
  plot_instance(axes, number_densities, network, 'N0')
  # for i, s in enumerate(network.species):
  #   x, y = i // 3, i % 3
  #   # end timescale
  #   im = axes[x, y].imshow(np.log10(number_densities[:, :, -1, i]),
  #                          origin='lower')
  #   cbar = plt.colorbar(im, ax=axes[x, y])
  #   # axes[x, y].set_xticks(density)
  #   # axes[x, y].set_yticks(temperature)
  #   axes[x, y].set_title(s)

  # plt.show()

  # 5. Store current network, number densities, paths
  simplification_dict[network_key] = {'network': deepcopy(network),
                                      'n': number_densities,
                                      'paths': all_point_paths}

  # 6. Create new network with only most important reactions N_i
  idx_paths = most_travelled_pair_idx_paths(all_point_paths.flatten(),
                                            source_target_pairs)
  # Most important reaction idxs from paths
  idxs = []
  for pair, counts in idx_paths.items():
    print(pair)
    print("\n".join([f'\t{k}: {v}' for k, v in counts.items()]))
    idxs += [rxn_idxs_from_path(k) for k in counts.keys()]

  idxs = list(set([i for j in idxs for i in j]))
  print(idxs)

  # Subsample original network using only new idxs
  reactions = [reaction_from_idx(idx, network) for idx in idxs]
  new_network = Network(reactions, network.temperature,
                        deepcopy(network.number_densities))

  # print(network.description())
  # print(network.species)
  # print(new_network.description())
  # print(new_network.species)

  # 7. Run dynamics with new network N_i
  new_number_densities, new_point_paths = \
      density_temperature_grid_with_pathfinding(density, temperature, timescales,
                                                new_network, initial_abundances,
                                                sources, targets,
                                                cutoff=10, max_paths=10)
  simplification_dict['N1'] = (new_network, new_number_densities.copy(),
                               deepcopy(new_point_paths))
  # 8. Compare results, see if tolerance sufficient
  # Caveat: number of species may be different! For this example they are not,
  # but need to keep track of species via the network in future
  difference = number_densities - new_number_densities
  ratio = number_densities / new_number_densities
  tolerance = 0.1  # relative!
  where = np.where(ratio > tolerance)
  plot_instance(axes, new_number_densities, new_network, 'N1')
  # Plot ratio
  fig, axes = plt.subplots(3, 3, figsize=(8, 8))
  for i, s in enumerate(network.species):
    x, y = i // 3, i % 3
    # end timescale
    im = axes[x, y].imshow(np.log10(ratio[:, :, -1, i]),
                           origin='lower')
    cbar = plt.colorbar(im, ax=axes[x, y])
    # axes[x, y].set_xticks(density)
    # axes[x, y].set_yticks(temperature)
    axes[x, y].set_title(s)

  print(len(network.reactions), len(new_network.reactions))
  plt.show()
  # 9. If good reduction, continue to reduce further

  # 10. If bad reduction, stop.
  return


if __name__ == "__main__":
  main()
