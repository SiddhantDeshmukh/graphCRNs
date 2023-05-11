# Create DataFrames from 3D model snaps for NN training input
from uio_tools.uio_loader import UIOLoader, UIOData
from typing import List, Dict
from sadchem.io.interface.gcrn_uio import load_corresponding_snapshot, quc_to_dict, snap_num_from_path
from sadchem.io.gcrn import load_jcrn_file
import os
import numpy as np
import vaex
from abundances import *


# Global paths
SSD_DIR = "/media/sdeshmukh/Crucial X6"
MODEL_DIR = f"{SSD_DIR}/cobold_runs/chem"
JCRN_DIR = f"{SSD_DIR}/jcrn_runs"


def add_abundance_cols(df, abundances, species):
  for s in species:
    print(f"Adding {abundances[s]} to column A_{s}")
    df[f"A_{s}"] = vaex.vconstant(abundances[s], len(df), dtype=np.float64)

  return df


def write_df(df, output_file: str):
  df.export_parquet(output_file)


def model_to_df(loader: UIOLoader, jcrn_filename: str,
                keys: List, species: List, abundances: Dict, output_file=None,
                x_spacing=2, y_spacing=2, z_spacing=1):
  output_data = {}
  # Load JCRN data and correct CO5BOLD snapshot
  # 'N_spacing' controls the cadence in spatial values. By default, we take
  # every height value, and every other x, y value, resulting in a 4x reduction
  # in size from the original 3D model
  idxs = (slice(None, None), slice(None, None), slice(None, None))
  snap_num = snap_num_from_path(jcrn_filename)
  model = load_corresponding_snapshot(loader, snap_num)
  output_shape = model['rho'].shape
  jcrn_data = load_jcrn_file(jcrn_filename, output_shape=output_shape)
  quc_data = quc_to_dict(model, idxs=idxs)

  # Collect data
  output_data.update({f"{k}_EQ": jcrn_data[k][::z_spacing, ::y_spacing, ::x_spacing]
                      for k in species})
  output_data.update({f"{k}_NEQ": quc_data[k][::z_spacing, ::y_spacing, ::x_spacing]
                      for k in species})
  output_data.update({k: model[k][::z_spacing, ::y_spacing, ::x_spacing]
                      for k in keys})
  # Flatten
  output_data = {k: output_data[k].flatten() for k in output_data.keys()}
  print(output_data.keys())
  # Convert to DF and write
  df = vaex.from_dict(output_data)
  # Add abundance data
  df = add_abundance_cols(df, abundances, species)

  # df.export_parquet(output_file)
  if output_file:
    write_df(df, output_file)

  return df


def models_to_single_df(model_ids: List, snap_nums: List, keys: List,
                        species: List, abundance_sets: List, output_file=None,
                        x_spacing=2, y_spacing=2, z_spacing=1):
  # Foreach model ID and snap num, load the corresponding CO5BOLD & JCRN data,
  # cast into a DF, and append to ongoing data
  # This is equivalent to concaetenating multiple DFs into one set while
  # including abundance information
  dfs = []
  for (model_id, snap_num, abundance_set) in zip(model_ids, snap_nums, abundance_sets):
    loader = UIOLoader(f"{MODEL_DIR}/{model_id}")
    df = model_to_df(loader, f"{JCRN_DIR}/{model_id}/catalyst_{snap_num}.csv", keys,
                     species, abundance_set, output_file=None,
                     x_spacing=x_spacing, y_spacing=y_spacing,
                     z_spacing=z_spacing)
    dfs.append(df)

  full_df = vaex.concat(dfs)
  print(f"{len(full_df)} rows")
  if output_file:
    write_df(full_df, output_file)

  return full_df


def main():
  model_ids = [
      # chem1
      # Dwarf CEMP
      "d3t63g40mm00chem1",
      "d3t63g40mm20chem1",
      "d3t63g40mm30chem1",
      "d3t63g40mm30c20n20o20chem1",
      "d3t63g40mm30c20n20o04chem1",
      # RGB
      "d3t36g10mm00chem1",
      "d3t40g15mm00chem1",
      "d3t40g15mm20chem1",
      "d3t40g15mm30chem1",
      "d3t50g25mm00chem1",
      "d3t50g25mm20chem1",
      "d3t50g25mm30chem1",
      # chem2
      # Dwarf CEMP
      # "d3t63g40mm00chem2",
      # "d3t63g40mm20chem2",
      # "d3t63g40mm30chem2",
      # "d3t63g40mm30c20n20o20chem2",
      # "d3t63g40mm30c20n20o04chem2"
  ]

  # Abundance mappings for models in order above
  dwarf_cemp_abundance_sets = [
      mm00_abundances,
      mm20a04_abundances,
      mm30a04_abundances,
      mm30a04c20n20o20_abundances,
      mm30a04c20n20o04_abundances
  ]

  rgb_abundance_sets = [
    mm00_abundances,
    mm00_abundances,
    mm20a04_abundances,
    mm30a04_abundances,
    mm00_abundances,
    mm20a04_abundances,
    mm30a04_abundances
  ]

  full_abundance_sets = [*dwarf_cemp_abundance_sets, *rgb_abundance_sets]

  # snap nums for JCRN
  dwarf_cemp_snap_nums = ["026", "074", "116"]
  rgb_snap_nums = ["026", "074", "116"]
  # rgb_snap_nums = ["023", "077", "113"]

  out_dir = "../res/df"
  os.system(f"mkdir -p {out_dir}")
  keys = ["density", "temperature"]
  # species = ["C", "H" ,"O", "N", "M", "CO", "CH", "OH", "H2", "NH", "CN", "NO", "O2", "C2", "N2"]
  species = ["C", "H", "O",  "M", "CO", "CH", "OH", "H2"]


  abundance_sets = full_abundance_sets
  snap_nums = dwarf_cemp_snap_nums
  for snap_num in snap_nums:
    models_to_single_df(model_ids, [snap_num] * len(model_ids), keys, species,
                        abundance_sets,
                        output_file=f"{out_dir}/combined_dwarf_cemp_rgb_{snap_num}.parquet")


if __name__ == "__main__":
  main()
