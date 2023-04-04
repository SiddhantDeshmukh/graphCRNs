# Create DataFrames from 3D model snaps for NN training input
from uio_tools.uio_loader import UIOLoader, UIOData
from typing import List
from sadchem.io.interface.gcrn_uio import load_corresponding_snapshot, quc_to_dict, snap_num_from_path
from sadchem.io.gcrn import load_jcrn_file
import os
import numpy as np
import vaex


def model_to_df(loader: UIOLoader, jcrn_filename: str,
                keys: List, species: List, output_file: str):
  output_data = {}
  # Load JCRN data and correct CO5BOLD snapshot
  snap_num = snap_num_from_path(jcrn_filename)
  model = load_corresponding_snapshot(loader, snap_num)
  jcrn_data = load_jcrn_file(jcrn_filename)
  quc_data = quc_to_dict(model)

  # Collect data
  output_data.update({f"{k}_EQ": jcrn_data[k] for k in species})
  output_data.update({f"{k}_NEQ": quc_data[k] for k in species})
  output_data.update({k: model[k] for k in keys})
  # Flatten
  output_data = {k: output_data[k].flatten() for k in output_data.keys()}
  print(output_data.keys())
  # Convert to DF and write
  df = vaex.from_dict(output_data)
  df.export_parquet(output_file)

def main():
  SSD_DIR = "/media/sdeshmukh/Crucial X6"
  MODEL_DIR = f"{SSD_DIR}/cobold_runs/chem"
  JCRN_DIR = f"{SSD_DIR}/jcrn_runs"

  model_ids = [
    "d3t63g40mm00chem1",
    "d3t63g40mm20chem1",
    "d3t63g40mm30chem1",
    "d3t63g40mm30c20n20o20chem1",
    "d3t63g40mm30c20n20o04chem1"
    # "d3t63g40mm00chem2",
    # "d3t63g40mm20chem2",
    # "d3t63g40mm30chem2",
    # "d3t63g40mm30c20n20o20chem2",
    # "d3t63g40mm30c20n20o04chem2"
  ]

  snap_num = "074"
  out_dir = "../res/df"
  os.system(f"mkdir -p {out_dir}")
  keys = ["density", "temperature"]
  # species = ["C", "H" ,"O", "N", "M", "CO", "CH", "OH", "H2", "NH", "CN", "NO", "O2", "C2", "N2"]
  species = ["C", "H" ,"O",  "M", "CO", "CH", "OH", "H2"]

  for model_id in model_ids:
    loader = UIOLoader(f"{MODEL_DIR}/{model_id}")
    model_to_df(loader, f"{JCRN_DIR}/{model_id}/catalyst_{snap_num}.csv", keys, species,
                f"{out_dir}/{model_id}_{snap_num}.parquet")

if __name__ == "__main__":
  main()