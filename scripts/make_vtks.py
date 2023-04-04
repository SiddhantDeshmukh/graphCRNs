# Make VTK files for 3D viz
from sadchem.io.convert_to_vtk import chem_file_to_VTK
import os


def main():
  model_ids = [
    "d3t63g40mm00chem2",
    "d3t63g40mm20chem2",
    "d3t63g40mm30chem2",
    "d3t63g40mm30c20n20o20chem2",
    "d3t63g40mm30c20n20o04chem2"
  ]

  snap_num = "077"
  out_dir = "../res/vtk"
  os.system(f"mkdir -p {out_dir}")
  keys = ["density", "temperature"]
  species = ["C", "H" ,"O", "N", "M", "CO", "CH", "OH", "H2", "NH", "CN", "NO", "O2", "C2", "N2"]
  
  for model_id in model_ids:
    chem_file_to_VTK(model_id, f"catalyst_{snap_num}.csv", 
                     f"{out_dir}/{model_id}_{snap_num}",
                     keys=keys,
                     species=species)

if __name__ == "__main__":
  main()