# Plot training losses for models of interest
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


def read_history(filepath: str):
  # Read a history .pkl file and return the dictionary
  with open(filepath, "rb") as infile:
    return pickle.load(infile)


def main():
  plt.style.use("standard-scientific")
  # Define paths
  models = {
    # abu
    "MLP (Abu.)": "TF_MLP6-8_abu_combined_rgb_3d.pkl",
    "CNN (Abu.)": "TF_CNN6-8_abu_combined_rgb_3d.pkl",
    "EncDec (Abu.)": "TF_EncDec6-8_abu_combined_rgb_3d.pkl",
    # logn
    "MLP (log(n))": "TF_MLP6-8_logn_combined_rgb_3d.pkl",
    "CNN (log(n))": "TF_CNN6-8_logn_combined_rgb_3d.pkl",
    "EncDec (log(n))": "TF_EncDec6-8_logn_combined_rgb_3d.pkl",
  }

  data_dir = "./train_history"
  for k in models.keys():
    models[k] = f"{data_dir}/{models[k]}"

  # Load training histories
  histories = {k: read_history(v) for k, v in models.items()}

  # Plot 2x3 subplots; top row is abu, bottom row is logn
  fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7.2), sharey=True)
  top_keys = ["MLP (Abu.)", "CNN (Abu.)", "EncDec (Abu.)"]
  bot_keys = ["MLP (log(n))", "CNN (log(n))", "EncDec (log(n))"]

  for i, keys in enumerate([top_keys, bot_keys]):
    for j, k in enumerate(keys):
      mse = histories[k]["loss"]
      val_mse = histories[k]["val_loss"]
      mae = histories[k]["mae"]
      val_mae = histories[k]["val_mae"]

      ax[i, j].plot(mae, lw=2, label="MAE")
      ax[i, j].plot(val_mae, lw=1, ls='-', label="Val. MAE")
      ax[i, j].plot(mse, lw=2, label="MSE")
      ax[i, j].plot(val_mse, lw=1, ls='-', label="Val. MSE")

      # Aesthetics
      if j == 0:
        ax[i, j].set_ylabel("Loss", fontsize="medium")
      ax[i, j].set_xlabel("Epochs", fontsize="medium")
      ax[i, j].set_yscale("log")
      ax[i, j].set_title(k, fontsize="large")
      ax[i, j].legend(fontsize="small")

  plt.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.07,
                      wspace=0., hspace=0.33)
  out_dir = "/home/sdeshmukh/Documents/chemicalAnalysis/out/figs/nn_loss"
  fig.savefig(f"{out_dir}/loss_combined_rgb_3d.png",
              bbox_inches="tight")
  plt.show()

if __name__ == "__main__":
  main()