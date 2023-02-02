# Implementation of GCN Regressor in PyTorch
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch_geometric.nn import GCNConv


class GCNReg(torch.nn.Module):
  def __init__(self, hidden_channels, num_features) -> None:
    super(GCNReg, self).__init__()
    torch.manual_seed(42)
    self.conv1 = GCNConv(num_features, hidden_channels)
    self.linear1 = torch.nn.Linear(hidden_channels, len(num_features) - 2)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = x.relu()
    x = self.linear1(x)
    return x


def train(network: GCNReg, train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader, num_iter=100):
  # Train 'network' for 'num_iter' iterations
  # Reject training sample if it exists
  # Add training sample to 'train_data'
  pass


def main():
  pass


if __name__ == "__main__":
  main()

# Input: [n_i..., T, t]  (initial number densities, temperature & time at end)
# Output: [n_f...]  (final number densities)
# Aim:
# - reliably reproduce kinetics for a given network
# - regularise and optimise hyperparameters to learn best mapping
#   What would be the best network structure for this?
"""
TODO:
- Create GCN Regressor class
- Setup GCNReg to have same setup as CO network
- Training loop to initialise number densities from density, create GCNReg input
  and pass in, immediately calc the GCRN final 'n' to compare (stability?)
- Evaluate & test
- If it works well, add the complexity reduction layer (more in-depth since we
  also want to analyse the trends that are created)
"""

# Ideas:
# - Start with a GCN that has the same structure as CO network and just train
# it to reproduce inputs. Is it faster than the GCRN evaluation? What's
# accuracy like across the (rho, T, t) grid?
# - We will start with n_species nodes and edges between all of them, the next
# layer needs to be a hyperparameter for simplification (we want to move from
# R^{n_species} to some lower-dimensional mapping without losing much accuracy)
