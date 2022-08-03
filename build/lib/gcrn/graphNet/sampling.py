# Tools to sample arrays for points
from typing import List, Tuple
import numpy as np


def sample_nd(bounds: List[Tuple], num_points: int):
  # Uniform sampling in 'bounds' for an n-tuple
  # 'bounds' is a list of (min, max) bounds, the output shape of samples
  # will be 'num_samples' x len(bounds)
  # TODO:
  # Reject duplicates
  samples = np.empty(shape=(len(bounds), num_points))
  for i, (low, high) in enumerate(bounds):
    samples[i] = np.random.uniform(low, high, num_points)

  return samples.T
