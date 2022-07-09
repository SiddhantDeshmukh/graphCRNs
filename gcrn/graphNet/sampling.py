# Tools to sample arrays for points
import numpy as np


def clean_array_1d(arr1: np.ndarray, arr2: np.ndarray, threshold=0.):
  # Change any elements from 'arr1' that are within 'arr2' + 'threshold'
  # distance to 'nan'
  mask = (arr1 arr2 + threshold)


def clean_array_nd(arr1: np.ndarray, arr2: np.ndarray, threshold=[]):
  # Similar to 'clean_array_1d()' but for multi-dimensional arrays
  # 'threshold' needs to be a threshold with dimensions as len(arr1)


def sample_uniform(low: float, high: float, num_samples: int,
                   blacklist_samples=[], blacklist_threshold=0.,
                   allow_duplicates=False):
  # Sample 'num_samples' points from 'arr', but reject any samples that are in
  # 'blacklist_samples' or are within 'blacklist_threshold' distance of a sample
  # in 'blacklist_samples'
  # If 'allow_duplicates' is 'False' (default), add each new sample to
  # 'blacklist_samples' during the sampling process
  samples = np.random.uniform(low, high, size=(num_samples))
  # No blacklisting
  if not blacklist_samples:
    return samples
  else:
    # Clean 'samples'

    # Resample if necessary
