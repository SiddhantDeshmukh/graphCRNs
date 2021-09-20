# Functions for evaluating rates with temperature limits
from typing import Callable, Union
import numpy as np
from scipy.special import expit


def accuracy_to_scale(accuracy: str) -> int:
  # From a UMIST12 accuracy (A-E), return a scale for the limit
  return scale_dict[accuracy]


def lower_limit_sigmoid(temperature: float, Tmin: float, scale=1) -> float:
  if (temperature - Tmin) < -225:
    return 0
  else:
    return expit(scale * (temperature - Tmin))


def upper_limit_sigmoid(temperature: float, Tmax: float, scale=1) -> float:
  if (temperature - Tmax) > 225:
    return 0
  else:
    return expit(scale * (Tmax - temperature))


def fading_function(rate: Union[Callable, float], temperature: float,
                    Tmin: float, Tmax: float,
                    lower_scale=1, upper_scale=1) -> float:
  if callable(rate):
    return rate(temperature) *\
        (lower_limit_sigmoid(temperature, Tmin, lower_scale) +
         upper_limit_sigmoid(temperature, Tmax, upper_scale) - 1)
  else:
    return rate * (lower_limit_sigmoid(temperature, Tmin, lower_scale) +
                   upper_limit_sigmoid(temperature, Tmax, upper_scale) - 1)


def cutoff_function(rate: Callable, temperature: float,
                    Tmin: float, Tmax: float) -> float:
  # Define the rate only between specified limits, zero otherwise
  if isinstance(temperature, np.ndarray):
    mask = (temperature >= Tmin) & (temperature <= Tmax)
    out = np.zeros(len(temperature))
    out[mask] = rate(temperature[mask])
  else:
    out = rate(temperature) if Tmin <= temperature <= Tmax else 0

  return out


limit_dict = {
    'weak': fading_function,
    'medium': fading_function,
    'strong': fading_function,
    'sharp': cutoff_function
}

scale_dict = {
    'weak': 1,
    'medium': 25,
    'strong': 50,
    'A': 50,
    'B': 25,
    'C': 10,
    'D': 5,
    'E': 1
}
