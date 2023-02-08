import numpy as np
from chop import chop
import matplotlib.pyplot as plt
from scipy.io import readsav
from typing import Dict
from scipy.optimize import curve_fit


consts = {
    # all in cgs'
    'c': 29979245800.0,
    'h': 6.62607015e-27
}


def minmax(arr: np.ndarray):
  return (np.nanmin(arr), np.nanmax(arr))


def generate_wavelengths(wl_min: float, wl_max: float, num=500, conversion_fac=1):
  # 'conversion_fac' is for unit conversions, e.g. if 'wl_min', 'wl_max' is in
  # Angstroms, conversion_fac = 1e-8 converts output to cm (as expected by chop)
  wavelengths = np.linspace(wl_min, wl_max, num=num) * conversion_fac
  return wavelengths


def plot(x: np.ndarray, y: np.ndarray, ax=None, **plot_kwargs):
  if ax is None:
    _, ax = plt.subplots()
  ax.plot(x, y, **plot_kwargs)

  return ax


def plot_cross_sections(wavelengths: np.ndarray, cross_sections: np.ndarray,
                        ax=None, **plot_kwargs):
  return plot(wavelengths, cross_sections, ax=ax, **plot_kwargs)


def plot_intensity(wavelengths: np.ndarray, intensity: np.ndarray,
                   scale_fac=3e-31, ax=None, **plot_kwargs):
  # 'scale_fac' used to put intensity on same scale as cross-sections
  return plot(wavelengths, intensity * scale_fac, ax=ax, **plot_kwargs)


def read_sed_sav(filename: str):
  # Read SED out of IDLSAVE file and output useful data
  data = readsav(filename)
  key = list(data.keys())[0]
  # Unpack dictionary
  dtype_keys = data[key].dtype.names
  sed_data = {k: data[key][k] for k in dtype_keys}

  return sed_data


def compute_intensity(sed_data: Dict, wmu: np.ndarray):
  wavelengths = sed_data['CLAM'][0]
  wl_conversion = consts['c'] * 1e8 / wavelengths**2  # convert to per A
  n_photon_conversion = 1e-8 / \
      (consts['h'] * consts['c'])  # conversion to photon num
  # estimate assumes no incoming radiation
  intensity = 0.5 * wl_conversion * n_photon_conversion * \
      (np.sum(sed_data['IMUNU'][0], axis=2).T @ wmu)

  return intensity


def chop_arr(temperature: float, wavelengths: np.ndarray, conversion_fac=1):
  # Call 'chop' for every element in 'wavelengths'
  # 'conversion_fac' used for unit conversion, 'chop' expects cm, so e.g. if
  # 'wavelengths' is in Angstroms, 'conversion_fac' = 1e-8
  return np.array([chop(temperature, w) for w in wavelengths * conversion_fac])


def compute_photo_rate(wavelengths: np.ndarray, cross_sections: np.ndarray,
                       intensity: np.ndarray):
  rate = 4. * np.pi * np.trapz(intensity * cross_sections, x=wavelengths)
  return rate


def rates_by_temperature(temperatures: np.ndarray, sed_data: Dict,
                         wmu: np.ndarray):
  # Compute a photodissociation rate foreach temperature in 'temperatures'
  wavelengths = sed_data["CLAM"][0]
  rates = []
  for T in temperatures:
    ch_cross_sections = chop_arr(T, wavelengths, conversion_fac=1e-8)
    intensity = compute_intensity(sed_data, wmu)
    rate = compute_photo_rate(wavelengths, ch_cross_sections, intensity)
    rates.append(rate)

  return np.array(rates)


def sigmoid_like(x: np.ndarray, a: float, b: float, c: float):
  return a / (b + np.exp(-((x/1e3) + c)))


def linear(x: np.ndarray, a: float, b: float):
  return a*x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float):
  return a*x**2 + b*x + c


def cubic(x: np.ndarray, a: float, b: float, c: float, d: float):
  return a*x**3 + b*x**2 + c*x + d


def quartic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float):
  return a*x**4 + b*x**3 + c*x**2 + d*x + e


def chi_squared(y1: np.ndarray, y2: np.ndarray):
  return np.sum((y1 - y2)**2 / y2)


def fit_temperature_dependent_rate(temperatures: np.ndarray, rates: np.ndarray):
  # Fit a temperature-dependent rate for a given (T, rate) profile
  # Visually, rate seems to saturate at T_rad = T_eff for a given SED
  # Metallicity leads to an offset, but also a change in low-T regime
  # (below 3500 K)
  # f = sigmoid_like
  # f = linear
  # f = quadratic
  f = cubic  # best so far!
  # f = quartic
  popt, pcov = curve_fit(f, np.log10(temperatures), rates)
  fitted_rates = f(np.log10(temperatures), *popt)

  print(chi_squared(fitted_rates, rates))

  return fitted_rates, popt, pcov


def metallicity_sed_plot(T_rad=4000.):
  # Create wavelength array
  mm00_data = read_sed_sav("d3t63g45mm00n01.0253327.idlsave")
  mm30_data = read_sed_sav("d3t63g45mm30n01.0311420.idlsave")
  metallicities = ["0.0", "-3.0"]
  # 'wavelengths' and 'T_rad' are the same for both data dicts
  wmu = np.array([0.0222222222, 0.1333059908,
                  0.2248893420, 0.2920426836, 0.3275397612])
  ch_cross_sects = chop_arr(T_rad, mm00_data["CLAM"][0], conversion_fac=1e-8)
  fig, ax = plt.subplots()
  for metallicity, data in zip(metallicities, [mm00_data, mm30_data]):
    wavelengths = data["CLAM"][0]  # in Angstroms
    intensity = compute_intensity(data, wmu)
    plot_intensity(wavelengths, intensity, scale_fac=3e-31, ax=ax,
                   **{"label": f"Intensity ([Fe/H] = {metallicity})"})
    rate = compute_photo_rate(wavelengths, ch_cross_sects, intensity)
    timescale = 1 / rate

    # Print summary
    print(f"[Fe/H] = {metallicity}")
    print(f"Rate: {rate:1.2e} [1/s]")
    print(f"Timescale: {timescale:1.2e} [s]")

  # CH photo-dissociation cross-sections for given temperature
  plot_cross_sections(wavelengths, ch_cross_sects, ax=ax,
                      **{"ls": "-", "label": f"CH cross-section"})
  # Aesthetics
  ax.legend()
  ax.set_xlim(0., 6500.)
  ax.set_title(f"T = {T_rad:.0f} [K]")


def temperature_rates_plot():
  temperatures = np.linspace(1000., 8999., num=100)  # Kurucz limits
  wmu = np.array([0.0222222222, 0.1333059908,
                  0.2248893420, 0.2920426836, 0.3275397612])
  mm00_data = read_sed_sav("d3t63g45mm00n01.0253327.idlsave")
  mm30_data = read_sed_sav("d3t63g45mm30n01.0311420.idlsave")

  mm00_rates = rates_by_temperature(temperatures, mm00_data, wmu)
  mm30_rates = rates_by_temperature(temperatures, mm30_data, wmu)
  fitted_rates_mm00, popt, pcov = fit_temperature_dependent_rate(temperatures,
                                                                 mm00_rates)
  print(popt)
  fitted_rates_mm30, popt, pcov = fit_temperature_dependent_rate(temperatures,
                                                                 mm30_rates)
  print(popt)

  # top panel is rates, bottom panel is timescales
  fig, axes = plt.subplots(2, 1)
  l = axes[0].plot(temperatures, mm00_rates,
                   label="[Fe/H] = 0.0", marker='o', ls='none', mfc='none')
  axes[0].plot(temperatures, fitted_rates_mm00,
               ls="-", c=l[0].get_color())
  l = axes[0].plot(temperatures, mm30_rates,
                   label="[Fe/H] = -3.0", marker='o', ls='none', mfc='none')
  axes[0].plot(temperatures, fitted_rates_mm30, ls="-", c=l[0].get_color())

  l = axes[1].plot(temperatures, 1 / mm00_rates,
                   label="[Fe/H] = 0.0", marker='o', ls='none', mfc='none')
  axes[1].plot(temperatures, 1 / fitted_rates_mm00,
               ls='-', c=l[0].get_color())
  l = axes[1].plot(temperatures, 1 / mm30_rates,
                   label="[Fe/H] = -3.0", marker='o', ls='none', mfc='none')
  axes[1].plot(temperatures, 1 / fitted_rates_mm30,
               ls='-', c=l[0].get_color())

  # Aesthetics
  for ax in axes:
    ax.legend()
    ax.set_xlabel("Temperature [K]")

  axes[0].set_ylabel(r"Rate [s$^{-1}$]")
  axes[1].set_ylabel("Timescale [s]")


def main():
  plt.style.use("standard-scientific")
  temperature_rates_plot()
  plt.show()


if __name__ == "__main__":
  main()


"""
TODO:
  - Fit temperature-dependent rate law
"""
