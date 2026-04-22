from math import erf
import numpy as np
import scipy.stats as spstats
import scipy.integrate as spint
from tqdm import tqdm

from ..models import FS

def get_distributions(data: np.ndarray) -> dict[str, list]:
  std = np.std(data)
  min = np.min(data)
  max = np.max(data)

  normals = [spstats.norm(loc, scale) for loc in np.arange(min, max, (max - min) / 20) for scale in np.arange(0.1, 2 * std, std / 10)]
  expons = [spstats.expon(loc, scale) for loc in np.arange(min, max, (max - min) / 20) for scale in np.arange(0.1, 2 * std, std / 10)]
  uniforms = [spstats.uniform(loc, scale) 
            for loc in np.arange(min - (max - min) / 2, min + (max - min) / 2, (max - min) / 20) 
            for scale in np.arange((max - min) / 2, 3 * (max - min) / 2, (max - min) / 10)]
  return {
    'normal': normals,
    'exponential': expons,
    'uniform': uniforms
  }

def phi(z):
    return (1.0 + erf(z / np.sqrt(2.0))) / 2.0

def _normal_integral(a, b, c, d, mu, sigma):
  f = lambda x: 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

  def xint(x0, x1):
    return mu * (phi((x1 - mu) / sigma) - phi((x0 - mu) / sigma)) + sigma ** 2 * (f(x0) - f(x1))
  
  def noxint(x0, x1):
    return phi((x1 - mu) / sigma) - phi((x0 - mu) / sigma)
  
  result = 0
  
  if a != b:
    result += xint(a, b) / (b - a) - a / (b - a) * noxint(a, b)
  if b != c:
    result += noxint(b, c)
  if c != d:
    result += d / (d - c) * noxint(c, d) - xint(c, d) / (d - c)

  return result

def _expon_integral(a, b, c, d, mu, l):
  def xint(x0, x1):
    x0 = max(x0, mu)
    x1 = max(x1, mu)
    return (l * x0 + 1) / l * np.exp(-l * (x0 - mu)) - (l * x1 + 1) / l * np.exp(-l * (x1 - mu))
  
  def noxint(x0, x1):
    x0 = max(x0, mu)
    x1 = max(x1, mu)
    return np.exp(-l * (x0 - mu)) - np.exp(-l * (x1 - mu))

  result = 0
  if a != b:
    result += xint(a, b) / (b - a) - a / (b - a) * noxint(a, b)
  if b != c:
    result += noxint(b, c)
  if c != d:
    result += d / (d - c) * noxint(c, d) - xint(c, d) / (d - c)
  return result

def _uniform_integral(a, b, c, d, u, v):
  def xint(x0, x1):
    x0 = max(x0, u)
    x1 = min(x1, v)
    if x1 <= x0:
      return 0
    return (x1 ** 2 - x0 ** 2) / (2 * (v - u))
  
  def noxint(x0, x1):
    x0 = max(x0, u)
    x1 = min(x1, v)
    if x1 <= x0:
      return 0
    return (x1 - x0) / (v - u)

  result = 0
  if a != b:
    result += xint(a, b) / (b - a) - a / (b - a) * noxint(a, b)
  if b != c:
    result += noxint(b, c)
  if c != d:
    result += d / (d - c) * noxint(c, d) - xint(c, d) / (d - c)
  return result

def calculate_integrals(a_part: FS, distributions: dict[str, list]):
  integrals = {}
  mf = a_part.membership_function()
  for key in distributions:
    if key == 'normal':
      integrals[key] = [_normal_integral(a_part.a, a_part.b, a_part.c, a_part.d, distribution.stats()[0], np.sqrt(distribution.stats()[1])) for distribution in distributions[key]]
    elif key == 'exponential':
      integrals[key] = [_expon_integral(a_part.a, a_part.b, a_part.c, a_part.d, distribution.stats()[0] - np.sqrt(distribution.stats()[1]), 1 / np.sqrt(distribution.stats()[1])) for distribution in distributions[key]]
    elif key == 'uniform':
      integrals[key] = [_uniform_integral(a_part.a, a_part.b, a_part.c, a_part.d, (distribution.stats()[0] * 2 - np.sqrt(12 * distribution.stats()[1])) / 2, (distribution.stats()[0] * 2 + np.sqrt(12 * distribution.stats()[1])) / 2) for distribution in distributions[key]]
    else: # невозможная ветка
      integrals[key] = []
      for distribution in distributions[key]:
        integrals[key].append(spint.quad(lambda x: distribution.pdf(x) * mf(x), a_part.a, a_part.d)[0])
    integrals[key] = np.array(integrals[key])
  return integrals

def convert_distances_into_similarities(distances: dict[str, list], p: float = 2) -> dict[str, dict[str, list]]:
  return {
    # 'exp': {key: (np.exp(-distances[key])) ** p for key in distances},
    # 'sigmoid': {key: (2 / (1 + np.exp(distances[key]))) ** p for key in distances},
    'F-based': {key: (np.min(distances[key]) / distances[key]) ** p for key in distances}
  }

#@log_with_timestamp
def calculate_distances(data: np.ndarray, distributions: dict[str, list], functions: list[str] = ['euclide']):
  histogram = histogram_function(data)

  def euclide_distance(f, g, l, r):
    return spint.quad(lambda x: (f(x) - g(x)) ** 2, l, r)[0]

  def chebyshev_distance(f, g, l, r):
    xs = np.arange(l, r, (r - l) / 1000)
    ys = [np.abs(f(x) - g(x)) for x in xs]
    return np.max(ys)

  def manhattan_distance(f, g, l, r):
    return spint.quad(lambda x: np.abs(f(x) - g(x)), l, r)[0]
  
  # Calculate distances
  euclide_dists = {}
  manhattan_dists = {}
  chebyshev_dists = {}
  for key in distributions:
    if 'euclide' in functions:
      euclide_dists[key] = []
    if 'manhattan' in functions:
      manhattan_dists[key] = []
    if 'chebyshev' in functions:
      chebyshev_dists[key] = []
    for distribution in tqdm(distributions[key]):
      if 'euclide' in functions:
        euclide_dists[key].append(euclide_distance(distribution.pdf, histogram, data.min(), data.max()))
      if 'manhattan' in functions:
        manhattan_dists[key].append(manhattan_distance(distribution.pdf, histogram, data.min(), data.max()))
      if 'chebyshev' in functions:
        chebyshev_dists[key].append(chebyshev_distance(distribution.pdf, histogram, data.min(), data.max()))
    if 'euclide' in functions: 
      euclide_dists[key] = np.array(euclide_dists[key])
    if 'manhattan' in functions:
      manhattan_dists[key] = np.array(manhattan_dists[key])
    if 'chebyshev' in functions:
      chebyshev_dists[key] = np.array(chebyshev_dists[key])

  return euclide_dists, manhattan_dists, chebyshev_dists

def histogram_function(data):
    heights, bins = np.histogram(data, density=True)
    def h(x):
      if x < bins[0]:
        return 0
      if x >= bins[-1]:
        return 0
      for i in range(len(heights)):
        if bins[i] <= x < bins[i + 1]:
          return heights[i]
      return -1
    return h