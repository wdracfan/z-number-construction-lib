import numpy as np

def find_center_a1(xs, ys):
  return xs[np.argsort(ys)[-1]]

def find_center_a2(xs, ys, d=0.1):
  max_mean = 0
  max_center = 0
  for left in np.arange(0, 1 - 2*d, 0.01):
    right = left + 2*d
    mean = np.mean(ys[(xs >= left) & (xs <= right)])
    if mean > max_mean:
      max_mean = mean
      max_center = left + d
  return max_center

def find_margins_b1(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  xm_less = np.mean(xs_less)
  ym_less = np.mean(ys_less)
  xm_greater = np.mean(xs_greater)
  ym_greater = np.mean(ys_greater)
  l = c - (1 / (1 - ym_less) * (c - xm_less))
  r = c + (1 / (1 - ym_greater) * (xm_greater - c))
  return l, c, r

def find_margins_b2(xs, ys, c, thr=0):
  xs_less = xs[(xs < c) & (ys > thr)]
  ys_less = ys[(xs < c) & (ys > thr)]
  xs_greater = xs[(xs > c) & (ys > thr)]
  ys_greater = ys[(xs > c) & (ys > thr)]
  l = np.mean((xs_less - c * ys_less) * (xs_less - c)) / np.mean((1 - ys_less) * (xs_less - c))
  r = np.mean((xs_greater - c * ys_greater) * (xs_greater - c)) / np.mean((1 - ys_greater) * (xs_greater - c))
  return l, c, r