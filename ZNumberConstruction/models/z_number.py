from .fuzzy_set import *
from ..z_number_construction import *

import numpy as np
from tqdm import tqdm

class ZNumber:
    def __init__(self, a: FS = None, b: FS = None):
        self.a = a
        self.b = b

    def plot(self, axis_a, axis_b, limits_a=None, **kwargs):
        self.a.plot(axis_a, limits_a, **kwargs)
        self.b.plot(axis_b, (0, 1), **kwargs)

    def plot_over_data(self, axis_a, axis_b, data, limits_a=None, **kwargs):
        vals, bins = np.histogram(data)
        vals = vals / vals.max()
        axis_a.stairs(vals, bins, fill=True)
        if limits_a == None:
            limits_a = min(data.min(), self.a.a), max(data.max(), self.a.d)
        self.a.plot(axis_a, limits_a, **kwargs)
        self.b.plot(axis_b, (0, 1), **kwargs)

    def fit(self, data: np.ndarray,
          u_min=None, u_max=None, u_step=None,
          optimize='specificity', beta=0.5, s_threshold=0.5, c_threshold=0.7,
          defuzzify='centroid', p=2):
        if u_min == None:
            u_min = min(data)
        if u_max == None:
            u_max = max(data)
        if u_step == None:
            u_step = (u_max - u_min) / 10

        best_score = None
        best_subscore = None
        best_A = None
        best_B = None

        # Generate distributions
        distributions = get_distributions(data)
        # Get distances
        euclide_dists, _, _ = calculate_distances(data, distributions, ['euclide'])

        for a in tqdm(np.linspace(u_min, u_max, int((u_max - u_min) / u_step) + 1)):
            for b in np.linspace(a + u_step, u_max, int((u_max - a - u_step) / u_step) + 1):
                for c in np.linspace(b, u_max, int((u_max - b) / u_step) + 1):
                    for d in np.linspace(c + u_step, u_max, int((u_max - c - u_step) / u_step) + 1):
                        A = FS(a, b, c, d)
                        specificity = A.specificity(u_max - u_min)

                        if optimize == 'b' and specificity < s_threshold:
                            continue
                        B = construct_b_part(data, A, distributions, euclide_dists, p=p)
                        if np.isnan(B.a) or np.isnan(B.b) or np.isnan(B.d) or np.isinf(B.a) or np.isinf(B.b) or np.isinf(B.d):
                            continue
                        b_defuzzified = (B.a + B.b + B.d) / 3 if defuzzify == 'centroid' else B.b

                        if optimize == 'specificity' and b_defuzzified < c_threshold:
                            continue

                        if optimize == 'specificity':
                            score = specificity
                            subscore = b_defuzzified
                        elif optimize == 'b':
                            score = b_defuzzified
                            subscore = specificity
                        elif optimize == 'both':
                            score = beta * b_defuzzified + (1 - beta) * specificity
                            subscore = score

                        if best_score == None or best_score < score or best_score == score and best_subscore < subscore:
                            best_score = score
                            best_subscore = subscore
                            best_A = A
                            best_B = B

        self.a = best_A
        self.b = best_B
        return self
