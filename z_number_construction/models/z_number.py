from .fuzzy_set import *
from ..z_number_construction import *

import numpy as np
from time import time
from tqdm import tqdm
import warnings

class ZNumber:
    '''Represents a Z-number (A, B), with both A and B being trapezoidal fuzzy sets.'''
    def __init__(self, a: FS = None, b: FS = None):
        self.a = a
        self.b = b

    def plot(self, axis_a, axis_b, limits_a=None, **kwargs):
        '''Plots the membership functions of the Z-number components on the given axes.'''
        self.a.plot(axis_a, limits_a, **kwargs)
        self.b.plot(axis_b, (0, 1), **kwargs)

    def plot_over_data(self, axis_a, axis_b, data, limits_a=None, **kwargs):
        '''Plots the membership functions of the Z-number components on the given axes, overlaid with the data histogram.'''
        vals, bins = np.histogram(data)
        vals = vals / vals.max()
        axis_a.stairs(vals, bins, fill=True)
        if limits_a == None:
            limits_a = min(data.min(), self.a.a), max(data.max(), self.a.d)
        self.a.plot(axis_a, limits_a, **kwargs)
        self.b.plot(axis_b, (0, 1), **kwargs)

    def fit(self, data: np.ndarray,
            p=2, distance='euclide',
            u_min=None, u_max=None,
            q=None, allow_discontinuities=False,
            optimize='specificity', beta=0.5, s_threshold=0.5, c_threshold=0.7,
            defuzzify='centroid'):
        '''
        Constructs a Z-number to represent the given data. Does not return anything, but sets the `a` and `b` attributes of the Z-number.
        
        Parameters:
            data (np.ndarray): Data set to be represented with a Z-number.
            p (float): Regulates the width of the B-part, `2` by default.
            distance (str): Distance metric to be used for data-based possibility distribution construction, `euclide` or `manhattan` or `chebyshev`, `euclide` by default.
            u_min (float): Left bound of the universal set, minimum of the data by default.
            u_max (float): Right bound of the universal set, maximum of the data by default.
            q (int or str): Number of points inside the universal set for A-part 'basis points' enumeration, calculated by Sturges formula by default.
            allow_discontinuities (bool): Whether to allow A-part FS(a,b,c,d) with a=b or c=d, `False` by default.
            optimize (str): Optimization criterion, `specificity` or `b` or `both`, `specificity` by default.
            beta (float): Weight of defuzzified B-part for the `both` criterion, `0.5` by default.
            s_threshold (float or str): Threshold for A-part specificity for the `b` criterion or `cum` for calculating based on data cumulativeness, `0.5` by default.
            c_threshold (float): Threshold for defuzzified B-part for the `specificity` criterion, `0.7` by default.
            defuzzify (str): Method of defuzzification, `centroid` or `maximum`, `centroid` by default.
        '''
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if u_min == None:
                u_min = min(data)
            if u_max == None:
                u_max = max(data)
            if q == None:
                q = np.floor(np.log2(len(data))) + 1 # Sturges formula
            u_step = (u_max - u_min) / q

            if s_threshold == 'cum':
                def cumulativeness(data: np.ndarray, l: float, r: float, bins: int) -> float:
                    hist, bins = np.histogram(data, bins=bins, range=(l, r))
                    interval = bins[1] - bins[0]
                    return hist.sum() * interval / ((r - l) * hist.max())
            
                s_threshold = 1 - cumulativeness(data, u_min, u_max, 'sturges')

            best_score = None
            best_subscore = None
            best_A = None
            best_B = None

            print('Converting data set into possibility distribution...')
            time_data_processing_start = time()

            # Generate distributions
            distributions = get_distributions(data)
            # Get distances
            dists = calculate_distances(data, distributions, distance)

            time_data_processing_end = time()
            print(f'Converting data set into possibility distribution [DONE], time elapsed: {time_data_processing_end - time_data_processing_start:.2f} s.')

            u_step_discontinuity = 0 if allow_discontinuities else u_step

            print('Calculating A-part and B-part...')
            time_parts_start = time()

            for a in tqdm(np.linspace(u_min, u_max, int((u_max - u_min) / u_step) + 1)):
                for b in np.linspace(a + u_step_discontinuity, u_max, int((u_max - a - u_step) / u_step) + 1):
                    for c in np.linspace(b, u_max, int((u_max - b) / u_step) + 1):
                        for d in np.linspace(c + u_step_discontinuity, u_max, int((u_max - c - u_step) / u_step) + 1):
                            A = FS(a, b, c, d)
                            specificity = A.specificity(u_max - u_min)

                            if optimize == 'b' and specificity < s_threshold:
                                continue
                            B = construct_b_part(data, A, distributions, dists, p=p)
                            if np.isnan(B.a) or np.isnan(B.b) or np.isnan(B.d) or np.isinf(B.a) or np.isinf(B.b) or np.isinf(B.d):
                                continue
                            b_defuzzified = B.defuzzify(method=defuzzify)

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

            time_parts_end = time()
            print(f'Calculating A-part and B-part [DONE], time elapsed: {time_parts_end - time_parts_start:.2f} s.')
            
            self.a = best_A
            self.b = best_B
            return self