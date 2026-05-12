import numpy as np

from ..models import FS
from .pif_construction import convert_distances_into_similarities, calculate_integrals
from .triangular_approximation import *

def construct_b_part(data: np.ndarray, a_part: FS, distributions: dict[str, list], dists: dict[str, np.ndarray], p: float = 2, 
                     center_method: str = 'a2', margins_method: str = 'b1', thr: float = 0.4) -> FS:
    # Get integrals
    integrals = np.concatenate(list(calculate_integrals(a_part, distributions).values()))

    # Convert distances into similarities
    euclide_similarities = np.concatenate(list(convert_distances_into_similarities(dists, p)['F-based'].values()))

    xs = np.array(integrals)
    ys = np.array(euclide_similarities)

    if center_method == 'a1':
        m = find_center_a1(xs, ys)
    elif center_method == 'a2':
        m = find_center_a2(xs, ys)
    else:
        raise ValueError('Unknown center method, must be "a1" or "a2".')
    
    if margins_method == 'b1':
        l, m, r = find_margins_b1(xs, ys, m, thr=thr)
    elif margins_method == 'b2':
        l, m, r = find_margins_b2(xs, ys, m, thr=thr)
    else:
        raise ValueError('Unknown margins method, must be "b1" or "b2".')
    
    l = max(0, l)
    r = min(1, r)
    return FS(l, m, m, r)