from ..pif_construction import *

def test_euclide_distance():
    f = lambda x: x ** 2
    g = lambda x: x ** 3
    assert abs(euclide_distance(f, g, 0, 1) - 1 / 105) < 1e-6

def test_manhattan_distance():
    f = lambda x: x ** 2
    g = lambda x: x ** 3
    assert abs(manhattan_distance(f, g, 0, 1) - 1 / 12) < 1e-6

def test_chebyshev_distance():
    f = lambda x: x ** 2
    g = lambda x: x ** 3
    assert abs(chebyshev_distance(f, g, 0, 1) - 4 / 27) < 1e-6