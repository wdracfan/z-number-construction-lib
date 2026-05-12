from ..pif_construction import *

def test_normal_integrate():
    assert int(normal_integral(0, 1, 2, 3, 0, 1) * 10000) / 10000 == 0.3075

def test_expon_integrate():
    assert int(expon_integral(0, 1, 2, 3, 0, 2) * 10000) / 10000 == 0.4244

def test_uniform_integrate():
    assert int(uniform_integral(0, 1, 2, 3, 0, 3) * 10000) / 10000 == 0.6666