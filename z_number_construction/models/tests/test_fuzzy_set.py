from ..fuzzy_set import *

def test_membership_function():
    fs = FS(0, 1, 2, 3)
    mf = fs.membership_function()
    assert mf(0) == 0
    assert mf(1) == 1
    assert mf(2) == 1
    assert mf(3) == 0
    assert mf(0.5) == 0.5
    assert mf(1.5) == 1
    assert mf(2.5) == 0.5
    assert mf(0.25) == 0.25
    assert mf(2.75) == 0.25
    assert mf(-1) == 0
    assert mf(4) == 0

def test_specificity():
    fs = FS(0, 1, 2, 3)
    assert abs(fs.specificity(3) - 1 / 3) < 1e-6

def test_defuzzify():
    fs = FS(0, 2, 5, 7)
    assert fs.defuzzify('maximum') == 3.5
    assert fs.defuzzify('centroid') == 3.5

    fs = FS(0, 4, 4, 5)
    assert fs.defuzzify('maximum') == 4
    assert fs.defuzzify('centroid') == 3