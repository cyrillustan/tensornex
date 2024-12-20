"""
Unit test file.
"""
import numpy as np
import tensorly as tl
import warnings
from tensorly.cp_tensor import _validate_cp_tensor
from tensorly.random import random_cp
from ..cp import delete_component, calcR2X, sort_factors, perform_CP



def createCube(missing=0.0, size=(10, 20, 25)):
    s = np.random.gamma(2, 2, np.prod(size))
    tensor = s.reshape(*size)
    if missing > 0.0:
        tensor[np.random.rand(*size) < missing] = np.nan
    return tensor

def test_cp():
    # Test that the CP decomposition code works.
    tensor = createCube(missing=0.2, size=(10, 20, 25))
    fac3 = perform_CP(tensor, r=3)
    fac6 = perform_CP(tensor, r=6)
    assert fac3.R2X < fac6.R2X
    assert fac3.R2X > 0.0
    if fac3.R2X < 0.67:
        warnings.warn("CP (r=3) with 20% missingness, R2X < 0.67 (expected)" + str(fac3.R2X))

    # test case where mode size < rank
    tensor2 = createCube(missing=0.2, size=(10, 4, 50))
    fac23 = perform_CP(tensor2, r=3)
    fac26 = perform_CP(tensor2, r=6)
    assert fac23.R2X < fac26.R2X
    assert fac23.R2X > 0.0


def test_delete():
    """ Test deleting a component results in a valid tensor. """
    tOrig = createCube(missing=0.2, size=(10, 20, 25))
    facT = perform_CP(tOrig, r=4)
    fullR2X = calcR2X(facT, tOrig)

    for ii in range(facT.rank):
        facTdel = delete_component(facT, ii)
        _validate_cp_tensor(facTdel)
        delR2X = calcR2X(facTdel, tOrig)
        assert delR2X < fullR2X


def test_sort():
    """ Test that sorting does not affect anything. """
    tOrig = createCube(missing=0.2, size=(10, 20, 25))

    tFac = random_cp(tOrig.shape, 3)

    R2X = calcR2X(tFac, tOrig)
    tRec = tl.cp_to_tensor(tFac)

    tFac = sort_factors(tFac)
    sR2X = calcR2X(tFac, tOrig)
    stRec = tl.cp_to_tensor(tFac)

    np.testing.assert_allclose(R2X, sR2X)
    np.testing.assert_allclose(tRec, stRec)
