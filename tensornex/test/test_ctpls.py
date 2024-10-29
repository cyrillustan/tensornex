import pytest
import numpy as np
from tensornex.ctpls import ctPLS
from tensornex.tpls import tPLS, fac2tensor
from tensornex.utils import calcR2X


def test_tPLS_equivalence():
    X = np.random.rand(10, 9, 8, 7)
    Y = np.random.rand(10, 5)
    pls0 = tPLS(6)
    pls0.fit(X, Y)
    pls1 = ctPLS(6)
    pls1.fit([X], Y)
    assert np.allclose(pls0.R2X, pls1.R2Xs[0])

@pytest.mark.parametrize("X0dim", [(10, 9, 8, 7), (10, 9, 8, 7, 6)])
@pytest.mark.parametrize("X1dim", [(10, 8, 7), (10, 9, 8, 7)])
@pytest.mark.parametrize("X2dim", [(10, 8), (10, 9, 8)])
def test_ctPLS_dimensions(X0dim, X1dim, X2dim):
    dims = [X0dim, X1dim, X2dim]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(6)
    pls.fit(Xs, Y)
    assert np.allclose(pls.factor_T, pls.transform(Xs))
    #assert all([np.all(np.diff(R2X) >= 0.0) for R2X in pls.R2Xs])
    assert np.all(np.diff(pls.R2Y))


def test_ctPLS_increasing_R2Y_synthetic():
    dims = [(10, 9, 8, 7), (10, 8, 7)]
    n_latent = 4
    Xs = [fac2tensor([np.random.rand(d, n_latent) for d in ds]) for ds in dims]
    Y = np.random.rand(10, 4) @ np.random.rand(5, 4).T
    pls = ctPLS(6)
    pls.fit(Xs, Y)
    # TODO: figure out how to keep R2Xs increasing
    #assert all([np.all(np.diff(R2X) >= 0.0) for R2X in pls.R2Xs])
    assert np.all(np.diff(pls.R2Y))


def test_ctPLS_transform():
    dims = [(10, 9, 8, 7), (10, 8, 7)]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(3)
    pls.fit(Xs, Y)
    assert np.allclose(pls.factor_T, pls.transform(Xs))

def test_ctPLS_missingvals():
    dims = [(10, 9, 8, 7), (10, 8, 7)]
    Xs = [np.random.rand(*d) for d in dims]
    Y = np.random.rand(10, 5)
    pls = ctPLS(3)
    pls.fit(Xs, Y)

    Xs[0][5, 4, 3, 2] = np.nan
    Xs[1][6, 5, 4] = np.nan
    pls_m = ctPLS(3)
    pls_m.fit(Xs, Y)

    assert calcR2X(pls.factor_T, pls_m.factor_T) > 0.9
    # this may actually fail every 1 in 10 trials.