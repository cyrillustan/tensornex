import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA
import tensorly as tl
from tensorly.cp_tensor import CPTensor, cp_normalize
from tensorly.metrics.factors import congruence_coefficient

from tensornex.tpls import tPLS
from tensornex.utils import fac2tensor


TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 4
N_LATENT = 8

def make_synthetic_test(cp_tensor: tl.cp_tensor, test_samples: int,
                        error: float = 0, seed: int = 215):
    """
    Generates test set from given factors.

    Parameters:
        cp_tensor (tl.cp_tensor): CP tensor
        test_samples (int): samples in testing set
        error (float, default: 0): standard error of added gaussian noise
        seed (int, default: 215): seed for random number generator

    Returns:
        x_test (np.array): tensor of measurements
        y_test (np.array): response variables
        test_tensor (tl.cp_tensor): CP tensor of x_test, y_test
    """
    rng = np.random.default_rng(seed)

    test_factors = cp_tensor.factors
    test_factors[0] = rng.normal(0, 1, size=(test_samples, cp_tensor.rank))
    test_tensor = tl.cp_tensor.CPTensor((None, test_factors))
    test_tensor.y_factor = cp_tensor.y_factor

    x_test = tl.cp_to_tensor(test_tensor)
    x_test += rng.normal(0, error, size=test_tensor.shape)
    y_test = tl.dot(test_tensor.factors[0], cp_tensor.y_factor.T)
    y_test += rng.normal(0, error, size=y_test.shape)

    return x_test, y_test, test_tensor


def import_synthetic(train_dimensions: tuple, n_response: int, n_latent: int,
                     error: float = 0, seed: int = 215):
    """
    Generates synthetic data.

    Parameters:
        train_dimensions (tuple): dimensions of training data
        n_response (int): number of response variables
        n_latent (int): number of latent variables in synthetic data
        error (float, default: 0): standard error of added gaussian noise
        seed (int, default: 215): seed for random number generator

    Returns:
        x (np.array): tensor of measurements
        y (np.array): response variables
        cp_tensor (tl.cp_tensor): CP tensor of x, y
    """
    rng = np.random.default_rng(seed)

    x_factors = [rng.normal(0, 1, size=(train_dimensions[0], n_latent))]
    y_factor = rng.normal(0, 1, size=(n_response, n_latent))

    for dimension in train_dimensions[1:]:
        x_factors.append(rng.normal(0, 1, size=(dimension, n_latent)))

    cp_tensor = tl.cp_tensor.CPTensor((None, x_factors))
    cp_tensor.y_factor = y_factor

    x = tl.cp_to_tensor(cp_tensor)
    x += rng.normal(0, error, size=train_dimensions)

    y = tl.dot(cp_tensor.factors[0], cp_tensor.y_factor.T)
    y += rng.normal(0, error, size=(train_dimensions[0], n_response))

    if y.shape[1] == 1:
        y = y.flatten()

    return x, y, cp_tensor

# Supporting Functions

def _get_standard_synthetic():
    x, y, cp_tensor = import_synthetic(TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT)
    pls = tPLS(N_LATENT)
    pls.fit(x, y)
    return x, y, cp_tensor, pls


# Class Structure Tests

def test_factor_normality():
    x, y, _, pls = _get_standard_synthetic()
    for x_factor in pls.X_factors[1:]:
        assert_allclose(tl.norm(x_factor, axis=0), 1)
    for y_factor in pls.Y_factors[1:]:
        assert_allclose(tl.norm(y_factor, axis=0), 1)


# This method should test for factor hyper-orthogonality; components seem
# very loosely hyper-orthogonal (cut-off of 1E-2 is generous).
def test_factor_orthogonality():
    x, y, _, pls = _get_standard_synthetic()
    x_cp = CPTensor((None, pls.X_factors))
    x_cp = cp_normalize(x_cp)

    for component_1 in range(x_cp.rank):
        for component_2 in range(component_1 + 1, x_cp.rank):
            factor_product = 1
            for factor in x_cp.factors:
                factor_product *= np.dot(
                    factor[:, component_1],
                    factor[:, component_2]
                )
            assert abs(factor_product) < 1E-2


def test_consistent_components():
    x, y, _, pls = _get_standard_synthetic()

    for x_factor in pls.X_factors:
        assert x_factor.shape[1] == N_LATENT

    for y_factor in pls.Y_factors:
        assert y_factor.shape[1] == N_LATENT


# Dimension Compatibility Tests

@pytest.mark.parametrize("idims", [(2, 1), (3, 1), (4, 1), (2, 4), (3, 4), (4, 4)])
def _test_dimension_compatibility(idims):
    x_rank, n_response = idims
    x, y, _ = import_synthetic(tuple([100] * x_rank), n_response, N_LATENT)
    try:
        pls = tPLS(N_LATENT)
        pls.fit(x, y)
    except ValueError:
        raise AssertionError(
            f'Fit failed for {len(x.shape)}-dimensional tensor with '
            f'{n_response} response variables in y'
        )


# Decomposition Accuracy Tests

def test_same_x_y():
    x, _, _ = import_synthetic((100, 100), N_RESPONSE, N_LATENT)
    pls = tPLS(N_LATENT)
    pca = PCA(N_LATENT)

    pls.fit(x, x)
    scores = pca.fit_transform(x)

    assert_allclose(pls.X_factors[0], pls.Y_factors[0], rtol=0, atol=1E-4)
    assert_allclose(pls.X_factors[1], pls.Y_factors[1], rtol=0, atol=1E-4)
    assert congruence_coefficient(pls.X_factors[0], scores)[0] > 0.95
    assert congruence_coefficient(pls.X_factors[1], pca.components_.T)[0] > 0.95


def test_zero_covariance_x():
    x, y, _ = import_synthetic(TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT)
    x[:, 0, :] = 1
    pls = tPLS(N_LATENT)
    pls.fit(x, y)

    assert_allclose(pls.X_factors[1][0, :], 0)



@pytest.mark.parametrize("idims", [(3, 1), (4, 1),  (3, 4), (4, 2)])
def _test_decomposition_accuracy(idims):
    x_rank, n_response = idims
    x, y, true_cp = import_synthetic(tuple([100] * x_rank), n_response, N_LATENT)
    pls = tPLS(N_LATENT)
    pls.fit(x, y)

    for pls_factor, true_factor in zip(pls.X_factors, true_cp.factors):
        assert congruence_coefficient(pls_factor, true_factor)[0] > 0.95

    assert congruence_coefficient(pls.Y_factors[1], true_cp.y_factor)[0] > 0.95


def _test_increasing_R2X(X, Y, info=""):
    tpls = tPLS(12)
    tpls.fit(X, Y)
    assert np.all(np.diff(tpls.R2X) >= 0.0), "R2X is not monotonically increasing"
    assert np.all(np.diff(tpls.R2Y) >= 0.0), \
        f"R2Y is not monotonically increasing. " \
        f"Streak till {np.where(np.diff(tpls.R2Y) <= 0.0)[0][0] + 1}-th component, " \
        f"R2Y = {tpls.R2Y[np.where(np.diff(tpls.R2Y) <= 0.0)[0][0]]}. " \
        f"Y shape = {Y.shape}. {info}"

@pytest.mark.parametrize("n_response", [5, 7, 9])
def test_increasing_R2X_random(n_response):
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, n_response)
    _test_increasing_R2X(X, Y)

@pytest.mark.parametrize("n_response", [5, 7, 9])
def test_increasing_R2X(n_response, n_latent=5):
    X, Y, _ = import_synthetic((20, 8, 6, 4), n_response, n_latent)
    _test_increasing_R2X(X, Y, info=f"n_latent = {n_latent}")


def test_reorient_factors():
    """ Test reorient_factors will not change factorization results """
    X = np.random.rand(20, 10, 8, 6)
    Y = np.random.rand(20, 5)
    tpls = tPLS(4)
    tpls.fit(X, Y)
    Xrecon, Yrecon = fac2tensor(tpls.X_factors), fac2tensor(tpls.Y_factors)
    tpls.reorient_factors()
    assert_allclose(Xrecon - fac2tensor(tpls.X_factors), 0.0)
    assert_allclose(Yrecon - fac2tensor(tpls.Y_factors), 0.0)


def test_transform():
    """ Test transform the original X and Y will give the first factors """
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, 5)
    tpls = tPLS(6)
    tpls.fit(X, Y)
    rord = np.arange(20)
    np.random.shuffle(rord)
    X_scores, Y_scores = tpls.transform(X[rord, :], Y[rord, :])
    assert np.allclose(X_scores, tpls.X_factors[0][rord, :])
    assert np.allclose(Y_scores, tpls.Y_factors[0][rord, :])