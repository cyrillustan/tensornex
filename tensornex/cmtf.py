"""
Coupled Matrix Tensor Factorization used in Molecular Systems Biology paper
"""

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from copy import deepcopy
from tqdm import tqdm
from .SVD_impute import IterativeSVD
from .linalg import mlstsq, calcR2X_TnB


tl.set_backend('numpy')


def buildMat(tFac):
    """ Build the matrix in CMTF from the factors. """
    if hasattr(tFac, 'mWeights'):
        return tFac.factors[0] @ (tFac.mFactor * tFac.mWeights).T
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)
    vTop, vBottom = 0.0, 0.0

    if tIn is not None:
        vs = calcR2X_TnB(tIn, tl.cp_to_tensor(tFac))
        vTop += vs[0]
        vBottom += vs[1]
    if mIn is not None:
        recon = tFac if isinstance(tFac, np.ndarray) else buildMat(tFac)
        vs = calcR2X_TnB(mIn, recon)
        vTop += vs[0]
        vBottom += vs[1]
    return 1.0 - vTop / vBottom


def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors])

    if hasattr(tFac, 'mFactor'):
        deg += tFac.mFactor.size

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the types to be positive
    tMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[1] *= tMeans[np.newaxis, :]
    tFac.factors[2] *= tMeans[np.newaxis, :]

    # Flip the cytokines to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]

    if hasattr(tFac, 'mFactor'):
        tFac.mFactor *= rMeans[np.newaxis, :]
    return tFac


def sort_factors(tFac):
    """ Sort the components from the largest variance to the smallest. """
    tensor = deepcopy(tFac)

    # Variance separated by component
    norm = np.copy(tFac.weights)
    for factor in tFac.factors:
        norm *= np.sum(np.square(factor), axis=0)

    # Add the variance of the matrix
    if hasattr(tFac, 'mFactor'):
        norm += np.sum(np.square(tFac.factors[0]), axis=0) * np.sum(np.square(tFac.mFactor), axis=0) * tFac.mWeights

    order = np.flip(np.argsort(norm))
    tensor.weights = tensor.weights[order]
    tensor.factors = [fac[:, order] for fac in tensor.factors]
    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor), atol=1e-9)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = tensor.mFactor[:, order]
        tensor.mWeights = tensor.mWeights[order]
        np.testing.assert_allclose(buildMat(tFac), buildMat(tensor), atol=1e-9)

    return tensor


def delete_component(tFac, compNum):
    """ Delete the indicated component. """
    tensor = deepcopy(tFac)
    compNum = np.array(compNum, dtype=int)

    # Assert that component # don't exceed range, and are unique
    assert np.amax(compNum) < tensor.rank
    assert np.unique(compNum).size == compNum.size

    tensor.rank -= compNum.size
    tensor.weights = np.delete(tensor.weights, compNum)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)
        tensor.mWeights = np.delete(tensor.mWeights, compNum)

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor



def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            mScales = np.linalg.norm(tFac.mFactor, ord=np.inf, axis=0)
            tFac.mWeights = scales * mScales
            tFac.mFactor /= mScales

        tFac.factors[i] /= scales

    return tFac


def initialize_cmtf(tensor: np.ndarray, matrix: np.ndarray, rank: int):
    r"""Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    factors = [np.ones((tensor.shape[i], rank)) for i in range(tensor.ndim)]

    # SVD init mode 0
    unfold = tl.unfold(tensor, 0)
    unfold = np.hstack((unfold, matrix))

    if np.sum(~np.isfinite(unfold)) > 0:
        si = IterativeSVD(rank=rank, random_state=1)
        unfold = si.fit_transform(unfold)
        factors[0] = si.U
    else:
        factors[0] = np.linalg.svd(unfold)[0][:, :rank]

    unfold = tl.unfold(tensor, 1)
    unfold = unfold[:, np.all(np.isfinite(unfold), axis=0)]
    factors[1] = np.linalg.svd(unfold)[0]
    factors[1] = factors[1].take(range(rank), axis=1, mode="wrap")
    return tl.cp_tensor.CPTensor((None, factors))



def perform_CMTF(tOrig, mOrig, r=9, tol=1e-6, maxiter=50, progress=True):
    """ Perform CMTF decomposition. """
    assert tOrig.dtype == float
    assert mOrig.dtype == float
    tFac = initialize_cmtf(tOrig, mOrig, r)

    # Pre-unfold
    unfolded = np.hstack((tl.unfold(tOrig, 0), mOrig))
    missingM = np.all(np.isfinite(mOrig), axis=1)
    assert np.sum(missingM) >= 1, "mOrig must contain at least one complete row"
    R2X = -np.inf

    # Precalculate the missingness patterns
    uniqueInfo = np.unique(np.isfinite(unfolded.T), axis=1, return_inverse=True)

    tq = tqdm(range(maxiter), disable=(not progress))
    for _ in tq:
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = mlstsq(kr, tl.unfold(tOrig, m).T).T

        # Solve for the glycan matrix fit
        tFac.mFactor = mlstsq(tFac.factors[0][missingM, :], mOrig[missingM, :]).T

        # Solve for subjects factors
        kr = khatri_rao(tFac.factors, skip_matrix=0)
        kr = np.vstack((kr, tFac.mFactor))
        tFac.factors[0] = mlstsq(kr, unfolded.T, uniqueInfo).T

        R2X_last = R2X
        R2X = calcR2X(tFac, tOrig, mOrig)
        tq.set_postfix(R2X=R2X, delta=R2X - R2X_last, refresh=False)
        assert R2X > 0.0

        if R2X - R2X_last < tol:
            break

    assert not np.all(tFac.mFactor == 0.0)
    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac = sort_factors(tFac)
    tFac.R2X = R2X

    return tFac