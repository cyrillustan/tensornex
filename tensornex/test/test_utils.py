import pytest
import numpy as np
from tensornex.utils import calcR2X, fac2tensor
from numpy.testing import assert_almost_equal

@pytest.mark.parametrize("tensor_shape", [(20, 8, 7), (20, 9, 8, 7), (20, 9, 8, 7, 6)])
def test_fac2tensor(tensor_shape):
    n_comp = 4
    factors = [np.random.rand(ii, n_comp) for ii in tensor_shape]
    recon = fac2tensor(factors)

    for _ in range(10):
        random_pos = tuple(np.random.randint(i) for i in tensor_shape)
        assert_almost_equal(recon[*random_pos],
                            np.sum([np.prod([factors[ii][random_pos[ii], rr]
                                             for ii in range(len(tensor_shape))])
                                    for rr in range(n_comp)])
                            )
