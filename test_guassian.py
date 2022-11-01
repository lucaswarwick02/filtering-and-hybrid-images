import pytest
import math
import numpy as np

from MyHybridImages import makeGaussianKernel

@pytest.mark.parametrize( "sigma,expected_size",
    [ (0, 1), (1.4, 13), (1.5, 13), (3, 25), (12, 97) ]
)
def test_kernel_size (sigma, expected_size):
    # Create Guassian Kernel
    guassian_kernel = makeGaussianKernel(sigma)

    # Compare shape of Guassian Kernel to predicted shape
    assert len(guassian_kernel) == expected_size

@pytest.mark.parametrize( "sigma",
    [ (0), (1.5), (3), (12) ]
)
def test_test_sum (sigma):
    # Create Guassian Kernel
    guassian_kernel = makeGaussianKernel(sigma)

    # Required as summtion causes floating point errors. We don't want to round Guassian values, so we round the sum to 5dp 
    assert round(np.sum(guassian_kernel), 5) == 1.0

