import numpy as np
from pytest import raises
from device_inductance.utils import _progressbar, gradient_order4


def test_progressbar():
    """Just make sure it runs and returns the input values"""
    vals = list(range(2, 7))
    bar = _progressbar(vals)
    for i, v in enumerate(bar):
        assert vals[i] == v


def test_gradient_order4():
    """Make sure order 2 and order 4 gradients converge to the same value"""

    def f(x, y):
        # Some bivariate function without important features under the grid size
        return np.sin(x / 6.0) + np.cos(y / 6.0)

    # Coarse grid
    xgrid1 = np.linspace(-5.0, 5.0, 10)
    ygrid1 = np.linspace(-7.0, 7.0, 10)
    xmesh1, ymesh1 = np.meshgrid(xgrid1, ygrid1, indexing="ij")
    z1 = f(xmesh1, ymesh1)
    grad1_order2 = np.gradient(z1, xgrid1, ygrid1)
    grad1_order4 = gradient_order4(z1, xmesh1, ymesh1)
    assert np.allclose(grad1_order2, grad1_order4, rtol=0.1)  # Related, but not great match
    with raises(AssertionError):
        # Order 2 method should not be very good at this resolution
        assert np.allclose(grad1_order2, grad1_order4, rtol=0.01)

    # Fine grid
    xgrid2 = np.linspace(-5.0, 5.0, 100)
    ygrid2 = np.linspace(-7.0, 7.0, 100)
    xmesh2, ymesh2 = np.meshgrid(xgrid2, ygrid2, indexing="ij")
    z2 = f(xmesh2, ymesh2)
    grad2_order2 = np.gradient(z2, xgrid2, ygrid2)
    grad2_order4 = gradient_order4(z2, xmesh2, ymesh2)
    assert np.allclose(grad2_order2, grad2_order4, rtol=0.01)  # Reasonably close
    with raises(AssertionError):
        # There's still significant benefit to using the order 4 method here
        assert np.allclose(grad1_order2, grad1_order4, rtol=0.005)
