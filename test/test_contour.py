import numpy as np
from numpy.typing import NDArray
from pytest import approx
from device_inductance.contour import trace_contour


def test_contour():
    """Make sure the contour can trace a circle"""

    xgrid = np.linspace(1.0, 5.0, 100)
    ygrid = np.linspace(-7.0, 7.0, 100)
    xmesh, ymesh = np.meshgrid(xgrid, ygrid, indexing="ij")
    x0 = 3.0
    z = guess_psi(xmesh, ymesh, x0, z0=0.0)

    mask = np.ones_like(z)

    # Draw a contour that should be a circle around the axis
    radius = 0.5
    start = (x0 + radius, 0.0)
    maxis = (x0, 0.0)
    limiter_circumference = 7.0 + 7.0 + 4.0 + 4.0
    contour = trace_contour(
        (xgrid, ygrid), z, start, maxis, limiter_circumference, mask
    )

    # Make sure the resulting contour is centered around the known center
    assert contour is not None
    assert np.mean(contour[0]) == approx(x0, abs=1e-2)
    assert np.mean(contour[1]) == approx(0.0, abs=1e-2)

    # Make sure the resulting contour is within tolerance of the true circular contour
    dx = contour[0] - x0
    dy = contour[1]
    dist = (dx**2 + dy**2) ** 0.5
    expected_dist = radius * np.ones_like(dist)
    assert np.allclose(dist, expected_dist, atol=2e-3)


def guess_psi(
    rmesh: NDArray, zmesh: NDArray, r0: float, z0: float, elongation: float = 1.0
) -> NDArray:
    """
    Qualitative initial guess for a flux distribution, centered at R0.
    Does not produce a specific minor radius or any other particulars, just a smooth
    falloff from a centroid at (r0, z0). Does not represent any attempt at
    a self-consistent or physically-correct solution.
    """
    rmin = rmesh[0][0]
    rmax = rmesh[-1][0]
    rnorm = (rmesh - r0) / (rmax - rmin)  # not in [0, 1]
    znorm = (zmesh - z0) / (rmax - rmin)  # not in [0, 1], normalized to R span
    psi = np.exp(-((rnorm**2 + (znorm / elongation) ** 2) ** 0.5))
    return psi  # [Wb] qualitative smooth distribution
