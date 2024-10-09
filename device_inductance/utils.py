from datetime import datetime
from typing import TypeVar, Iterator

import numpy as np
from numpy.typing import NDArray


T = TypeVar("T")

# Finite-difference coeffs
# https://en.wikipedia.org/wiki/Finite_difference_coefficient
_DDX_CENTRAL_ORDER4 = np.array(
    [
        # 4th-order central difference for first derivative
        (-2, 1 / 12),
        (-1, -2 / 3),
        # (0, 0.0),
        (1, 2 / 3),
        (2, -1 / 12),
    ]
)

_DDX_FWD_ORDER4 = np.array(
    [
        # 4th-order forward difference for first derivative
        (0, -25 / 12),
        (1, 4),
        (2, -3),
        (3, 4 / 3),
        (4, -1 / 4),
    ]
)

_DDX_BWD_ORDER4 = -_DDX_FWD_ORDER4  # Reverse & flip signs


def _progressbar(it: list[T], suffix="", show_every: int = 1) -> Iterator[T]:
    """A simple terminal progressbar."""
    size = 30
    count = len(it)

    start = datetime.now()

    def show(j):
        x = int(size * j / count)
        print(
            "[{}{}] {}/{} {}, Elapsed: {:.3f} [s]".format(
                "#" * x,
                "." * (size - x),
                j,
                count,
                suffix,
                (datetime.now() - start).total_seconds(),
            ),
            end="\r",
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        if (i % show_every == 0) or (i == count - 1):
            show(i + 1)
    print("\n", flush=True)


def gradient_order4(
    z: NDArray, xmesh: NDArray, ymesh: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Calculate gradient by 4th-order finite difference.

    `numpy.gradient` exists and is fast and convenient, but only uses a second-order difference,
    which produces unacceptable error in B-fields (well over 1% for typical geometries).

    ## Errors

        * If the input grids are not regular
        * If any input grid dimensions have size less than 5

    ## References

        * [1] “Finite difference coefficient,” Wikipedia. Aug. 22, 2023.
              Accessed: Mar. 29, 2024. [Online].
              Available: https://en.wikipedia.org/w/index.php?title=Finite_difference_coefficient

    Args:
        z: [<xunits>] 2D array of values on which to calculate the gradient
        xmesh: [m] 2D array of coordinates of first dimension
        ymesh: [m] 2D array of coordinates of second dimension

    Returns:
        (dzdx, dzdy) [<xunits>/m] 2D arrays of gradient components
    """
    nx, ny = z.shape
    dx = xmesh[1][0] - xmesh[0][0]
    dy = ymesh[0][1] - ymesh[0][0]

    # Check regular grid assumption
    assert np.all(
        np.abs(np.diff(xmesh[:, 0]) - dx) / dx < 1e-6
    ), "This method is only implemented for a regular grid"
    assert np.all(
        np.abs(np.diff(ymesh[0, :]) - dy) / dy < 1e-6
    ), "This method is only implemented for a regular grid"

    dzdx = np.zeros_like(z)
    for offs, w in _DDX_CENTRAL_ORDER4:
        start = int(2 + offs)
        end = int(nx - 2 + offs)
        dzdx[2:-2, :] += (
            w * z[start:end, :] / dx
        )  # Central difference on interior points
    for offs, w in _DDX_FWD_ORDER4:
        offs = int(offs)
        dzdx[0:2, :] += (
            w * z[offs : offs + 2, :] / dx
        )  # One-sided difference on left side
    for offs, w in _DDX_BWD_ORDER4:
        start = int(-2 + offs)
        end = int(nx + offs)
        dzdx[-2:, :] += w * z[start:end, :] / dx  # right side

    dzdy = np.zeros_like(z)
    for offs, w in _DDX_CENTRAL_ORDER4:
        start = int(2 + offs)
        end = int(ny - 2 + offs)
        dzdy[:, 2:-2] += w * z[:, start:end] / dy  # Interior points
    for offs, w in _DDX_FWD_ORDER4:
        offs = int(offs)
        dzdy[:, 0:2] += w * z[:, offs : offs + 2] / dy  # One-sided difference on bottom
    for offs, w in _DDX_BWD_ORDER4:
        start = int(-2 + offs)
        end = int(ny + offs)
        dzdy[:, -2:] += w * z[:, start:end] / dy  # top

    return dzdx, dzdy


def calc_flux_density_from_flux(
    psi: NDArray, rmesh: NDArray, zmesh: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Back-calculate B-field from poloidal flux per Wesson eqn 3.2.2 by 4th-order finite difference,
    modified to use total poloidal flux instead of flux per radian.

    This avoids an expensive sum over filamentized contributions at the expense of some numerical error.

    # Errors

        * If the input grids are not regular
        * If any input grid dimensions have size less than 5

    # References

        * [1] J. Wesson, Tokamaks. Oxford, New York: Clarendon Press, 1987.

    Args:
        psi: [Wb] poloidal flux
        rmesh: [m] 2D r-coordinates
        zmesh: [m] 2D z-coordinates

    Returns:
        (br, bz) [T] 2D arrays of poloidal flux density
    """

    dpsidr, dpsidz = gradient_order4(psi, rmesh, zmesh)

    r_inv = rmesh**-1

    br = -r_inv * dpsidz / (2.0 * np.pi)  # [T]
    bz = r_inv * dpsidr / (2.0 * np.pi)  # [T]

    return (br, bz)