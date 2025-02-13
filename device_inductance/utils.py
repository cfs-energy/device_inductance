from datetime import datetime
from typing import TypeVar, Iterator, Optional, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.constants import mu_0

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

from cfsem import gs_operator_order4, flux_circular_filament


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


def flux_solver(grids: tuple[NDArray, NDArray]) -> Callable[[NDArray], NDArray]:
    """
    Linear solver for extracting a flux field from a toroidal current density distribution
    using a 4th-order finite difference approximation of the Grad-Shafranov PDE.
    For `jtor` toroidal current density shaped like (nr, nz), call like `psi = flux_solver(rhs)`
    to get `psi` in [Wb] or [V-s], where `rhs = -2.0 * np.pi * mu_0 * rmesh * jtor` with the boundary
    values set to the circular-filament solved flux.

    Args:
        grids: [m] regular 1D r,z grids

    Returns:
        solver: factorized solver for Grad-Shafranov differential operator
    """
    # Build Grad-Shafranov Delta* linear operator for finite difference
    # as a sparse matrix
    _ = _check_regular(grids)
    rgrid, zgrid = grids
    nr = rgrid.size
    nz = zgrid.size
    vals, rows, cols = gs_operator_order4(*grids)
    operator = csc_matrix((vals, (rows, cols)), shape=(nr * nz, nr * nz))
    # Store LU factorization of operator matrix to allow fast, reusable
    # solves using different right-hand-side (different current density)
    return factorized(operator)


def solve_flux_axisymmetric(
    grids: tuple[NDArray, NDArray],
    meshes: tuple[NDArray, NDArray],
    current_density: NDArray,
    solver: Optional[Callable[[NDArray], NDArray]] = None,
) -> NDArray:
    """
    Calculate the flux field associated with a given toroidal current density distribution,
    by solving the Grad-Shafranov PDE.

    This calculation is most commonly used for the plasma, but is in fact more general,
    and applies to anything with an equivalent toroidal current density and axisymmetry.

    Args:
        grids: [m] 1D r,z regular coordinate grids
        meshes: [m] 2D meshgrids made from grids like np.meshgrid(*grids, indexing="ij")
        current_density: [A/m^2], shape (nr, nz), toroidal current density on finite-difference mesh
        solver: Optionally, provide a pre-initialized linear solver. See `cfsem.utils.flux_solver`.

    Returns:
        poloidal flux field, [Wb] with shape (nr, nz)
    """
    # Build the differential operator, if needed
    solver = solver or flux_solver(grids)

    # Unpack and filter down to just useful inputs
    dr, dz = _check_regular(grids)  # [m] grid spacing
    area = dr * dz  # [m^2]
    rmesh, zmesh = meshes  # [m]
    nonzero_inds = np.where(current_density != 0.0)
    current_density_nonzero = np.ascontiguousarray(
        current_density[nonzero_inds]
    )  # [A/m^2]
    rmesh_nonzero = np.ascontiguousarray(rmesh[nonzero_inds])  # [m]
    zmesh_nonzero = np.ascontiguousarray(zmesh[nonzero_inds])  # [m]
    # Solve `Delta* @ psi = -mu_0 * 2pi * rmesh * jtor`
    #   Set up right-hand-side of Grad-Shafranov
    rhs = -(2.0 * np.pi * mu_0) * rmesh * current_density  # [Wb/m^2]
    #   Set flux boundary condition
    #   For most relevant grid sizes (up to 500 X 500), doing the O(N^3)
    #   circular-filament flux calc is faster than the linear solve
    #   and therefore faster than doing an extra fixed-boundary linear solve
    #   in order to use Von Hagenow's asymptotically-O(N^2logN) method.
    ifil = (area * current_density_nonzero).flatten()  # [A] plasma filament current
    rfil = rmesh_nonzero.flatten()
    zfil = zmesh_nonzero.flatten()
    for s in [[0, ...], [-1, ...], [..., 0], [..., -1]]:  # All boundary slices
        rhs[s[0], s[1]] = flux_circular_filament(
            ifil, rfil, zfil, rmesh[s[0], s[1]], zmesh[s[0], s[1]]
        )
    #   Do the actual linear solve
    psi = solver(rhs.flatten()).reshape(rmesh.shape)  # [Wb]

    return psi


def _check_regular(grids: tuple[NDArray, NDArray], tol=1e-6) -> tuple[float, float]:
    """Check that grids are regular and returns spacing"""
    rgrid, zgrid = grids
    drs = np.diff(rgrid)
    dzs = np.diff(zgrid)
    drmean = float(np.mean(drs))
    dzmean = float(np.mean(dzs))
    assert np.all(np.abs(drs - drmean) / drmean < 1e-4), "Grids must be regular"
    assert np.all(np.abs(dzs - dzmean) / dzmean < 1e-4), "Grids must be regular"

    return drmean, dzmean  # [m]


def _rect_mask(
    meshes: tuple[NDArray, NDArray],
    extent: tuple[float, float, float, float],
    pad: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray, tuple[NDArray[np.intp], ...]]:
    """
    Get a boolean mask and interior indices of the rectangular region
    of an R-Z mesh spanned by an extent with padding.

    Args:
        meshes: [m] 2D r,z meshgrids
        extent: [m] rmin, rmax, zmin, zmax extent of region to mask
        pad: [m] r,z padding to add on either side of the extent. Defaults to (0.0, 0.0).

    Returns:
        2D mask, interior indices
    """
    rmin, rmax, zmin, zmax = extent  # all [m]
    rpad, zpad = pad
    rmesh, zmesh = meshes

    # Apply padding
    rmin, rmax, zmin, zmax = rmin - rpad, rmax + rpad, zmin - zpad, zmax + zpad

    # Build mask
    mask = np.ones_like(rmesh)
    mask *= np.where(rmesh >= rmin, True, False)
    mask *= np.where(rmesh <= rmax, True, False)
    mask *= np.where(zmesh >= zmin, True, False)
    mask *= np.where(zmesh <= zmax, True, False)
    inds = np.where(mask > 0.0)

    return mask, inds


def _pad_extent(
    extent: tuple[float, float, float, float], pad: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Add r,z padding to both sides of an extent"""
    rmin, rmax, zmin, zmax = extent
    rpad, zpad = pad
    return rmin - rpad, rmax + rpad, zmin - zpad, zmax + zpad


def _join_extents(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """The union of two rmin, rmax, zmin, zmax extents"""
    rmina, rmaxa, zmina, zmaxa = a
    rminb, rmaxb, zminb, zmaxb = b
    return min(rmina, rminb), max(rmaxa, rmaxb), min(zmina, zminb), max(zmaxa, zmaxb)
