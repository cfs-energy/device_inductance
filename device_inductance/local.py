"""
Local flux solve in the vicinity of a collection of filaments with both point and polygon representations
that do not necessarily fall on a rectangular grid.
"""

from itertools import chain

import numpy as np
from numpy.typing import NDArray

from shapely import Polygon

from .logging import log
from .utils import solve_flux_axisymmetric, calc_flux_density_from_flux


RMIN = 2e-2
"""[m] minimum r-value to allow in local solver grid"""

N_INSIDE = 12
"""Target number of grid points inside filament extent"""

N_OUTSIDE = 6
"""Target number of grid points on either side of filament extent"""


def _filament_extent(fil_polygons: list[Polygon]) -> tuple[float, float, float, float]:
    """Get the (rmin, rmax, zmin, zmax) extent that bounds the filament polygons."""
    r = np.array(chain([p.boundary.xy[0] for p in fil_polygons]))
    z = np.array(chain([p.boundary.xy[1] for p in fil_polygons]))

    rmin, rmax = np.min(r), np.max(r)
    zmin, zmax = np.min(z), np.max(z)
    return (*[float(x) for x in (rmin, rmax, zmin, zmax)],)


def _make_grids(fil_polygons: list[Polygon]) -> tuple[NDArray, NDArray]:
    """Make grids that bound the filament extent plus at least 6 cells outside to support a 4th order difference method."""
    rmin, rmax, zmin, zmax = _filament_extent(fil_polygons)

    if rmin < 0.0:
        raise ValueError(
            f"Minimum r-coordinate of a filament is <{RMIN} [m], which does not leave enough room to set boundary conditions."
        )
    if rmin < RMIN:
        log().warning(
            f"Clamping filament polygon bounds to minimum radius of {RMIN} [m], which may distort fields."
        )
        rmin = RMIN

    rspan = rmax - rmin
    zspan = zmax - zmin

    # This is the grid step size if there is no conflict with crossing R=0
    dr = rspan / N_INSIDE
    dz = zspan / N_INSIDE

    # Take the smaller step size of the two for both;
    # the method can handle different step sizes on each axis, but best
    # results are obtained with grid cells of equal aspect ratio.
    dr = min(dr, dz)
    dz = dr

    # Check if we are interfering with R=0
    # and decrease size of both r-step and z-step until that is resolved
    # to keep the innermost point off of R=0.
    while rmin - dr * N_OUTSIDE < RMIN / 2.0:
        dr /= 2.0
    dz = dr

    # Make the actual grids
    rgrid = np.arange(rmin - dr * N_OUTSIDE, rmax + dr * N_OUTSIDE + dr, dr)
    zgrid = np.arange(zmin - dz * N_OUTSIDE, zmax + dz * N_OUTSIDE + dz, dz)

    nr, nz = len(rgrid), len(zgrid)
    if nz > 100 or nz > 100:
        log().warning(
            f"Using an excessively large mesh ({nr} X {nz}) to represent filament local flux solve due to proximity to R=0."
        )

    return rgrid, zgrid


def _make_mesh(grids: tuple[NDArray, NDArray]) -> tuple[NDArray, NDArray]:
    return np.meshgrid(*grids, indexing="ij")


def _allocate_current(
    fil_rzn: tuple[NDArray, NDArray, NDArray],
    fil_polygons: list[Polygon],
    meshes: tuple[NDArray, NDArray],
    dxgrid: tuple[float, float],
) -> NDArray:
    """Convert a collection of filaments with polygon representations to a jtor array mapped on to the meshes."""
    _, _, ns = fil_rzn
    dr, dz = dxgrid
    rmesh, zmesh = meshes

    def cell_to_poly(r, z) -> Polygon:
        """Rectangular polygon representing a mesh cell"""
        return Polygon.from_bounds(r - dr / 2, r + dr / 2, z - dz / 2, z + dz / 2)

    mesh_polygons = [
        cell_to_poly(r, z) for r, z in zip(rmesh.flatten(), zmesh.flatten())
    ]

    # For each polygon, find the fraction of its area
    # that falls in each mesh cell.
    # The fraction of the total area assigned to each filament should be already accounted in the `n` for the filament.
    itor = np.zeros_like(rmesh.flatten())
    for i in range(len(itor)):
        mp = mesh_polygons[i]
        for n, fp in zip(ns, fil_polygons):
            intersection = fp.intersection(mp)
            if isinstance(intersection, Polygon):
                itor[i] += n * intersection.area / fp.area

    # Normalize to unit total current
    itor /= np.sum(itor)

    # Current density from current
    jtor = itor / (dr * dz)

    return jtor.reshape(rmesh.shape)


def _local_fields(
    jtor: NDArray, grids: tuple[NDArray, NDArray], meshes: tuple[NDArray, NDArray]
) -> tuple[NDArray, NDArray, NDArray]:
    rmesh, zmesh = meshes

    # Solve flux field
    psi = solve_flux_axisymmetric(grids, meshes, jtor)  # [Wb/A]

    # Extract flux density
    br, bz = calc_flux_density_from_flux(psi, rmesh, zmesh)  # [T/A]

    return psi, br, bz


def local_fields(
    fil_rzn: tuple[NDArray, NDArray, NDArray], fil_polygons: list[Polygon]
) -> tuple[NDArray, NDArray, NDArray, tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """
    Estimate local self-field of a collection of filaments with polygon representations.

    """
    grids = _make_grids(fil_polygons)
    meshes = _make_mesh(grids)
    dr = grids[0][1] - grids[0][0]
    dz = grids[1][1] - grids[1][0]
    dxgrid = (dr, dz)
    jtor = _allocate_current(fil_rzn, fil_polygons, meshes, dxgrid)

    psi, br, bz = _local_fields(jtor, grids, meshes)

    return psi, br, bz, grids, meshes
