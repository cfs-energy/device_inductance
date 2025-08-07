"""
Local flux solve in the vicinity of a collection of filaments with polygon sections
that do not necessarily fall on a rectangular grid by allocating the fraction of
intersecting polygon area to each mesh cell.
"""

from dataclasses import dataclass
from itertools import chain

import numpy as np
from interpn import MulticubicRegular
from numpy.typing import NDArray
from pytest import approx
from scipy.optimize import fsolve
from shapely import GeometryCollection, MultiPolygon, Polygon

from device_inductance.logging import log
from device_inductance.utils import calc_flux_density_from_flux, solve_flux_axisymmetric

RMIN = 2e-2
"""[m] minimum r-value to allow in local solver grid"""

N_INSIDE = 8
"""Target number of grid points inside filament extent"""

N_OUTSIDE = 9
"""Target number of grid points on either side of filament extent"""


@dataclass(frozen=True)
class LocalFields:
    """Outputs of local field solve"""

    jtor_per_amp: NDArray
    """[1/m^2] normalized current density map"""

    psi_per_amp: NDArray
    """[Wb/A] normalized poloidal flux"""

    br_per_amp: NDArray
    """[T/A] normalized flux density, r-component"""

    bz_per_amp: NDArray
    """[T/A] normalized flux density, z-component"""

    psi_per_amp_interpolator: MulticubicRegular
    """[Wb/A] normalized poloidal flux, hermite spline interpolator"""

    br_per_amp_interpolator: MulticubicRegular
    """[T/A] normalized flux density, r-component, hermite spline interpolator"""

    bz_per_amp_interpolator: MulticubicRegular
    """[T/A] normalized flux density, z-component, hermite spline interpolator"""

    grids: tuple[NDArray, NDArray]
    """[m] 1D computational grids"""

    meshes: tuple[NDArray, NDArray]
    """[m] 2D computational meshgrids"""


def local_fields(fil_rzn: tuple[NDArray, NDArray, NDArray], polygons: list[Polygon]) -> LocalFields:
    """
    Estimate local self-field of a collection of filaments with polygon representations.

    Args:
        fil_rzn: r-coord, z-coord, and number of turns for each filament
        polygons: polygon section of each filament

    Returns:
        Structure containing local field map components and interpolators
    """
    rs, zs, ns = fil_rzn
    grids_regular = _make_grids_regular(rs, zs)

    # If the winding pack lands on a regular grid, use the corresponding regular grid.
    # Otherwise, fall back on a heuristic grid that spans the winding pack.
    grids = grids_regular or _make_grids_irregular(polygons)

    meshes = _make_mesh(grids)
    dr = grids[0][1] - grids[0][0]
    dz = grids[1][1] - grids[1][0]
    dxgrid = (dr, dz)

    if grids_regular is not None:
        # Just put all the current for each filament on its corresponding mesh cell
        jtor_per_amp = _allocate_current_regular(fil_rzn, grids, meshes)
    else:
        # Split polygons between mesh cells, run flux solve, and back-out B-fields
        jtor_per_amp = _allocate_current_irregular(fil_rzn, polygons, meshes, dxgrid)

    psi_per_amp, br_per_amp, bz_per_amp = _local_fields(jtor_per_amp, grids, meshes)

    # Make interpolators
    dims = [len(grids[0]), len(grids[1])]
    starts = np.array([grids[0][0], grids[1][0]])
    steps = np.array(dxgrid)
    psi_per_amp_interpolator = MulticubicRegular.new(dims, starts, steps, psi_per_amp.flatten())
    br_per_amp_interpolator = MulticubicRegular.new(dims, starts, steps, br_per_amp.flatten())
    bz_per_amp_interpolator = MulticubicRegular.new(dims, starts, steps, bz_per_amp.flatten())

    # Pack numerous outputs into a struct
    result = LocalFields(
        jtor_per_amp,
        psi_per_amp,
        br_per_amp,
        bz_per_amp,
        psi_per_amp_interpolator,
        br_per_amp_interpolator,
        bz_per_amp_interpolator,
        grids,
        meshes,
    )

    return result


def _filament_extent(polygons: list[Polygon]) -> tuple[float, float, float, float]:
    """Get the (rmin, rmax, zmin, zmax) extent that bounds the filament polygons."""
    r = np.array([*chain([p.boundary.xy[0] for p in polygons])])
    z = np.array([*chain([p.boundary.xy[1] for p in polygons])])

    rmin, rmax = np.min(r), np.max(r)
    zmin, zmax = np.min(z), np.max(z)
    extent = (float(rmin), float(rmax), float(zmin), float(zmax))

    return extent


def _make_grids_irregular(polygons: list[Polygon]) -> tuple[NDArray, NDArray]:
    """Make grids that bound the filament extent
    plus at least 7 cells outside to support a 4th order difference method."""
    rmin, rmax, zmin, zmax = _filament_extent(polygons)

    if rmin < 0.0:
        raise ValueError(
            f"Minimum r-coordinate of a filament is <{RMIN} [m],"
            " which does not leave enough room to set boundary conditions."
        )
    if rmin < RMIN:
        log().warning(
            f"Clamping filament polygon bounds to minimum radius of {RMIN} [m], which may distort fields."
        )
        rmin = RMIN

    rspan = rmax - rmin
    zspan = zmax - zmin

    # This is the grid step size if there is no conflict with crossing R=0
    n_inside = max(int(float(len(polygons) ** 0.5)), N_INSIDE)
    dr = rspan / n_inside
    dz = zspan / n_inside

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
            f"Using an excessively large mesh ({nr} X {nz}) "
            "to represent filament local flux solve due to proximity to R=0."
        )

    return rgrid, zgrid


def _make_mesh(grids: tuple[NDArray, NDArray]) -> tuple[NDArray, NDArray]:
    rmesh, zmesh = np.meshgrid(*grids, indexing="ij")
    return rmesh, zmesh


def _allocate_current_irregular(
    fil_rzn: tuple[NDArray, NDArray, NDArray],
    fil_polygons: list[Polygon],
    meshes: tuple[NDArray, NDArray],
    dxgrid: tuple[float, float],
) -> NDArray:
    """Convert a collection of filaments with polygon
    representations to a jtor array mapped on to the meshes."""
    rs, zs, ns = fil_rzn
    total_turns = np.sum(ns)
    dr, dz = dxgrid
    rmesh, zmesh = meshes

    # Make polygons for each mesh cell
    def cell_to_poly(r, z) -> Polygon:
        """Rectangular polygon representing a mesh cell"""
        return Polygon.from_bounds(r - dr / 2, z - dz / 2, r + dr / 2, z + dz / 2)

    mesh_polygons = [cell_to_poly(r, z) for r, z in zip(rmesh.flatten(), zmesh.flatten(), strict=True)]

    # For each filament polygon, find the fraction of its area
    # that falls in each mesh cell polygon.
    # The fraction of the total area assigned to each filament
    # should be already accounted in the `n` for the filament.
    itor_per_amp = np.zeros_like(rmesh.flatten())
    for i in range(len(itor_per_amp)):
        mp = mesh_polygons[i]
        for n, fp in zip(ns, fil_polygons, strict=True):
            # Find overlapping area between this filament and this grid cell.
            # If the filament polygon is not well-behaved, we can end up with more
            # than one distinct overlapping region.
            intersection = fp.intersection(mp)
            if isinstance(intersection, Polygon):
                # Simple intersection
                overlap_area = intersection.area
            elif isinstance(intersection, GeometryCollection | MultiPolygon):
                # Non-simple intersection
                g = intersection.geoms
                overlap_area = sum([x.area for x in g if isinstance(x, Polygon)])
            else:
                raise TypeError(f"Unexpected intersection result: {intersection}")

            # Add contribution to this grid cell from this filament
            itor_per_amp[i] += n * overlap_area / fp.area

    assert np.sum(itor_per_amp) == approx(total_turns, rel=1e-6, abs=1e-6)
    itor_per_amp = itor_per_amp.reshape(rmesh.shape)

    # Shift the centroid of the current map to match the centroid of the filaments
    # Otherwise, there will be a slight discontinuity between the local field map and the
    # far-field values calculated from direct filament functions.
    rcfil = np.sum(rs * ns) / total_turns  # [m]
    zcfil = np.sum(zs * ns) / total_turns
    fil_centroid = np.array((rcfil, zcfil))

    def shift2d(v, shift):
        """Shift a fraction of each cell's current to its neighbor in a 2D array"""
        rshift, zshift = shift
        nr, nz = v.shape

        # Make linear basis functions about the center of the array
        rinds = np.array(range(nr), dtype=float)  # Indices as coordinate
        zinds = np.array(range(nz), dtype=float)
        rmid = (nr - 1) / 2  # Normalization factor to map coordinates to [-1, 1]
        zmid = (nz - 1) / 2
        rcoord = (rinds - rmid) / rmid  # Normalized array coordinates
        zcoord = (zinds - zmid) / zmid
        rbasis = rshift * rcoord  # Shift basis functions
        zbasis = zshift * zcoord

        # Broadcast shift bases
        vshifted = v + v * rbasis.reshape(nr, 1) + v * zbasis.reshape(1, nz)

        # Rescale to preserve total
        vshifted *= np.sum(v) / np.sum(vshifted)

        return vshifted

    def itor_centroid(shift):
        """Find the mesh current centroid given some percent shift in current between neighboring cells"""
        itor_shifted = shift2d(itor_per_amp, shift)
        isum = np.sum(itor_shifted)
        rc = np.sum(itor_shifted * rmesh) / isum
        zc = np.sum(itor_shifted * zmesh) / isum

        return np.array((rc, zc))

    # Solve a shift to match the current centroid of the mesh representation to the
    # current centroid of the filament representation.
    shift = fsolve(lambda shift: itor_centroid(shift) - fil_centroid, x0=np.zeros(2))
    itor_per_amp = shift2d(itor_per_amp, shift)

    # Current density from current
    jtor_per_amp = itor_per_amp / (dr * dz)
    assert np.sum(jtor_per_amp) == approx(total_turns / (dr * dz), rel=1e-6, abs=1e-6)

    return jtor_per_amp


def _allocate_current_regular(
    fil_rzn: tuple[NDArray, NDArray, NDArray],
    grids: tuple[NDArray, NDArray],
    meshes: tuple[NDArray, NDArray],
) -> NDArray:
    """Assign filament current-turn density for filaments that are known to land on grid points"""
    rgrid, zgrid = grids
    dr = rgrid[1] - rgrid[0]  # [m]
    dz = zgrid[1] - zgrid[0]  # [m]
    area = dr * dz  # [m^2]

    # Map current density per amp
    jtor_per_amp = np.zeros_like(meshes[0])  # [A-turns/m^2 / A]
    for r, z, n in zip(*fil_rzn, strict=True):
        # Get indices of location of this filament
        ri = np.argmin(np.abs(rgrid - r))
        zi = np.argmin(np.abs(zgrid - z))
        # Set current density for that unit cell
        # so that the total for the cell comes out to the
        # correct total current
        jtor_per_amp[ri, zi] += n / area

    return jtor_per_amp


def _local_fields(
    jtor: NDArray, grids: tuple[NDArray, NDArray], meshes: tuple[NDArray, NDArray]
) -> tuple[NDArray, NDArray, NDArray]:
    rmesh, zmesh = meshes

    # Solve flux field
    psi = solve_flux_axisymmetric(grids, meshes, jtor)  # [Wb/A]

    # Extract flux density
    br, bz = calc_flux_density_from_flux(psi, rmesh, zmesh)  # [T/A]

    return psi, br, bz


def _make_grids_regular(rs: NDArray, zs: NDArray) -> tuple[NDArray, NDArray] | None:
    """Generate a set of regular r,z grids that span the coil winding pack centers exactly
    if possible, or None if the winding pack can't be represented exactly.

    Adds 4 grid cells of padding around the winding pack to deconflict the
    cells with nonzero current density from the boundary conditions of a
    flux solve.

    If only one unit cell is present on either axis, the grid will be
    expanded 1cm in either direction.
    """

    # Get coordinates with a unit cell
    unique_r = np.array(sorted(list(set(rs))))  # [m]
    unique_z = np.array(sorted(list(set(zs))))

    # Make sure there are enough unit cells to work with,
    # expanding dimensions if necessary
    if len(unique_r) == 1:
        r = unique_r[0]
        unique_r = [r - 1e-2, r, r + 1e-2]
    if len(unique_z) == 1:
        z = unique_z[0]
        unique_z = [z - 1e-2, z, z + 1e-2]
    if len(unique_r) < 2 or len(unique_z) < 2:
        log().error("Failed to expand grid dimensionality. This is a bug.")
        return None

    # Check if the coordinates have regular spacing,
    # which is required to support the finite difference solve
    drs = np.diff(unique_r)
    drmean = np.mean(drs)
    if np.any(np.abs(drs - drmean) / drmean > 1e-4):
        return None
    dzs = np.diff(unique_z)
    dzmean = np.mean(dzs)
    if np.any(np.abs(dzs - dzmean) / dzmean > 1e-4):
        return None

    # Extend grids by a few cells outside the winding pack
    # 7 is the true minimum; 2x 4th-order finite difference patches
    # will be stacked to extract the flux then the flux density, which means
    # 7 cells see direct interaction with boundary conditions and must not
    # have nonzero current density to produce sane results; npad = 6 produces junk outputs.
    npad = 7
    nr = len(unique_r) + 2 * npad
    nz = len(unique_z) + 2 * npad
    r_pad = npad * drmean
    z_pad = npad * dzmean
    rgrid = np.linspace(unique_r[0] - r_pad, unique_r[-1] + r_pad, nr)
    zgrid = np.linspace(unique_z[0] - z_pad, unique_z[-1] + z_pad, nz)

    # Make sure the grid doesn't cross zero.
    # If this check becomes a problem, there is an alternate strategy
    # to double resolution and spread the coil's current density mapping
    # across more than one neighboring cell, but that is too much complexity
    # to implement proactively.
    if rgrid[0] < 0.0:
        return None

    return (rgrid, zgrid)
