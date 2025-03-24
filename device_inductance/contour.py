"""
Specialized contour-tracing algo for finding the last closed flux contour
defining the edge of a tokamak plasma.

Among other uses, this contour is needed in order to estimate the plasma's
self-inductance.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from interpn import MulticubicRectilinear, MultilinearRectilinear
from scipy.optimize import minimize

from device_inductance.logging import log


def trace_contour(
    grids: tuple[NDArray, NDArray],
    psi_total: NDArray,
    start: tuple[float, float],
    maxis: tuple[float, float],
    limiter_circumference: float,
    mask_limiter: NDArray,
    ds: float = 1e-2,
    tol: float = 1e-4,
) -> Optional[tuple[NDArray, NDArray]]:
    """
    Specialized contour-tracing algo for finding the last closed flux contour
    defining the edge of a tokamak plasma.
    Among other uses, this contour is needed in order to estimate the plasma's
    self-inductance.

    Attempts to trace a closed contour starting at `start` and returns None
    if the contour crosses the limiter or does not make a full loop, or if
    the resulting contour ends up being excessively long (longer than the limiter).

    Args:
        grids: [m] 1D coordinate grids for computational domain
        psi_total: [T-m^2] 2D grid of total poloidal flux
        start: [m] Initial guess coordinates
        maxis: [m] Magnetic axis coords from O-point search
        limiter_circumference: [m] Limiter length in R-Z plane, to bound expected contour length
        mask_limiter: Float mask of limiter interior (1.0 inside, 0.0 outside)
        ds: [m] Step size. Defaults to 1e-2.
        tol: [T-m^2] Contour flux level tolerance. Defaults to 1e-4.

    Returns:
        Optional[(rpath, zpath)] contour path coordinates if a closed loop was found
    """

    # Some notes about speeding this up later if it becomes an issue
    # because this is not optimized for speed at all yet
    # * Use regular grid interpolation methods
    # * Batch interpolation calls for gradient
    # * Add jac function for rootfinder
    # * Do multilevel fixed-grid line searches from magnetic axis instead of rootfinder
    # * Do a batch check for limiter crossings once every N points
    # * Compile it if it comes to that - everything needed is available in rust, should be easy

    # Unpack
    maxis_3d = np.array([maxis[0], 0.0, maxis[1]])  # (r, phi, z)

    # Initialize limiter mask interpolator
    # This one should be linear because it's representing linear segments
    # and cubic interpolators produce over/undershoot near steps
    mask_interpolator = MultilinearRectilinear.new([x for x in grids], mask_limiter)

    # Initialize flux interpolator
    # In this case, we need a cubic function so that it can
    # properly represent a local minimum that is not exactly on a grid point.
    psi_interpolator = MulticubicRectilinear.new([x for x in grids], psi_total)

    def mask_interp_point(r, z) -> float:
        pt = [np.atleast_1d(r), np.atleast_1d(z)]
        return mask_interpolator.eval(pt)[0]

    def psi_interp_point(r, z) -> float:
        pt = [np.atleast_1d(r), np.atleast_1d(z)]
        return psi_interpolator.eval(pt)[0]

    def grad_psi(r, z, eps=1e-6) -> NDArray:
        dpsi_dr = (psi_interp_point(r + eps, z) - psi_interp_point(r - eps, z)) / (
            2.0 * eps
        )
        dpsi_dz = (psi_interp_point(r, z + eps) - psi_interp_point(r, z - eps)) / (
            2.0 * eps
        )
        return np.array((dpsi_dr, dpsi_dz))

    # Initialize
    #  Total limiter length _should_ bound the length of any reasonable flux contour
    #  since the limiter is guaranteed to be outside & typically has a less-like-a-circle shape
    #  than a typical flux contour. This is a heuristic, but a fairly dependable one
    #  as long as the flux field is sane.
    #  Preallocate output array based on length estimate and chosen step size
    n = int(np.ceil(limiter_circumference / ds)) + 1
    r = np.zeros(n)
    z = np.zeros(n)
    #  Populate starting point
    r[0], z[0] = start
    psi0 = psi_interp_point(r[0], z[0])
    # Initialize accumulated winding angle
    theta = 0.0
    for i in range(n - 2):  # access pattern is i, i+1
        # Local minimization to keep previous point on psi=psi0 contour
        sol = minimize(
            fun=lambda x: (psi_interp_point(x[0], x[1]) - psi0) ** 2,
            x0=[r[i], z[i]],
            options=dict(maxiter=100),  # type: ignore
            tol=tol,
        )
        (r[i], z[i]) = sol.x

        # Check convergence of local minimization solve.
        # If the flux field is well-behaved, it will converge,
        # and we're looking for a contour on a well-behaved region,
        # so failure to converge can be taken to mean that the
        # contour we're looking for isn't here.
        if not sol.success:
            log().error("Contour local solve did not converge")
            return None

        # Take a step along the psi=constant contour
        psi_r, psi_z = grad_psi(r[i], z[i])
        pow = 0.5  # use 0.5 for constant step sizes, less than 0.5 for adaptive steps
        step = ds / (psi_r**2 + psi_z**2) ** pow
        r[i + 1] = r[i] + psi_z * step
        z[i + 1] = z[i] - psi_r * step

        # Measure change in angle of rotation; after 2pi rotation, terminate
        # Expand vectors to (r, phi, z) coords in order to use cross product
        u = np.array([r[i + 1], 0.0, z[i + 1]]) - maxis_3d
        v = np.array([r[i], 0.0, z[i]]) - maxis_3d
        theta += np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))

        # Check if we've made a full revolution.
        # If so, finalize and return results
        if np.abs(theta) > 2.0 * np.pi:
            # Take the populated part
            r = r[: i + 2]
            z = z[: i + 2]
            # Close the contour
            r[-1] = r[0]
            z[-1] = z[0]
            return r, z  # , area

        # Check if the last point was outside the limiter after
        # its adjustment toward the psi=psi0 contour
        mask_val = mask_interp_point(r[i], z[i])
        if mask_val < 0.5:
            log().error("Contour crossed limiter")
            return None

    # We didn't make a full revolution or end up outside the limiter
    # It's possible there was a closed contour to find, but we didn't find it
    log().error("Contour exceeded maximum length without making a full revolution")
    return None
