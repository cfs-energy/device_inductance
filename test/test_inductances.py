import numpy as np
import shapely

from numpy.typing import NDArray

import device_inductance
from cfsem import (
    self_inductance_lyle6,
    flux_circular_filament,
    mutual_inductance_of_circular_filaments,
    self_inductance_circular_ring_wien,
    gs_operator_order2,
    self_inductance_distributed_axisymmetric_conductor,
)

from pytest import approx

from scipy.constants import mu_0
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized

from . import typical_outputs  # Required fixture

__all__ = ["typical_outputs"]


def test_coil_inductances(typical_outputs: device_inductance.TypicalOutputs):
    """
    Test each coil's self inductances against the rectangular-section calc on a rectangle bounding
    the coil's filaments because we don't have the coil's nominal bounding rectangle stored in the
    ODS, and also to handle the coils that are not really an upright rectangle.

    Test mutual inductances between coils by placing a single filament with their total number of turns
    at the center of each coil, then calculating mutual inductance between each of those single-filament
    approximations. This will produce variable results for different coil shapes; in particular, the
    single-filament representation does not capture the geometry of the central solenoid well.
    """
    device = typical_outputs.device
    coils = device.coils

    # Collapse coils to single filaments
    # and estimate self-inductance as rectangular section
    coils_collapsed = {}
    for coil in coils:
        rs = [f.r for f in coil.filaments]  # [m]
        zs = [f.z for f in coil.filaments]  # [m]

        r = np.mean(rs)  # [m]
        z = np.mean(zs)  # [m]

        w = max(np.ptp(rs), 0.025)  # [m] with a minimum width and height of 2.5cm
        h = max(np.ptp(zs), 0.025)  # [m]

        n = np.sum([f.n for f in coil.filaments])  # total number of turns

        self_inductance_approx = self_inductance_lyle6(r, w, h, n)  # [H]

        coils_collapsed[coil.name] = {
            "r": r,
            "z": z,
            "n": n,
            "w": w,
            "h": h,
            "L_approx": self_inductance_approx,
        }

    # Check self inductances - because the non-rectangular ones have small aspect ratio,
    # this estimate should be quite close
    for coil in coils:
        assert np.allclose(
            [coil.self_inductance], coils_collapsed[coil.name]["L_approx"], rtol=0.2
        )

    # Check mutual inductances
    ncoils = len(coils)
    mcc_approx = np.ones((ncoils, ncoils))
    for i in range(ncoils):
        for j in range(ncoils):
            if i == j:
                mcc_approx[i, j] = coils_collapsed[coils[i].name]["L_approx"]
            else:
                coil1 = coils[i]
                coil2 = coils[j]

                ifil = np.array([1.0])
                rfil = np.array([coils_collapsed[coil1.name]["r"]])
                zfil = np.array([coils_collapsed[coil1.name]["z"]])

                rprime = np.array([coils_collapsed[coil2.name]["r"]])
                zprime = np.array([coils_collapsed[coil2.name]["z"]])

                n1 = coils_collapsed[coil1.name]["n"]
                n2 = coils_collapsed[coil2.name]["n"]
                mcc_approx[i, j] = (
                    n1
                    * n2
                    * flux_circular_filament(ifil, rfil, zfil, rprime, zprime)[0]
                )

    # Zero-out the CS columns and test the well-posed terms alone
    cs_inds = [i for i, coil in enumerate(coils) if "cs" in coil.name.lower()]
    mcc_approx_without_cs = mcc_approx.copy()
    mcc_without_cs = typical_outputs.mcc.copy()
    mcc_without_cs[:, cs_inds] = 0.0
    mcc_without_cs[cs_inds, :] = 0.0
    mcc_approx_without_cs[:, cs_inds] = 0.0
    mcc_approx_without_cs[cs_inds, :] = 0.0

    # This test only works for the coils that can reasonably be approximated
    # as a single filament, so we don't test against the CS here; they're treated the same
    # under the hood, so this still covers all the same functionality, just not every
    # available example
    assert np.allclose(mcc_without_cs, mcc_approx_without_cs, rtol=0.1, atol=1e-6)


def test_structure_self_inductances_against_filamentized(
    typical_outputs: device_inductance.TypicalOutputs,
):
    """
    Test self-inductance approximation for conducting structure against a filamentized self-inductance calc.

    To balance test rigor against time spent running tests, 50 filaments are tested.
    """
    rtol = 0.15
    structures = typical_outputs.device.structures
    nstruct = len(structures)
    nskip = max((nstruct // 50), 1)

    inductance_ratio_err_filamentized = []

    for s in structures[::nskip]:
        poly = s.polygon

        # Extract the boundary of the object
        pathr, pathz = poly.exterior.xy
        padding = 0.01
        extent = [
            np.min(pathr) - padding,
            np.max(pathr) + padding,
            np.min(pathz) - padding,
            np.max(pathz) + padding,
        ]

        # Make a meshgrid spanning the object
        ngrid = 50
        rgrid = np.linspace(extent[0], extent[1], ngrid, endpoint=True)
        zgrid = np.linspace(extent[2], extent[3], ngrid, endpoint=True)
        rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
        dr = rgrid[1] - rgrid[0]
        dz = zgrid[1] - zgrid[0]

        # Build a normalized positive mask of the interior of the object
        # which will represent a unit current distributed more-or-less
        # evenly over the interior
        mask = np.zeros_like(rmesh)
        for ir in range(ngrid):
            for iz in range(ngrid):
                if poly.contains(shapely.Point(rgrid[ir], zgrid[iz])):
                    mask[ir, iz] = 1.0
        interior_inds = np.where(mask > 0.0)

        rs = rmesh[interior_inds]  # [m] filament r coords
        zs = zmesh[interior_inds]  # [m] filament z coords
        ns = np.ones_like(rs) / np.sum(mask)  # number of turns per filament
        a = (
            (dr**2 + dz**2) ** 0.5 / 2 * np.ones_like(rs)
        )  # [m] minor radius of individual filaments

        L_filamentized = _self_inductance_filamentized(rs, zs, ns, a)

        inductance_ratio_err_filamentized.append(s.self_inductance / L_filamentized)
        assert s.self_inductance == approx(L_filamentized, rel=rtol)

    # Keeping these here because we might revisit the error plots
    # import matplotlib.pyplot as plt

    # plt.hist(inductance_ratio_err_filamentized)
    # n = len(inductance_ratio_err_filamentized)
    # plt.title(f"Relative error\nL_rectangular / L_filamentized\nN={n}")
    # plt.show()


def test_structure_self_inductances_against_grad_shafranov(
    typical_outputs: device_inductance.TypicalOutputs,
):
    """
    Test self-inductance approximation for conducting structure against an arbitrary-section
    self-inductance calc usually used for the plasma. In order to use that more detailed calc,
    we need to solve the B-field on the interior of the shape, which requires a G-S flux solve.

    To balance test rigor against time spent running tests, a very small set of filaments are tested.
    """
    rtol = 0.15
    structures = typical_outputs.device.structures
    nstruct = len(structures)
    nskip = max((nstruct // 5), 1)

    inductance_ratio_err_gs = []

    for _i, s in enumerate(structures[::nskip]):
        # print(i, end="\r", flush=True)
        poly = s.polygon

        # Extract the boundary of the object
        pathr, pathz = poly.exterior.xy
        boundary = (
            shapely.LineString(np.array((pathr, pathz)).T).segmentize(1e-3).xy
        )  # Discretize surface to 1mm segments
        boundary = (
            np.array([x for x in boundary[0]]),
            np.array([x for x in boundary[1]]),
        )  # Repack bad array format

        # Make a finer mesh with more padding -
        # While this does use a free boundary condition,
        # the sharp corners of the computational domain are still
        # apparent in the solution for some distance from the boundary,
        # so we need a bit of space between the boundary
        # and the object under study,
        # but need the domain to be small enough that a significant number
        # of cells are inside the object
        padding = 0.1
        extent_gs = [
            np.min(pathr) - padding,
            np.max(pathr) + padding,
            np.min(pathz) - padding,
            np.max(pathz) + padding,
        ]

        # Make a meshgrid spanning the object
        ngrid = 100
        rgrid = np.linspace(extent_gs[0], extent_gs[1], ngrid, endpoint=True)
        zgrid = np.linspace(extent_gs[2], extent_gs[3], ngrid, endpoint=True)
        rmesh, zmesh = np.meshgrid(rgrid, zgrid, indexing="ij")
        dr = rgrid[1] - rgrid[0]
        dz = zgrid[1] - zgrid[0]
        cell_area = dr * dz  # [m^2]

        # Build a normalized positive mask of the interior of the object
        # which will represent a unit current distributed more-or-less
        # evenly over the interior
        mask = np.zeros_like(rmesh)
        for ir in range(ngrid):
            for iz in range(ngrid):
                if poly.contains(shapely.Point(rgrid[ir], zgrid[iz])):
                    mask[ir, iz] = 1.0
        interior_inds = np.where(mask > 0.0)

        # Filament parameters
        ref_current = 1.0  # [A]
        nmask = mask / np.sum(mask)  # Number of turns per filament, as mask
        itor = nmask * ref_current  # [A] toroidal current filaments
        jtor = itor / cell_area  # [A/m^2]

        # Get the Grad-Shafranov operator for this mesh
        vals, rows, cols = gs_operator_order2(rgrid, zgrid)
        operator = csc_matrix(
            (vals, (rows, cols)), shape=(ngrid**2, ngrid**2), copy=True
        )

        # LU factorize the operator
        solver = factorized(operator)

        # Build the G-S system
        rhs = -mu_0 * rmesh * jtor  # [Wb/rad]

        # Set the Grad-Shafranov free boundary condition
        ifil = itor[interior_inds].flatten()  # [A]
        rfil = rmesh[interior_inds].flatten()  # [m]
        zfil = zmesh[interior_inds].flatten()  # [m]
        for ir in range(ngrid):
            for iz in [0, ngrid - 1]:
                rhs[ir, iz] = np.sum(
                    flux_circular_filament(
                        ifil, rfil, zfil, np.array(rgrid[ir]), np.array(zgrid[iz])
                    )
                    / (2.0 * np.pi)  # [Wb/rad])
                )
        for ir in [0, ngrid - 1]:
            for iz in range(ngrid):
                rhs[ir, iz] = np.sum(
                    flux_circular_filament(
                        ifil, rfil, zfil, np.array(rgrid[ir]), np.array(zgrid[iz])
                    )
                    / (2.0 * np.pi)
                )

        # Solve G-S system
        psi_gs = (
            2.0 * np.pi * solver(rhs.flatten()).reshape(rmesh.shape)
        ) / ref_current  # [Wb/A] normalized flux

        # Unlike a plasma, all the filaments carry the same current here
        # so we can approximate the self-inductance by summing over the product of
        # the number of turns and the flux.
        L_gs = np.sum(np.sum(psi_gs * nmask))

        inductance_ratio_err_gs.append(s.self_inductance / L_gs)
        assert s.self_inductance == approx(L_gs, rel=rtol)

        # We can also use the arbitrary-section inductance calc to back out the inductance
        # from the G-S solve results

        # Extract B-fields from solved flux per radian
        br, bz = _calc_B_from_psi(
            psi_gs / (2.0 * np.pi), rmesh, zmesh
        )  # [T/A] normalized B field

        # Estimate self-inductance using distributed calc
        L_distributed, _, _ = self_inductance_distributed_axisymmetric_conductor(
            current=1.0,
            grid=(rgrid, zgrid),
            mesh=(rmesh, zmesh),
            b_part=(br, bz),
            psi_part=psi_gs,
            mask=mask,
            edge_path=boundary,
        )

        assert s.self_inductance == approx(L_distributed, rel=rtol)

    # Keeping these here because we might revisit the error plots
    # import matplotlib.pyplot as plt

    # plt.hist(inductance_ratio_err_gs)
    # n = len(inductance_ratio_err_gs)
    # plt.title(f"Relative error\nL_rectangular / L_gs\nN={n}")
    # plt.show()


def _self_inductance_filamentized(r, z, n, a) -> float:
    """
    Estimate self-inductance of filamentized coil pack
    using an approximation for the self-inductance of filaments as
    a circular-section loop.

    Args:
        r (ndarray): radius, coil center
        z (ndarray): axial position, coil center
        n (ndarray): number of turns
        a (ndarray): effective minor radius of individual filaments
    Returns:
        float: [H], estimated self-inductance
    """

    # Make estimate of same coil's self-inductance based on filamentized model
    nfil = r.size
    fs = np.array((r, z, n)).T
    M = np.zeros((nfil, nfil))
    for i in range(nfil):
        for j in range(nfil):
            if i != j:
                # If this is a mutual inductance between two different filaments, use that calc
                M[i, j] = mutual_inductance_of_circular_filaments(
                    fs[i, :], fs[j, :]
                )  # [H]
            else:
                # Self-inductance of this filament
                major_radius_filament = fs[i, :][0]  # [m] Filament radius
                minor_radius_filament = a[
                    i
                ]  # [m] Heuristic approximation for effective wire radius of filament
                num_turns = fs[i, :][2]  # [] Filament number of turns
                L_f = num_turns**2 * self_inductance_circular_ring_wien(
                    major_radius_filament, minor_radius_filament
                )  # [H]
                M[i, j] = (
                    L_f  # [H] Rough estimate of self-inductance of conductor cross-section assigned to this filament
                )

    # Since all current values are equal across the filaments, effective L is just the sum of all elements of M
    L = np.sum(np.sum(M))  # [H]

    return L  # [H]


def _calc_B_from_psi(psi_per_radian, rmesh, zmesh) -> tuple[NDArray, NDArray]:
    """
    Back-calculate B-field from poloidal flux per radian, per Wesson eqn 3.2.2,
    by 4th-order finite difference on a regular grid.
    """

    dr = rmesh[1][0] - rmesh[0][0]
    dz = zmesh[0][1] - zmesh[0][0]

    # Finite-difference coeffs
    # https:#en.wikipedia.org/wiki/Finite_difference_coefficient
    ddx_central = np.array(
        [
            # 4th-order central difference for first derivative
            (-2, 1 / 12),
            (-1, -2 / 3),
            # (0, 0.0),
            (1, 2 / 3),
            (2, -1 / 12),
        ]
    )

    ddx_fwd = np.array(
        [
            # 4th-order forward difference for first derivative
            (0, -25 / 12),
            (1, 4),
            (2, -3),
            (3, 4 / 3),
            (4, -1 / 4),
        ]
    )
    ddx_bwd = -ddx_fwd  # Reverse & flip signs

    dpsidR = np.zeros_like(psi_per_radian)
    for offs, w in ddx_central:
        start = int(2 + offs)
        end = int(-3 + offs)
        dpsidR[2:-3, :] += (
            w * psi_per_radian[start:end, :] / dr
        )  # Central difference on interior points
    for offs, w in ddx_fwd:
        offs = int(offs)
        dpsidR[0:2, :] += (
            w * psi_per_radian[offs : offs + 2, :] / dr
        )  # One-sided difference on left side
    for offs, w in ddx_bwd:
        start = int(-3 + offs)
        end = int(-2 + offs)
        dpsidR[-3:, :] += w * psi_per_radian[start:end, :] / dr  # right side

    dpsidZ = np.zeros_like(psi_per_radian)
    for offs, w in ddx_central:
        start = int(2 + offs)
        end = int(-3 + offs)
        dpsidZ[:, 2:-3] += w * psi_per_radian[:, start:end] / dz  # Interior points
    for offs, w in ddx_fwd:
        offs = int(offs)
        dpsidZ[:, 0:2] += (
            w * psi_per_radian[:, offs : offs + 2] / dz
        )  # One-sided difference on bottom
    for offs, w in ddx_bwd:
        start = int(-3 + offs)
        end = int(-2 + offs)
        dpsidZ[:, -3:] += w * psi_per_radian[:, start:end] / dz  # top

    r_inv = rmesh**-1

    Br = -r_inv * dpsidZ  # [T]
    Bz = r_inv * dpsidR  # [T]

    return (Br, Bz)
