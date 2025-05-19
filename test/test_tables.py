import numpy as np

import matplotlib.pyplot as plt

import device_inductance
from device_inductance.utils import calc_flux_density_from_flux
from device_inductance.contour import trace_contour

from cfsem import self_inductance_circular_ring_wien

from pytest import approx

from interpn import MulticubicRegular

from . import typical_outputs  # Required fixture

__all__ = ["typical_outputs"]


def test_coil_tables(typical_outputs: device_inductance.TypicalOutputs):
    """
    Test flux tables by using them to back-calculate off-diagonal
    mutual inductance terms.

    Test B-field tables against finite difference calc.
    """
    rtol = 2e-3  # Flux error is mostly from grid resolution
    atol_b = 3e-8  # [T/A-turn] small values see some numerical error
    psi = typical_outputs.psi_c  # [Wb/A]
    br, bz = typical_outputs.device.coil_flux_density_tables  # [T/A]
    mcc = typical_outputs.mcc  # [H]

    coils = typical_outputs.device.coils
    ncoils = len(coils)

    # Interpolator inputs
    dims = np.array(psi[0, :, :].shape)
    starts = np.array([x[0] for x in typical_outputs.grids])
    steps = np.array(typical_outputs.dxgrid)

    # Test flux tables against mutual inductance matrix
    for i in range(ncoils):
        psi_interp = MulticubicRegular.new(dims, starts, steps, psi[i, :, :])
        for j in range(ncoils):
            if j == i:
                continue  # Skip self-inductance terms
            # Repack target filament locations
            robs = np.array([x.r for x in coils[j].filaments])  # [m]
            zobs = np.array([x.z for x in coils[j].filaments])  # [m]
            nobs = np.array([x.n for x in coils[j].filaments])  # target filament turns

            mutual_inductance_interped = np.sum(nobs * psi_interp.eval([robs, zobs]))
            assert mutual_inductance_interped == approx(mcc[i, j], rel=rtol)

        # Calculate B-field by alternative method
        br_from_psi, bz_from_psi = calc_flux_density_from_flux(
            psi[i, :, :], *typical_outputs.meshes
        )

        # Check B-field inside the limiter, where the results affect shaping,
        # and where the results are not affected by filaments that come too close
        # to grid points
        mask = typical_outputs.device.limiter_mask
        assert np.allclose(br[i, :, :] * mask, br_from_psi * mask, rtol, atol_b)
        assert np.allclose(bz[i, :, :] * mask, bz_from_psi * mask, rtol, atol_b)

        limiter_path = typical_outputs.device.limiter.boundary.xy

        _fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
        axes = np.array(axes).flatten()
        plt.sca(axes[0])
        plt.plot(*limiter_path, color="k")
        plt.imshow(
            ((br[i, :, :] - br_from_psi) / br[i, :, :]).T,
            origin="lower",
            extent=typical_outputs.extent,
        )
        plt.colorbar()
        plt.title("br rel err")

        plt.sca(axes[1])
        plt.plot(*limiter_path, color="k")
        plt.imshow(
            (br[i, :, :] - br_from_psi).T, origin="lower", extent=typical_outputs.extent
        )
        plt.colorbar()
        plt.title("br err")

        plt.sca(axes[2])
        plt.plot(*limiter_path, color="k")
        plt.imshow(br_from_psi.T, origin="lower", extent=typical_outputs.extent)
        plt.colorbar()
        plt.title("br from psi")

        plt.sca(axes[3])
        plt.plot(*limiter_path, color="k")
        plt.imshow(br[i, :, :].T, origin="lower", extent=typical_outputs.extent)
        plt.colorbar()
        plt.title("br from filaments")

        plt.sca(axes[4])
        plt.plot(*limiter_path, color="k")
        plt.imshow(
            ((bz[i, :, :] - bz_from_psi) / bz[i, :, :]).T,
            origin="lower",
            extent=typical_outputs.extent,
        )
        plt.colorbar()
        plt.title("bz rel err")

        plt.sca(axes[5])
        plt.plot(*limiter_path, color="k")
        plt.imshow(
            (bz[i, :, :] - bz_from_psi).T, origin="lower", extent=typical_outputs.extent
        )
        plt.colorbar()
        plt.title("bz err")

        plt.sca(axes[6])
        plt.plot(*limiter_path, color="k")
        plt.imshow(bz_from_psi.T, origin="lower", extent=typical_outputs.extent)
        plt.colorbar()
        plt.title("bz from psi")

        plt.sca(axes[7])
        plt.plot(*limiter_path, color="k")
        plt.imshow(bz[i, :, :].T, origin="lower", extent=typical_outputs.extent)
        plt.colorbar()
        plt.title("bz from filaments")

        plt.suptitle(coils[i].name)
        plt.tight_layout()

    plt.show()


def test_structure_tables(typical_outputs: device_inductance.TypicalOutputs):
    """
    Test flux tables by using them to back-calculate off-diagonal
    mutual inductance terms.
    """
    rtol = 1e-2  # Error is mostly from grid resolution
    psi_s = typical_outputs.psi_s  # [Wb/A]
    br, bz = typical_outputs.device.structure_flux_density_tables  # [T/A]
    mss = typical_outputs.mss  # [H]

    structures = typical_outputs.device.structures
    nstructs = len(structures)

    # interpolator inputs
    dims = [x for x in psi_s[0, :, :].shape]
    starts = np.array([x[0] for x in typical_outputs.grids])
    steps = np.array(typical_outputs.dxgrid)

    for i in range(nstructs):
        psi_interp = MulticubicRegular.new(dims, starts, steps, psi_s[i, :, :])
        # r = structures[i].rs
        # z = structures[i].zs
        for j in range(nstructs):
            if j == i:
                continue  # Skip self-inductance terms

            robs = structures[j].rs
            zobs = structures[j].zs

            mutual_inductance_interped = np.sum(structures[j].ns * psi_interp.eval([robs, zobs]))
            assert mutual_inductance_interped == approx(mss[i, j], rel=rtol)

        # For the structure's effect on B-field, we mostly care about the interior
        # of the limiter, where it affects plasma shaping, but we can test it everywhere
        meshes = typical_outputs.meshes
        br_from_psi, bz_from_psi = calc_flux_density_from_flux(psi_s[i, :, :], *meshes)

        assert np.allclose(br[i, :, :], br_from_psi, rtol=1e-3, atol=3e-8)
        assert np.allclose(bz[i, :, :], bz_from_psi, rtol=1e-3, atol=3e-8)


def test_plasma_tables_solver_inductance(
    typical_outputs: device_inductance.TypicalOutputs,
):
    """
    Compare plasma flux tables to results of a grad-shafranov solve, and compare
    plasma self-inductance to the circular ring formula.
    """

    rtol = 2e-3

    # Use the test mesh for the G-S solve for simplicity
    rmesh, zmesh = typical_outputs.meshes
    dr, dz = typical_outputs.dxgrid

    # Make a mockup of a current density profile
    ref_current = 1e3 * np.sqrt(
        np.e
    )  # [A] some number not an integer or multiple of pi
    minor_radius = 0.4  # [m] some number inside limiter
    r0 = 1.8
    z0 = 0.0
    dl = ((rmesh - r0) ** 2 + (zmesh - z0) ** 2) ** 0.5
    jtor_smooth = 1.0 / (1.0 + dl**2)
    jtor = jtor_smooth.copy()
    jtor[np.where(dl > minor_radius)] = 0.0
    jtor *= ref_current / (np.sum(jtor) * dr * dz)

    # Get plasma flux field by summing tables and by doing a grad-shafranov solve
    psi_tabulated = typical_outputs.device.calc_plasma_flux(
        current_density=jtor, calc_method="table"
    )
    psi_gs = typical_outputs.device.calc_plasma_flux(
        current_density=jtor, calc_method="solve"
    )

    # Compare tabulated flux summation to G-S solve
    assert np.allclose(psi_tabulated, psi_gs, rtol=rtol)

    # Plasma tables are already made by this method so it's a little redundant,
    # but does help to make sure the flux gradient is not too different between
    # the two methods and that the plasma flux calc passthrough runs at all
    br_plasma_1, bz_plasma_1 = typical_outputs.device.calc_plasma_flux_density(
        psi_tabulated
    )
    br_plasma_2, bz_plasma_2 = typical_outputs.device.calc_plasma_flux_density(psi_gs)
    assert np.allclose(br_plasma_1, br_plasma_2, rtol=rtol, atol=1e-8 * ref_current)
    assert np.allclose(bz_plasma_1, bz_plasma_2, rtol=rtol, atol=1e-8 * ref_current)

    # Check self-inductance against circular-section self-inductance calc
    #     Trace boundary
    rstart = np.max(rmesh[np.where(jtor > 0.0)])
    contour = trace_contour(
        typical_outputs.grids,
        jtor_smooth,  # Use jtor because we never formulated a consistent total flux field
        (rstart, z0),
        (r0, z0),
        30.0,
        typical_outputs.device.limiter_mask,
        ds=1e-3,
        tol=1e-6,
    )
    assert contour is not None
    #    Calculate self-inductance
    plasma_self_inductance, _, _ = typical_outputs.device.calc_plasma_self_inductance(
        ref_current,
        psi_gs,
        br_plasma_2,
        bz_plasma_2,
        contour,
        np.where(jtor > 0.0, 1.0, 0.0),
    )  # [H]
    approx_plasma_self_inductance = self_inductance_circular_ring_wien(
        r0, minor_radius
    )  # [H]

    #    The result won't match exactly because this is not a uniform current density distribution,
    #    but it should be fairly close
    assert plasma_self_inductance == approx(approx_plasma_self_inductance, rel=0.15)

    # Plot flux fields and error
    _fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    plt.sca(axes[0])
    plt.imshow(jtor.T, extent=typical_outputs.extent_for_plotting, origin="lower")
    plt.plot(*contour, color="b")
    plt.title("jtor [A/m^2]")

    plt.sca(axes[1])
    plt.imshow(psi_gs.T, extent=typical_outputs.extent_for_plotting, origin="lower")
    plt.plot(*contour, color="b")
    plt.title("Solved flux from G-S [Wb]")

    plt.sca(axes[2])
    plt.imshow(
        psi_tabulated.T, extent=typical_outputs.extent_for_plotting, origin="lower"
    )
    plt.plot(*contour, color="b")
    plt.title("Summed flux from tables [Wb]")

    plt.sca(axes[3])
    plt.imshow(
        (psi_tabulated.T - psi_gs.T) / psi_gs.T,
        extent=typical_outputs.extent_for_plotting,
        origin="lower",
    )
    plt.colorbar()
    plt.plot(*contour, color="b")
    plt.title("Error []")

    plt.tight_layout()

    plt.show()
