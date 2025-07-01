import numpy as np

import device_inductance

from pytest import approx

from interpn import MulticubicRectilinear


from . import typical_outputs, typical_outputs_stabilized_eigenmode  # Required fixture

__all__ = ["typical_outputs", "typical_outputs_stabilized_eigenmode"]


def test_coil_coil_forces(typical_outputs: device_inductance.TypicalOutputs):
    """Spot-check an entry in the coil force matrix by capitalizing on the equivalence of
    integrating I*dL and J*dV"""

    device = typical_outputs.device
    coils = device.coils

    coil_flux_density_tables = device.coil_flux_density_tables
    grids = device.grids

    for i in range(len(coils)):
        for j in range(len(coils)):
            if i == j:
                continue

            fab_mat_r, fab_mat_z = (
                device.coil_coil_forces[0][i, j],
                device.coil_coil_forces[1][i, j],
            )  # [N/A^2]

            br = coil_flux_density_tables[0][i, :, :]  # [T/A]
            bz = coil_flux_density_tables[1][i, :, :]
            br_interp = MulticubicRectilinear.new(grids, br)  # [T/A] vs. [m]
            bz_interp = MulticubicRectilinear.new(grids, bz)

            # Integral of I*cross(dL,B)/I with dL in +phi direction = 2*pi*r * nturns * (Bz, 0.0, -Br)
            length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
            obs = [
                coils[j].rs,
                coils[j].zs,
            ]  # [m] observation points (filament locations)
            fr_interped = np.sum(length_factor * bz_interp.eval(obs))
            fz_interped = np.sum(-length_factor * br_interp.eval(obs))

            # We see a bit of interpolation error here due to the coarse grid
            # for coils that border on each other
            assert fab_mat_r == approx(fr_interped, rel=2e-2, abs=3e-6)
            assert fab_mat_z == approx(fz_interped, rel=2e-2, abs=3e-6)


def test_coil_coil_z_force_against_femm(typical_outputs: device_inductance.TypicalOutputs):
    """
    Make sure the magnitude and direction of Z-axis force between two coils
    roughly matches the result from a FEMM axisymmetric finite element model.
    See `femm_coil_test.[FEM,ans]` for FEMM problem setup and results.
    """
    device = typical_outputs.device

    coila_index = device.coil_index_dict["DV1U"]
    coilb_index = device.coil_index_dict["DV2U"]

    coila = device.coils[coila_index]
    coilb = device.coils[coilb_index]

    #
    # Leave this here for extracting the envelope to update the tests
    #

    # def coil_envelope(c: device_inductance.Coil) -> tuple[float, float, float, float]:
    #     """Coil extent only includes the filaments; this expanded extent is the input that would
    #     produce this filamentization"""
    #     rgrid, zgrid = c.grids
    #     dr, dz = np.diff(rgrid)[0], np.diff(zgrid)[0]
    #     rmin, rmax, zmin, zmax = c.extent
    #     return (rmin - dr/2, rmax + dr/2, zmin - dz/2, zmax + dz/2)
    
    # envelope_a = coil_envelope(coila)
    # envelope_b = coil_envelope(coilb)

    # print(envelope_a, envelope_b)

    # raise ValueError

    fzab = device.coil_coil_forces[1][coila_index, coilb_index]  # [N/A^2] * 1A * 1A -> [N]
    # Solved result from a FEMM model that approximately matches the coil envelope with a uniform current density
    fzab_expected = np.sum(coila.ns) * np.sum(coilb.ns) * 1.41356e-5  # [N] at 1A in each coil
    # Expect a bit of error due to FEA discretization, slight differences in envelope geometry,
    # and incomplete joint turns being distributed over the top row of filaments in device_description
    assert fzab == approx(fzab_expected, rel=0.1)


def test_circuit_coil_forces(typical_outputs: device_inductance.TypicalOutputs):
    device = typical_outputs.device
    coils = device.coils
    circuits = device.circuits
    grids = device.grids

    fr, fz = device.circuit_coil_forces
    br, bz = device.circuit_flux_density_tables

    ncoil = len(coils)
    ncirc = len(circuits)

    for i in range(ncirc):
        br_interp = MulticubicRectilinear.new(grids, br[i, :, :])  # [T/A] vs. [m]
        bz_interp = MulticubicRectilinear.new(grids, bz[i, :, :])

        circuit_coil_names = [coils[k].name for k, _ in circuits[i].coils]
        any_coils_no_self_field = any(
            [coils[k].grids is None for k, _ in circuits[i].coils]
        )

        for j in range(ncoil):
            r = coils[j].rs
            z = coils[j].zs
            n = coils[j].ns
            length_factor = 2.0 * np.pi * r * n
            fr_interped = sum(length_factor * bz_interp.eval([r, z]))
            fz_interped = sum(-length_factor * br_interp.eval([r, z]))

            if coils[j].name in circuit_coil_names and any_coils_no_self_field:
                # If this circuit-coil combination includes self-field for a coil that does not
                # have a smooth self-field calc available, then this calc will not match the test method
                pass
            elif coils[j].name in circuit_coil_names:
                # Self-field with smooth field available
                assert fr[i, j] == approx(fr_interped, rel=6e-2, abs=6e-6)

                # In single-coil circuits, only self-field is present, and z-force should be near zero.
                # Because some of the coils' self-field is based on an inexact mapping of turns on to a regular grid,
                # the force may not be _exactly_ zero, but should be small enough not to be important to structural
                # estimates.
                assert fz[i, j] == approx(0.0, abs=5e-3)
            else:
                assert fr[i, j] == approx(fr_interped, rel=6e-2, abs=6e-6)
                assert fz[i, j] == approx(fz_interped, rel=6e-2, abs=6e-6)


def test_structure_coil_forces(typical_outputs: device_inductance.TypicalOutputs):
    device = typical_outputs.device
    coils = device.coils

    fr, fz = device.structure_coil_forces

    # Calculate by alternative method (interpolating on field tables)
    fr_alt, fz_alt = _calc_structure_coil_forces(
        coils, device.grids, device.structure_flux_density_tables
    )

    # The interpolation method is not very good for some coils that are very closely coupled to structures,
    # so this comparison is best done in bulk across the whole population of filaments
    # and with a wide tolerance. Because the error in individual outliers is unbounded and depends primarily
    # on coincidental grid locations, the error check is implemented by limiting the number of total outlier values.
    rtol = 1e-2

    fz_rel_err = (fz - fz_alt) / fz
    fz_n_outliers = len(np.where(np.abs(fz_rel_err) > rtol)[0])

    fr_rel_err = (fr - fr_alt) / fr
    fr_n_outliers = len(np.where(np.abs(fr_rel_err) > rtol)[0])

    n_entries = fr.flatten().size
    assert fr_n_outliers < 0.05 * n_entries
    assert fz_n_outliers < 0.05 * n_entries


def test_structure_mode_coil_forces(typical_outputs: device_inductance.TypicalOutputs):
    device = typical_outputs.device
    
    fr, fz = device.structure_mode_coil_forces
    frc = fr.copy()  # Copy to mutate safely
    fzc = fz.copy()

    # Calculate by alternative method (interpolating on field tables)
    fr_alt, fz_alt = _calc_structure_mode_coil_forces(device.coils, device.grids, device.structure_mode_flux_density_tables, show_prog=False)

    # Remove VS and DV coil rows because their nearby structure is too close-coupled for this test to work well -
    # the interpolated method becomes very sensitive to how close the structures happen to be
    # to the nearest grid cell and produce intermittent test failures despite configuring gmsh to minimize
    # randomization of mesh outputs
    vs_names = [x for x in device.coil_names if "VS" in x or "DV" in x]
    vs_inds = [device.coil_index_dict[x] for x in vs_names]
    frc[:, vs_inds] = 0.0
    fzc[:, vs_inds] = 0.0
    fr_alt[:, vs_inds] = 0.0
    fz_alt[:, vs_inds] = 0.0

    # The interpolation method is not very good for some coils that are very closely coupled to structures,
    # so this comparison is best done in bulk across the whole population of filaments
    # and with a wide tolerance
    assert np.allclose(np.sum(frc, axis=0), np.sum(fr_alt, axis=0), rtol=0.2, atol=3e-6)
    assert np.allclose(np.sum(fzc, axis=0), np.sum(fz_alt, axis=0), rtol=0.2, atol=3e-6)


def test_plasma_coil_force(typical_outputs: device_inductance.TypicalOutputs, typical_outputs_stabilized_eigenmode: device_inductance.TypicalOutputs):
    device_interpolating = typical_outputs.device
    device_masked_direct = typical_outputs_stabilized_eigenmode.device
    limiter_mask = device_masked_direct.limiter_mask
    mask_inds = np.where(limiter_mask.flatten())

    # Make sure that the nonzero entries common between both methods match reasonably well
    fr_interped = device_interpolating.plasma_coil_forces[0][mask_inds, :]
    fz_interped = device_interpolating.plasma_coil_forces[1][mask_inds, :]

    fr_masked = device_masked_direct.plasma_coil_forces[0][mask_inds, :]
    fz_masked = device_masked_direct.plasma_coil_forces[1][mask_inds, :]

    assert np.allclose(fr_interped, fr_masked, rtol=2e-2, atol=1e-6)
    assert np.allclose(fz_interped, fz_masked, rtol=2e-2, atol=1e-6)



def _calc_structure_coil_forces(coils, grids, structure_flux_density_tables):
    ncoils = len(coils)
    nstruct = structure_flux_density_tables[0].shape[0]
    fr = np.zeros((nstruct, ncoils))  # [N/A^2]
    fz = np.zeros((nstruct, ncoils))

    for i in range(nstruct):
        br = structure_flux_density_tables[0][i, :, :]  # [T/A]
        bz = structure_flux_density_tables[1][i, :, :]
        br_interp = MulticubicRectilinear.new(grids, br)  # [T/A] vs. [m]
        bz_interp = MulticubicRectilinear.new(grids, bz)
        for j in range(ncoils):
            # Integral of I*cross(dL,B)/I with dL in +phi direction = 2*pi*r * nturns * (Bz, 0.0, -Br)
            length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
            obs = [
                coils[j].rs,
                coils[j].zs,
            ]  # [m] observation points (filament locations)
            fr[i][j] = np.sum(length_factor * bz_interp.eval(obs))
            fz[i][j] = np.sum(-length_factor * br_interp.eval(obs))

    return (fr, fz)


def _calc_structure_mode_coil_forces(
    coils,
    grids,
    structure_mode_flux_density_tables,
    show_prog: bool = True,
):
    ncoils = len(coils)
    nmodes = structure_mode_flux_density_tables[0].shape[0]
    fr = np.zeros((nmodes, ncoils))  # [N/A^2]
    fz = np.zeros((nmodes, ncoils))

    for i in range(nmodes):
        br = structure_mode_flux_density_tables[0][i, :, :]  # [T/A]
        bz = structure_mode_flux_density_tables[1][i, :, :]
        br_interp = MulticubicRectilinear.new(grids, br)  # [T/A] vs. [m]
        bz_interp = MulticubicRectilinear.new(grids, bz)
        for j in range(ncoils):
            # Integral of I*cross(dL,B)/I with dL in +phi direction = 2*pi*r * nturns * (Bz, 0.0, -Br)
            length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
            obs = [
                coils[j].rs,
                coils[j].zs,
            ]  # [m] observation points (filament locations)
            fr[i][j] = np.sum(length_factor * bz_interp.eval(obs))
            fz[i][j] = np.sum(-length_factor * br_interp.eval(obs))

    return (fr, fz)
