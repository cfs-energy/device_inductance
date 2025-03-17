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
        any_coils_no_self_field = any([coils[k].grids is None for k, _ in circuits[i].coils])

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
            else:
                # print(i, j, circuits[i].name, coils[j].name, f"{fr[i,j]:e},{fr_interped:e}", f"{fz[i,j]:e},{fz_interped:e}")
                assert fr[i, j] == approx(fr_interped, rel=6e-2, abs=6e-6)
                assert fz[i, j] == approx(fz_interped, rel=6e-2, abs=6e-6)