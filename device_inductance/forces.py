"""Calculation of forces between current-carrying conductors"""

import numpy as np
from numpy.typing import NDArray

from device_inductance.coils import Coil
from device_inductance.structures import PassiveStructureFilament
from device_inductance.circuits import CoilSeriesCircuit
from device_inductance.utils import _progressbar, calc_flux_density_from_flux
from device_inductance.logging import log

from cfsem import (
    flux_density_circular_filament,
    body_force_density_circular_filament_cartesian,
)

from interpn import MulticubicRectilinear


def _calc_coil_coil_forces(
    coils: list[Coil],
    grids: tuple[NDArray, NDArray],
    coil_flux_density_tables: tuple[NDArray, NDArray],
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    ncoils = len(coils)
    fr = np.zeros((ncoils, ncoils))  # [N/A^2]
    fz = np.zeros((ncoils, ncoils))
    gridlist = [x for x in grids]

    # Calculate force per amp from each coil `i` to each coil `j`
    # using the baked tables, which include the self-field solve patch
    # when it is available (for coils that fall on a regular grid)
    items = (
        _progressbar([x for x in range(ncoils)], "Coil-coil force rows")
        if show_prog
        else range(ncoils)
    )
    for i in items:
        bz = coil_flux_density_tables[1][i, :, :]
        bz_interp = MulticubicRectilinear.new(gridlist, bz)
        for j in range(ncoils):
            if i == j and coils[i].local_fields is None:
                # If we can't make a sane self-field estimate, skip and issue a warning
                log().warning(
                    f"Skipping self-force contribution for coil {coils[i].name} due to lack of smooth local field approximation"
                )
                continue
            elif i == j:
                # If we're doing self-field and we have a local field solve, interpolate on the
                # smooth local field to resolve the singularity issue

                # Integral of I*cross(dL,B)/I with dL in +phi direction = 2*pi*r * nturns * (Bz, 0.0, -Br)
                length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
                obs = [
                    coils[j].rs,
                    coils[j].zs,
                ]  # [m] observation points (filament locations)
                fr[i][j] = np.sum(length_factor * bz_interp.eval(obs))
                fz[i][j] = 0.0  # No self-propulsion; interpolation would produce slightly nonzero value
            else:
                # If these are two separate coils, we can use a full IxB calc
                # which is slower but more accurate than interpolation

                coila = coils[i]
                coilb = coils[j]

                ra, za, na = coila.rs, coila.zs, coila.ns
                rb, zb, nb = coilb.rs, coilb.zs, coilb.ns
                zero = np.zeros_like(rb)
                # Replacing J with I*dL gives body force instead of body force density
                # and we can use the full circular length to scale the I*dL product in the toroidal direction
                fab_jxb_r, fab_jxb_y, fab_jxb_z = (
                    body_force_density_circular_filament_cartesian(
                        na,
                        ra,
                        za,
                        obs=(rb, zero, zb),
                        j=(zero, 2.0 * np.pi * rb * nb, zero),
                    )
                )  # [N/A^2]
                assert sum(fab_jxb_y) == 0.0  # Sanity check
                # Sum contributions at each filament
                fr[i][j] = sum(fab_jxb_r)
                fz[i][j] = sum(fab_jxb_z)

    return (fr, fz)


def _calc_circuit_coil_forces(
    coils: list[Coil],
    circuits: list[CoilSeriesCircuit],
    coil_coil_forces: tuple[NDArray, NDArray],
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    ncirc = len(circuits)
    ncoils = len(coils)
    fr = np.zeros((ncirc, ncoils))  # [N/A^2]
    fz = np.zeros((ncirc, ncoils))

    # Calculate force per amp from each circuit `i` to each coil `j` using coil-coil force tables
    items = (
        _progressbar([x for x in range(ncirc)], "Circuit-coil force rows")
        if show_prog
        else range(ncirc)
    )
    for i in items:
        for j, sign in circuits[i].coils:
            # For each coil in the circuit, add the signed force from that coil
            # on each of the others.
            # If any coils do not have self-force estimates, that will be handled earlier in the coil-coil force tables.
            fr[i, :] += sign * coil_coil_forces[0][j, :]
            fz[i, :] += sign * coil_coil_forces[1][j, :]

    return (fr, fz)


def _calc_structure_coil_forces(
    coils: list[Coil],
    structures: list[PassiveStructureFilament],
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    ncoils = len(coils)
    nstruct = len(structures)

    fr = np.zeros((nstruct, ncoils))  # [N/A^2]
    fz = np.zeros((nstruct, ncoils))

    items = (
        _progressbar([x for x in range(nstruct)], "Structure-coil force rows")
        if show_prog
        else range(nstruct)
    )
    for i in items:
        sr = np.atleast_1d(structures[i].r)
        sz = np.atleast_1d(structures[i].z)
        si = np.ones_like(sr)
        for j in range(ncoils):
            r = coils[j].rs
            z = coils[j].zs
            n = coils[j].ns
            length_factor = 2.0 * np.pi * r * n
            zero = np.zeros_like(r)

            # Replacing J with I*dL gives body force instead of body force density
            # and we can use the full circular length to scale the I*dL product in the toroidal direction
            frij, _, fzij = body_force_density_circular_filament_cartesian(
                si, sr, sz, obs=(r, zero, z), j=(zero, length_factor, zero), par=False
            )
            fr[i, j] = sum(frij)
            fz[i, j] = sum(fzij)

    return (fr, fz)

def _calc_plasma_coil_forces(
    coils: list[Coil],
    grids: tuple[NDArray, NDArray],
    meshes: tuple[NDArray, NDArray],
    plasma_flux_tables_or_limiter_mask: NDArray,
    full_flux_tables: bool = False,
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    ncoil = len(coils)
    nr, nz = (len(grids[0]), len(grids[1]))
    nrnz = nr * nz
    rmesh, zmesh = meshes
    fr = np.zeros((nrnz, ncoil))
    fz = np.zeros((nrnz, ncoil))
    gridlist = [x for x in grids]

    if full_flux_tables:
        # We're using the full tables
        plasma_flux_tables = plasma_flux_tables_or_limiter_mask
        items = (
            _progressbar(
                [x for x in range(nrnz)], "Mesh cell-coil force rows", show_every=nr
            )
            if show_prog
            else range(nrnz)
        )
        for i in items:
            br, bz = calc_flux_density_from_flux(
                plasma_flux_tables[i, :, :], *meshes
            )  # [T/A]
            br_interp = MulticubicRectilinear.new(gridlist, br)  # [T/A] vs. [m]
            bz_interp = MulticubicRectilinear.new(gridlist, bz)

            for j in range(ncoil):
                # Integral of I*cross(dL,B)/I with dL in +phi direction = 2*pi*r * nturns * (Bz, 0.0, -Br)
                length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
                obs = [
                    coils[j].rs,
                    coils[j].zs,
                ]  # [m] observation points (filament locations)
                fr[i][j] = np.sum(length_factor * bz_interp.eval(obs))
                fz[i][j] = np.sum(-length_factor * br_interp.eval(obs))
    else:
        # We're using the limiter mask, so we can calculate contributions directly,
        # visiting only the points on the interior of the limiter and leaving the others as zeroes
        limiter_mask = plasma_flux_tables_or_limiter_mask.flatten()
        items = (
            _progressbar(
                [x for x in range(nrnz)], "Mesh cell-coil force rows", show_every=nr
            )
            if show_prog
            else range(nrnz)
        )
        current = np.atleast_1d(np.ones(1))
        for i in items:
            if limiter_mask[i]:
                for j in range(ncoil):
                    length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
                    rprime, zprime = [
                        coils[j].rs,
                        coils[j].zs,
                    ]  # [m] observation points (filament locations)
                    rfil = np.atleast_1d(rmesh.flatten()[i])
                    zfil = np.atleast_1d(zmesh.flatten()[i])
                    brs, bzs = flux_density_circular_filament(
                        current, rfil, zfil, rprime, zprime, par=False
                    )  # [T/A], not enough points to benefit from parallel impl
                    fr[i][j] = np.sum(length_factor * bzs)
                    fz[i][j] = np.sum(-length_factor * brs)

    return fr, fz
