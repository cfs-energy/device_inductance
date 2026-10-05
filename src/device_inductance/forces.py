"""Calculation of forces between current-carrying conductors"""

import numpy as np
from cfsem import (
    body_force_density_circular_filament_cartesian,
    flux_density_circular_filament,
)
from interpn import MulticubicRectilinear
from numpy.typing import NDArray

from device_inductance.circuits import CoilSeriesCircuit
from device_inductance.coils import Coil
from device_inductance.logging import log
from device_inductance.structures import PassiveStructureLoop
from device_inductance.utils import _progressbar, calc_flux_density_from_flux


def _calc_coil_coil_forces(
    coils: list[Coil],
    grids: tuple[NDArray, NDArray],
    coil_flux_density_tables: tuple[NDArray, NDArray],
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    """Calculate radial and axial force coefficients between coils.

    Mutual forces use ideal circular filaments centered at (0, 0, z), with
    radii given by the source coil and normals along +z. Signed, fractional
    turn counts set the filament currents per terminal ampere. Source centers
    and normals follow the source filament count, which can differ from the
    number of target filaments where the force is evaluated.

    Self-force uses the smooth local-field table when available; otherwise the
    diagonal entry stays zero and a warning is logged. Axial self-force is zero.

    Args:
        coils: Source and target coils, in force-matrix order.
        grids: Radial and axial grid coordinates [m] for the field tables.
        coil_flux_density_tables: Radial and axial fields [T/A], each with shape
            (ncoils, nr, nz). The axial table supplies the radial self-force.
        show_prog: Whether to display progress.

    Returns:
        Radial and axial arrays [N/A^2], each with shape (ncoils, ncoils).
        Entry [i, j] gives force on coil j from coil i per product of their
        terminal currents; multiplying by I_i * I_j gives the force in newtons.
    """
    ncoils = len(coils)
    fr = np.zeros((ncoils, ncoils))  # [N/A^2]
    fz = np.zeros((ncoils, ncoils))
    gridlist = [x for x in grids]

    # Calculate force per amp from each coil `i` to each coil `j`
    # using the baked tables, which include the self-field solve patch
    # when it is available (for coils that fall on a regular grid)
    items = _progressbar([x for x in range(ncoils)], "Coil-coil force rows") if show_prog else range(ncoils)
    for i in items:
        bz = coil_flux_density_tables[1][i, :, :]
        bz_interp = MulticubicRectilinear.new(gridlist, bz)
        for j in range(ncoils):
            if i == j and coils[i].local_fields is None:
                # If we can't make a sane self-field estimate, skip and issue a warning
                log().warning(
                    f"Skipping self-force contribution for coil {coils[i].name} "
                    "due to lack of smooth local field approximation"
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
                source_zero = np.zeros_like(ra)
                # Replacing J with I*dL gives body force instead of body force density
                # and we can use the full circular length to scale the I*dL product in the toroidal direction
                fab_jxb_r, fab_jxb_y, fab_jxb_z = body_force_density_circular_filament_cartesian(
                    ifil=na,
                    rfil=ra,
                    loc=(source_zero, source_zero, za),
                    normal=(source_zero, source_zero, np.ones_like(ra)),
                    wire_radius=None,
                    obs=(rb, zero, zb),
                    j=(zero, 2.0 * np.pi * rb * nb, zero),
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
    items = _progressbar([x for x in range(ncirc)], "Circuit-coil force rows") if show_prog else range(ncirc)
    for i in items:
        for j, sign in circuits[i].coils:
            # For each coil in the circuit, add the signed force from that coil
            # on each of the others.
            # If any coils do not have self-force estimates,
            # that will be handled earlier in the coil-coil force tables.
            fr[i, :] += sign * coil_coil_forces[0][j, :]
            fz[i, :] += sign * coil_coil_forces[1][j, :]

    return (fr, fz)


def _calc_structure_coil_forces(
    coils: list[Coil],
    structures: list[PassiveStructureLoop],
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    """Calculate radial and axial force coefficients on coils from structures.

    Each structure supplies ideal circular source filaments centered at
    (0, 0, z), with normals along +z. Its fractional turn weights distribute a
    unit loop current among those filaments. Force is evaluated at the target
    coil filaments and weighted by their signed turn counts and circumferences,
    so the source and target can have different filament counts.

    Args:
        coils: Target coils, in force-matrix column order.
        structures: Source structure loops, in force-matrix row order.
        show_prog: Whether to display progress.

    Returns:
        Radial and axial arrays [N/A^2], each with shape (nstruct, ncoils).
        Entry [i, j] gives force on coil j from structure i per product of the
        structure-loop current and coil terminal current.
    """
    ncoils = len(coils)
    nstruct = len(structures)

    fr = np.zeros((nstruct, ncoils))  # [N/A^2]
    fz = np.zeros((nstruct, ncoils))
    items = _progressbar(structures, "Structure-coil force rows") if show_prog else structures
    for i, s in enumerate(items):
        ifil = s.ns  # [dimensionless] unit reference current for normalization times number of turns
        rfil = s.rs  # [m]
        zfil = s.zs  # [m]
        for j in range(ncoils):
            r = coils[j].rs  # [m]
            z = coils[j].zs  # [m]
            n = coils[j].ns  # [dimensionless]
            length_factor = 2.0 * np.pi * r * n
            zero = np.zeros_like(r)
            source_zero = np.zeros_like(rfil)

            # Replacing J with I*dL gives body force instead of body force density
            # and we can use the full circular length to scale the I*dL product in the toroidal direction
            frij, _, fzij = body_force_density_circular_filament_cartesian(
                ifil=ifil,
                rfil=rfil,
                loc=(source_zero, source_zero, zfil),
                normal=(source_zero, source_zero, np.ones_like(rfil)),
                wire_radius=None,
                obs=(r, zero, z),
                j=(zero, length_factor, zero),
                par=False,
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
            _progressbar([x for x in range(nrnz)], "Mesh cell-coil force rows", show_every=nr)
            if show_prog
            else range(nrnz)
        )
        for i in items:
            br, bz = calc_flux_density_from_flux(plasma_flux_tables[i, :, :], *meshes)  # [T/A]
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
            _progressbar([x for x in range(nrnz)], "Mesh cell-coil force rows", show_every=nr)
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
