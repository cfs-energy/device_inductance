"""Calculation of forces between current-carrying conductors"""

import numpy as np
from numpy.typing import NDArray

from device_inductance import Coil, PassiveStructureFilament
from device_inductance.device import F64
from device_inductance.circuits import CoilSeriesCircuit
from device_inductance.utils import _progressbar
from device_inductance.logging import log

from interpn import MulticubicRectilinear

def _calc_coil_coil_forces(coils: list[Coil], grids: tuple[NDArray[F64], NDArray[F64]], coil_flux_density_tables: tuple[NDArray[F64], NDArray[F64]], show_prog: bool = True) -> tuple[NDArray[F64], NDArray[F64]]:
    ncoils = len(coils)
    fr = np.zeros((ncoils, ncoils))  # [N/A^2]
    fz = np.zeros((ncoils, ncoils))

    # Calculate force per amp from each coil `i` to each coil `j`
    # using the baked tables, which include the self-field solve patch
    # when it is available (for coils that fall on a regular grid)
    items = _progressbar([x for x in range(ncoils)], "Coil-coil force rows") if show_prog else range(ncoils)
    for i in items:
        br = coil_flux_density_tables[0][i, :, :]  # [T/A]
        bz = coil_flux_density_tables[1][i, :, :]
        br_interp = MulticubicRectilinear.new(grids, br)  # [T/A] vs. [m]
        bz_interp = MulticubicRectilinear.new(grids, bz)
        for j in range(ncoils):
            if i == j and coils[i].local_fields is None:
                log().warning(f"Skipping self-force contribution for coil {coils[i].name} due to lack of smooth local field approximation")
                continue
            length_factor = 2.0 * np.pi * coils[j].rs * coils[j].ns  # [m]-turns
            obs = [coils[j].rs, coils[j].zs]  # [m] observation points (filament locations)
            fr[i][j] = np.sum(length_factor * bz_interp.eval(obs))
            fz[i][j] = np.sum(-length_factor * br_interp.eval(obs))
    
    return (fr, fz)

def _calc_circuit_coil_forces(coils: list[Coil], circuits: list[CoilSeriesCircuit], coil_coil_forces: tuple[NDArray[F64], NDArray[F64]], show_prog: bool = True) -> tuple[NDArray[F64], NDArray[F64]]:
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
            # If any coils do not have self-force estimates, that will be handled earlier in the coil-coil force tables.
            fr[i, :] += sign * coil_coil_forces[0][j, :]
            fz[i, :] += sign * coil_coil_forces[1][j, :]

    return (fr, fz)
