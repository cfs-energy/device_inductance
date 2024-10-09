from itertools import product

import numpy as np
from numpy.typing import NDArray

from device_inductance.coils import Coil
from device_inductance.circuits import CoilSeriesCircuit
from device_inductance.structures import PassiveStructureFilament
from device_inductance.utils import _progressbar

from cfsem import mutual_inductance_of_cylindrical_coils, flux_circular_filament


def _calc_coil_mutual_inductances(coils: list[Coil], show_prog: bool = True) -> NDArray:
    """Populate coil-coil mutual inductance matrix"""

    ncoil = len(coils)
    mcc = np.zeros((ncoil, ncoil))  # [H]
    #   Coil self-inductances
    for i, c in enumerate(coils):
        mcc[i, i] = c.self_inductance
    #   Coil mutual inductances
    items = list(product(enumerate(coils), enumerate(coils)))
    if show_prog:
        items = _progressbar(items, "Coil-coil mutual inductances", show_every=ncoil)
    for (i, c1), (j, c2) in items:
        if i == j:
            # This is a self-inductance term and was already populated
            continue
        if mcc[i, j] != 0.0 and mcc[j, i] != 0.0:
            # We already populated this one from the other direction
            continue

        elems_1 = c1.filaments
        elem_rs_1 = [x.r for x in elems_1]
        elem_zs_1 = [x.z for x in elems_1]
        elem_ns_1 = [x.n for x in elems_1]
        fils1 = np.vstack((elem_rs_1, elem_zs_1, elem_ns_1))

        elems_2 = c2.filaments
        elem_rs_2 = [x.r for x in elems_2]
        elem_zs_2 = [x.z for x in elems_2]
        elem_ns_2 = [x.n for x in elems_2]
        fils2 = np.vstack((elem_rs_2, elem_zs_2, elem_ns_2))

        mcc[i, j] = mutual_inductance_of_cylindrical_coils(fils1, fils2)
        mcc[j, i] = mcc[i, j]  # Mutual inductance is reflexive

    return np.ascontiguousarray(mcc)  # [H]


def _calc_structure_mutual_inductances(
    structures: list[PassiveStructureFilament], show_prog: bool = True
) -> NDArray:
    # Populate passive system mutual inductance matrix
    npassive = len(structures)
    mss = np.zeros((npassive, npassive))
    passive_rs = np.array([x.r for x in structures])
    passive_zs = np.array([x.z for x in structures])

    items = [x for x in enumerate(structures)]
    show_every = max(1, npassive // 100)  # Don't spam too much
    if show_prog:
        items = _progressbar(items, "Structure mutual inductance columns", show_every)
    for i, p1 in items:
        r1 = np.array(p1.r)
        z1 = np.array(p1.z)

        # Calculate mutual inductance contributions to other passive filaments
        # All number of turns = 1, so n1 * n2 = 1 and is dropped here.
        # A unit current of 1A is used for normalization.
        contribs = flux_circular_filament(np.array(1.0), r1, z1, passive_rs, passive_zs)
        # Override the self-inductance component, which is otherwise singular
        contribs[i] = p1.self_inductance
        # Populate column in matrix
        mss[i, :] = contribs

    return np.ascontiguousarray(mss)  # [H]


def _calc_coil_structure_mutual_inductances(
    coils: list[Coil],
    structures: list[PassiveStructureFilament],
    show_prog: bool = True,
) -> NDArray:
    # Populate coil-passive interaction mutual inductance matrix
    ncoil = len(coils)
    npassive = len(structures)

    passive_rs = np.array([x.r for x in structures])
    passive_zs = np.array([x.z for x in structures])
    passive_ns = np.ones(npassive)

    mcs = np.zeros((ncoil, npassive))  # [H]
    items = [x for x in enumerate(coils)]
    if show_prog:
        items = _progressbar(items, "Coil-passive mutual inductances")
    for i, c in items:
        coil_elems = c.filaments
        coil_elem_rs = np.array([x.r for x in coil_elems])
        coil_elem_zs = np.array([x.z for x in coil_elems])
        coil_elem_ns = np.array([x.n for x in coil_elems])

        # Use number of turns as filament current to capture (nturns * [unit current])
        mcs[i, :] = passive_ns * flux_circular_filament(
            coil_elem_ns, coil_elem_rs, coil_elem_zs, passive_rs, passive_zs
        )

    return np.ascontiguousarray(mcs)  # [H]


def _calc_circuit_mutual_inductances(
    circuits: list[CoilSeriesCircuit],
    coil_mutual_inductances: NDArray,
    show_prog: bool = True,
) -> NDArray:
    # Populate circuit-circuit mutual inductance matrix from coils
    ncirc = len(circuits)

    # Wiring direction affects mutual inductances but not self-inductances,
    # so it's convenient to work with a modified version of the coil-coil matrix
    mcc_signed = coil_mutual_inductances.copy()
    for circ in circuits:
        for i, sign in circ.coils:
            mcc_signed[i, :] *= sign
            mcc_signed[:, i] *= sign
            # Self inductance is still positive even if sign is negative
            # Any two coils wired in reverse will have their mutual inductance
            # sign cancel out and become positive again

    m = np.zeros((ncirc, ncirc))  # [H]
    items = [x for x in enumerate(circuits)]
    if show_prog:
        items = _progressbar(items, "Circuit-circuit mutual inductances")
    for i, circi in items:
        icoilinds = [c[0] for c in circi.coils]
        for j, circj in enumerate(circuits):
            # This procedure works for both self- and mutual- terms
            jcoilinds = [c[0] for c in circj.coils]
            m[i, j] = np.sum(mcc_signed[icoilinds, jcoilinds])
            m[j, i] = m[i, j]

    return m  # [H]


def _calc_circuit_structure_mutual_inductances(
    circuits: list[CoilSeriesCircuit],
    coil_structure_mutual_inductances: NDArray,
    show_prog: bool = True,
) -> NDArray:
    ncirc = len(circuits)
    nstruct = coil_structure_mutual_inductances.shape[1]

    m = np.zeros((ncirc, nstruct))  # [H]
    items = [x for x in enumerate(circuits)]
    if show_prog:
        items = _progressbar(items, "Circuit-structure mutual inductances")
    for i, circ in items:
        for j, sign in circ.coils:
            m[i, :] += sign * coil_structure_mutual_inductances[j, :]

    return m  # [H]
