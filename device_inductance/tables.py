"""Tabulation of flux & related values on a provided computational grid"""

from itertools import product

import numpy as np
from numpy.typing import NDArray

from device_inductance.coils import Coil
from device_inductance.circuits import CoilSeriesCircuit
from device_inductance.structures import PassiveStructureFilament
from device_inductance.utils import (
    _progressbar,
    calc_flux_density_from_flux,
    _rect_mask,
)

from cfsem import (
    flux_circular_filament,
    flux_density_circular_filament,
    self_inductance_lyle6,
)

_MIN_DIST = 0.15  # [m]
"""
Under this distance from a filament, use a finite-difference calc on the
flux field to get the B-field instead of using a filament calc for B, because
the flux calcs are numerically better-conditioned.
"""


def _table_mask(
    meshes: tuple[NDArray, NDArray], extent: tuple[float, float, float, float]
) -> tuple[NDArray, tuple[NDArray[np.intp], ...]]:
    """Get a mask with padding of max(2*dx, _MIN_DIST) on each axis"""
    rmesh, zmesh = meshes
    dr = rmesh[1, 0] - rmesh[0, 0]  # [m]
    dz = zmesh[0, 1] - zmesh[0, 0]  # [m]
    rdelta = max(2 * dr, _MIN_DIST)  # [m]
    zdelta = max(2 * dz, _MIN_DIST)  # [m]
    return _rect_mask(meshes, extent, pad=(rdelta, zdelta))


def _calc_coil_flux_tables(
    coils: list[Coil], meshes: tuple[NDArray, NDArray], show_prog: bool = True
) -> NDArray:
    # Unpack
    ncoil = len(coils)
    rmesh, zmesh = meshes  # [m]
    shape = meshes[0].shape
    nr, nz = meshes[0].shape

    # Calculate tables using filament calcs
    coil_table_shape = (ncoil, nr, nz)  # This ordering makes each table contiguous
    psi_mesh_coils = np.zeros(coil_table_shape)  # [Wb/A]
    items = [x for x in enumerate(coils)]
    if show_prog:
        items = _progressbar(items, "Coil flux tables")
    for i, c in items:
        # Add contribution from each coil to its place in the table
        ifil = np.array([e.n for e in c.filaments])  # Effective current is nturns * 1A
        rfil = np.array([e.r for e in c.filaments])  # [m]
        zfil = np.array([e.z for e in c.filaments])  # [m]
        psi_mesh_coils[i, :, :] = flux_circular_filament(
            ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten()
        ).reshape(shape)  # [Wb/A]

        # For coils with their winding pack on a regular grid, patch over the field near the coil
        # with a local axisymmetric flux solve
        cgrids = c.grids
        cmeshes = c.meshes
        cinterp = c.local_field_interpolators
        if cgrids is not None and cmeshes is not None and cinterp is not None:
            # Figure out which points to replace with the local field
            cpsi, _, _ = cinterp
            extent = c.extent
            _, inds = _table_mask(meshes, extent)

            # Replace points with local field solve
            robs = rmesh[inds].flatten()
            zobs = zmesh[inds].flatten()
            psi_mesh_coils[i, :, :][inds] = cpsi.eval([robs, zobs])

    return np.ascontiguousarray(psi_mesh_coils)  # [Wb/A]


def _calc_coil_flux_density_tables(
    coils: list[Coil],
    meshes: tuple[NDArray, NDArray],
    coil_flux_tables: NDArray,
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    # Unpack
    ncoil = len(coils)
    rmesh, zmesh = meshes  # [m]
    shape = meshes[0].shape
    nr, nz = meshes[0].shape

    # Calculate
    coil_table_shape = (ncoil, nr, nz)  # This ordering makes each table contiguous
    br_mesh_coils = np.zeros(coil_table_shape)  # [T/A]
    bz_mesh_coils = np.zeros(coil_table_shape)  # [T/A]
    items = [x for x in enumerate(coils)]
    if show_prog:
        items = _progressbar(items, "Coil flux density (B-field) tables")
    for i, c in items:
        # Add contribution from each coil to its place in the table
        ifil = np.array([e.n for e in c.filaments])  # Effective current is nturns * 1A
        rfil = np.array([e.r for e in c.filaments])  # [m]
        zfil = np.array([e.z for e in c.filaments])  # [m]
        b = flux_density_circular_filament(
            ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten()
        )  # [T/A]
        br_mesh_coils[i, :, :] = b[0].reshape(shape)  # [T/A]
        bz_mesh_coils[i, :, :] = b[1].reshape(shape)  # [T/A]

        # Because the flux density filament calc has an additional 1/r^2 factor compared to
        # the flux calc, it is not as well-conditioned numerically and will tend to give
        # worse results for observation points very close to the coil filaments.
        # In order to remedy this, we can patch over the region immediately near the coil
        # winding pack using a B-field calc that comes from the flux calc, giving better
        # floating-point error at the expense of some discretization error.
        #    Figure out what part we're replacing
        _, inds = _table_mask(meshes, c.extent)
        #    Do the replacement
        br_from_psi, bz_from_psi = calc_flux_density_from_flux(
            coil_flux_tables[i, :, :], rmesh, zmesh
        )  # [T/A]
        br_mesh_coils[i][inds] = br_from_psi[inds]  # [T/A]
        bz_mesh_coils[i][inds] = bz_from_psi[inds]  # [T/A]

    br_mesh_coils = np.ascontiguousarray(br_mesh_coils)  # [T/A]
    bz_mesh_coils = np.ascontiguousarray(bz_mesh_coils)  # [T/A]

    return (br_mesh_coils, bz_mesh_coils)  # [T/A]


def _calc_structure_flux_tables(
    structures: list[PassiveStructureFilament],
    meshes: tuple[NDArray, NDArray],
    show_prog: bool = True,
) -> NDArray:
    # Unpack
    npassive = len(structures)
    rmesh, zmesh = meshes  # [m]
    shape = meshes[0].shape
    nr, nz = meshes[0].shape

    # Calculate
    structure_table_shape = (npassive, nr, nz)
    psi_mesh_structures = np.zeros(structure_table_shape)  # [Wb/A]
    items = [x for x in enumerate(structures)]
    show_every = max(1, npassive // 100)  # Don't spam too much
    if show_prog:
        items = _progressbar(items, "Structure flux tables", show_every)
    for i, e in items:
        # Add contribution from each structure filament to its place in the table
        # Number of turns for passive filaments is always 1, so it is dropped here.
        ifil = np.array([1.0])  # [A] unit reference current for normalization
        rfil = np.array([e.r])  # [m]
        zfil = np.array([e.z])  # [m]
        psi_mesh_structures[i, :, :] = flux_circular_filament(
            ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten()
        ).reshape(shape)  # [Wb/A]

    return np.ascontiguousarray(psi_mesh_structures)  # [Wb/A]


def _calc_structure_flux_density_tables(
    structures: list[PassiveStructureFilament],
    meshes: tuple[NDArray, NDArray],
    structure_flux_tables: NDArray,
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    # Unpack
    npassive = len(structures)
    rmesh, zmesh = meshes  # [m]
    shape = meshes[0].shape
    nr, nz = meshes[0].shape

    # Calculate
    structure_table_shape = (npassive, nr, nz)
    br_mesh_structures = np.zeros(structure_table_shape)  # [T/A]
    bz_mesh_structures = np.zeros(structure_table_shape)  # [T/A]
    items = [x for x in enumerate(structures)]
    show_every = max(1, npassive // 100)  # Don't spam too much
    if show_prog:
        items = _progressbar(
            items, "Structure flux density (B-field) tables", show_every
        )
    for i, e in items:
        # Add contribution from each structure filament to its place in the table
        # Number of turns for passive filaments is always 1, so it is dropped here.
        ifil = np.array([1.0])  # [A] unit reference current for normalization
        rfil = np.array([e.r])  # [m]
        zfil = np.array([e.z])  # [m]
        b = flux_density_circular_filament(
            ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten()
        )  # [T/A]
        br_mesh_structures[i, :, :] = b[0].reshape(shape)  # [T/A]
        bz_mesh_structures[i, :, :] = b[1].reshape(shape)  # [T/A]

        # Similar to the coils, we'll get better results very close to the
        # filaments using a finite difference on the flux values.
        # Another option would be to make a loop of linear segments and do biot-savart
        #    Figure out what part we're replacing
        dist = ((rmesh - rfil) ** 2 + (zmesh - zfil) ** 2) ** 0.5  # [m]
        inds = np.where(dist < _MIN_DIST)
        #    Do the replacement
        br_from_psi, bz_from_psi = calc_flux_density_from_flux(
            structure_flux_tables[i, :, :], rmesh, zmesh
        )  # [T/A]
        br_mesh_structures[i, :, :][inds] = br_from_psi[inds]  # [T/A]
        bz_mesh_structures[i, :, :][inds] = bz_from_psi[inds]  # [T/A]

    br_mesh_structures = np.ascontiguousarray(br_mesh_structures)  # [T/A]
    bz_mesh_structures = np.ascontiguousarray(bz_mesh_structures)  # [T/A]

    return br_mesh_structures, bz_mesh_structures  # [T/A]


def _calc_mesh_flux_tables(
    meshes: tuple[NDArray, NDArray],
    grids: tuple[NDArray, NDArray],
    show_prog: bool = True,
) -> NDArray:
    # Unpack
    rgrid, zgrid = grids  # [m]
    rmesh, zmesh = meshes
    shape = meshes[0].shape
    nr, nz = shape
    dr = rgrid[1] - rgrid[0]
    dz = zgrid[1] - zgrid[0]
    assert np.allclose(
        np.diff(rgrid), dr, atol=1e-6
    ), "Self-inductance calc requires uniform grid"
    assert np.allclose(
        np.diff(zgrid), dz, atol=1e-6
    ), "Self-inductance calc requires uniform grid"

    # Calculate
    mesh_table_shape = (nr * nz, nr, nz)
    psi_mesh_mesh = np.zeros(mesh_table_shape)  # [Wb/A]
    items = [x for x in enumerate(product(range(nr), range(nz)))]
    show_every = max(1, (nr * nz) // 100)  # Don't spam too much
    if show_prog:
        items = _progressbar(items, "Mesh flux tables", show_every)
    ifil = np.array([1.0])  # [A] unit reference current for normalization
    rfil = np.array([np.nan])
    zfil = np.array([np.nan])
    for i, (ir, iz) in items:
        # For each grid location, calculate its contribution to all grid locations
        # including its own, using the axisymmetric filament calc for the non-self terms
        # and falling back on the rectangular-section calc for the self-term to resolve
        # the singularity.
        ifil = np.array([1.0])
        rfil = rgrid[ir]
        zfil = zgrid[iz]
        psi_mesh_mesh[i, :, :] = flux_circular_filament(
            ifil, rfil, zfil, rmesh.flatten(), zmesh.flatten()
        ).reshape(shape)
        # Replace singular self-term with 6th order rectangular-section calc
        psi_mesh_mesh[i, ir, iz] = self_inductance_lyle6(
            float(rgrid[ir]), float(dr), float(dz), n=1.0
        )

    return np.ascontiguousarray(psi_mesh_mesh)  # [Wb/A]


def _calc_structure_mode_flux_tables(
    psi_mesh_structures: NDArray,
    tuv: NDArray,
    show_prog: bool = True,
) -> NDArray:
    # Unpack
    neig = tuv.shape[1]
    npassive = psi_mesh_structures.shape[0]
    shape = psi_mesh_structures[0, :, :].shape
    nr, nz = shape

    # Calculate
    #    Eigenmode maps are a linear combination of
    #    the flux maps of the structure filaments
    eig_table_shape = (neig, nr, nz)
    psi_mesh_eig = np.zeros(eig_table_shape)  # [Wb/A]
    items = [x for x in product(range(neig), range(npassive))]
    if show_prog:
        items = _progressbar(items, "Structure mode flux tables")
    for i, j in items:
        psi_mesh_eig[i, :, :] += tuv[j, i] * psi_mesh_structures[j]

    return np.ascontiguousarray(psi_mesh_eig)


def _calc_structure_mode_flux_density_tables(
    br_mesh_structures: NDArray,
    bz_mesh_structures: NDArray,
    tuv: NDArray,
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    # Unpack
    neig = tuv.shape[1]
    npassive = br_mesh_structures.shape[0]
    shape = br_mesh_structures[0, :, :].shape
    nr, nz = shape

    # Calculate
    #    Eigenmode maps are a linear combination of
    #    the maps of the structure filaments
    eig_table_shape = (neig, nr, nz)
    br_mesh_eig = np.zeros(eig_table_shape)  # [T/A]
    bz_mesh_eig = np.zeros(eig_table_shape)  # [T/A]
    items = [x for x in product(range(neig), range(npassive))]
    if show_prog:
        items = _progressbar(items, "Structure mode flux density (B-field) tables")
    for i, j in items:
        br_mesh_eig[i, :, :] += tuv[j, i] * br_mesh_structures[j]
        bz_mesh_eig[i, :, :] += tuv[j, i] * bz_mesh_structures[j]

    br_mesh_eig = np.ascontiguousarray(br_mesh_eig)
    bz_mesh_eig = np.ascontiguousarray(bz_mesh_eig)

    return br_mesh_eig, bz_mesh_eig  # [T/A]


def _calc_circuit_flux_tables(
    circuits: list[CoilSeriesCircuit], coil_flux_tables: NDArray, show_prog: bool = True
) -> NDArray:
    # Unpack
    ncirc = len(circuits)
    _, nr, nz = coil_flux_tables.shape

    # Sum coil contributions to each circuit response,
    # applying the series or anti-series orientation of each coil
    psi_circ = np.zeros((ncirc, nr, nz))  # [Wb/A]
    items = [x for x in enumerate(circuits)]
    if show_prog:
        items = _progressbar(items, "Circuit flux tables")
    for i, circuit in items:
        for coil_index, sign in circuit.coils:
            psi_circ[i, :, :] += sign * coil_flux_tables[coil_index, :, :]

    return psi_circ  # [Wb/A]


def _calc_circuit_flux_density_tables(
    circuits: list[CoilSeriesCircuit],
    br_mesh_coils: NDArray,
    bz_mesh_coils: NDArray,
    show_prog: bool = True,
) -> tuple[NDArray, NDArray]:
    # Unpack
    ncirc = len(circuits)
    _, nr, nz = br_mesh_coils.shape

    # Sum coil contributions to each circuit response,
    # applying the series or anti-series orientation of each coil
    br_circ = np.zeros((ncirc, nr, nz))  # [T/A]
    bz_circ = np.zeros((ncirc, nr, nz))  # [T/A]
    items = [x for x in enumerate(circuits)]
    if show_prog:
        items = _progressbar(items, "Circuit flux density (B-field) tables")
    for i, circuit in items:
        for coil_index, sign in circuit.coils:
            br_circ[i, :, :] += sign * br_mesh_coils[coil_index, :, :]
            bz_circ[i, :, :] += sign * bz_mesh_coils[coil_index, :, :]

    return br_circ, bz_circ  # [T/A]
