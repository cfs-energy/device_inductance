from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from cfsem import MU_0
from numpy.typing import NDArray

from device_inductance.mutuals import (
    _calc_circuit_plasma_mutual_inductances,
    _calc_coil_plasma_mutual_inductances,
    _calc_structure_mode_plasma_mutual_inductances,
    _calc_structure_plasma_mutual_inductances,
)
from device_inductance.utils import (
    calc_flux_density_from_flux,
    gradient_order4,
)

from .device import DeviceInductance


def plasma_response_linearization(
    device: DeviceInductance,
    plasma_current: float,
    plasma_resistance: float,
    plasma_self_inductance: float,
    plasma_current_density: NDArray,
    plasma_poloidal_flux: NDArray,
    actuator_current: NDArray,
    structure_current: NDArray,
    actuators: Literal["circuits", "coils"] = "circuits",
    structures: Literal["modes", "direct"] = "modes",
    gamma_multiplier: float = 1.0,
    include_plasma_in_system: bool = True,
    include_cross_terms: bool = True,
) -> PlasmaLinearization:
    """
    Linearization of the plasma+coil+structure system's current-voltage response
    and of the motion of the plasma centroid, for a given set of plasma condition and conductor currents.

    ## References

    * [1] M. L. Walker and D. A. Humphreys,
        “Valid Coordinate Systems for Linearized Plasma Shape Response Models in Tokamaks,”
        Fusion Science and Technology, vol. 50, no. 4, pp. 473-489, Nov. 2006, doi: 10.13182/FST06-A1271.

    ## Abbreviations

    * ...a: referring to actuators (coils or circuits)
    * na: number of actuators
    * ...s: referring to structures (structure filaments or modes)
    * ns: number of structure elements
    * ...x: referring to combined vector of actuator and structure currents
    * ncond: number of conducting elements (not including plasma)
    * ...p: referring to plasma in a scalar sense (not the meshgrid on which it is defined)
    * ...mesh: referring to the elements of the meshgrid on which the plasma fields are defined
    * nr, nz: number of grid elements
    * j: plasma _current_ distribution in [A], _not_ current density!
        This is a departure from the usual notation to prevent carrying
        excessive grid cell area multiplications.

    Args:
        plasma_current: [A] scalar total plasma toroidal current
        plasma_resistance: [ohm] scalar plasma total loop resistance
        plasma_self_inductance: [H] possibly calculated using device.calc_plasma_self_inductance()
        plasma_current_density: [A/m^2] with shape (nr, nz) plasma current density field
        plasma_poloidal_flux: [Wb] with shape (nr, nz) possibly calculated using device.calc_plasma_flux()
        actuator_current: [A] with shape (na, 1) current for actuators (circuits or coils)
        structure_current: [A] with shape (ns, 1)
        actuators: Whether to use coils or circuits as actuators. Defaults to "circuits".
        structures: Whether to use the full structure system or modal model reduction.
                    Defaults to "modes".
        gamma_multiplier: Artificial scaling to apply to vertical instability growth timescale.
                        Defaults to 1.0.
            This is sometimes used because rigid models underpredict gamma compared to nonrigid.
        include_plasma_in_system: Whether to include the plasma as a conducting element
                                in the state-space system. Defaults to True.
        include_cross_terms: whether to include cross-terms
                            in calculation of plasma current filament forces. Defaults to True.
            If this is flag is set, the effect of current centroid motion in `z`
            is included in the estimate of plasma filament forces in the `r` direction.

    Returns:
        State-space matrices, sensitivity of plasma current centroid to actuators,
        and various auxiliary outputs
    """
    # Shorthand
    ip = plasma_current  # [A]
    lp = plasma_self_inductance  # [H]
    rp = plasma_resistance  # [ohm]

    # Unpack mesh info
    rmesh, zmesh = device.meshes  # [m] 2d meshgrids
    dr, dz = device.dxgrid  # [m] grid step size

    # Get the base set of mutual inductances for the plasma
    psi_mesh_per_amp = device.plasma_flux_tables  # [Wb/A], (nrnz, nr, nz)
    mcp = _calc_coil_plasma_mutual_inductances(
        plasma_current,
        plasma_poloidal_flux,
        device.grids,
        device.coil_filament_rzn,
        show_prog=device.show_prog,
    )  # [H] coil-plasma mutual inductances

    mfp = _calc_structure_plasma_mutual_inductances(
        plasma_current,
        plasma_poloidal_flux,
        device.grids,
        device.structures,
        show_prog=device.show_prog,
    )  # [H] structure_filament-plasma mutual inductances

    # Figure out what combination of actuators and passive system we're using
    # and extract their various tables and matrices.
    # It might be better to move this to a different function at some point,
    # but for now, here it sits.
    maa: NDArray  # [H] actuator-actuator mutual inductance
    mas: NDArray  # [H] actuator-structure mutual inductance
    mss: NDArray  # [H] structure-structure mutual inductance
    map_: NDArray  # [H] actuator-plasma mutual inductance
    msp: NDArray  # [H] structure-plasma mutual inductance
    bra: NDArray  # [T/A] actuator-to-mesh B-field tables, r-component
    bza: NDArray  # [T/A] actuator-to-mesh B-field tables, z-component
    brs: NDArray  # [T/A] structure-to-mesh B-field tables, r-component
    bzs: NDArray  # [T/A] structure-to-mesh B-field tables, z-component
    psia: NDArray  # [Wb/A] actuator-to-mesh flux tables
    psis: NDArray  # [Wb/A] structure-to-mesh flux tables
    ra: NDArray  # [ohm] actuator resistance
    rs: NDArray  # [ohm] structure resistance
    #   Parts that depend on actuator choice
    if actuators == "circuits":
        maa = device.circuit_mutual_inductances
        bra, bza = device.circuit_flux_density_tables
        psia = device.circuit_flux_tables
        map_ = _calc_circuit_plasma_mutual_inductances(device.circuits, mcp, show_prog=device.show_prog)
        ra = device.circuit_resistances
        if structures == "modes":
            mas = device.circuit_structure_mode_mutual_inductances
        elif structures == "direct":
            mas = device.circuit_structure_mutual_inductances
        else:
            raise ValueError(f"Unrecognized structure kind `{structures}`")

    elif actuators == "coils":
        maa = device.coil_mutual_inductances
        bra, bza = device.coil_flux_density_tables
        psia = device.coil_flux_tables
        map_ = mcp
        ra = device.coil_resistances
        if structures == "modes":
            mas = device.coil_structure_mode_mutual_inductances
        elif structures == "direct":
            mas = device.coil_structure_mutual_inductances
        else:
            raise ValueError(f"Unrecognized structure kind `{structures}`")

    else:
        raise ValueError(f"Unrecognized actuator kind `{actuators}`")

    #   Parts that don't depend on actuator choice
    if structures == "modes":
        mss = device.structure_mode_mutual_inductances
        brs, bzs = device.structure_mode_flux_density_tables
        psis = device.structure_mode_flux_tables
        msp = _calc_structure_mode_plasma_mutual_inductances(mfp, device.structure_model_reduction)
        rs = device.structure_mode_resistances
    elif structures == "direct":
        mss = device.structure_mutual_inductances
        brs, bzs = device.structure_flux_density_tables
        psis = device.structure_flux_tables
        msp = mfp
        rs = device.structure_resistances
    else:
        raise ValueError(f"Unrecognized structure kind `{structures}`")

    # Unpack shapes
    nr, nz = device.nr, device.nz  # Number of grid points on each axis
    nrnz = nr * nz  # Total number of mesh points
    na, ns = mas.shape  # Number of actuators and structure elements
    nx = na + ns  # Number of conductors

    # Assemble combined matrices and tables
    x = np.concatenate(
        (actuator_current.flatten(), structure_current.flatten()), axis=0
    )  # [A] conductor current
    #     Each with shape (ncoils + nstruct, nr, nz) == (nx, nr, nz)
    brx_per_amp = np.concatenate((bra, brs), axis=0)  # [T/A] conductor Br per amp on mesh
    bzx_per_amp = np.concatenate((bza, bzs), axis=0)  # [T/A] conductor Bz per amp on mesh
    psix_per_amp = np.concatenate((psia, psis), axis=0)  # [Wb/A] conductor flux per amp on mesh

    #
    # Do calculations
    #

    # Magnetic field generated by conductors,
    # denoted "vacuum" because it is not caused by the plasma
    brvac = np.zeros_like(rmesh)  # [T]
    bzvac = np.zeros_like(rmesh)  # [T]
    for i in range(nx):
        brvac += x[i] * brx_per_amp[i, :, :]
        bzvac += x[i] * bzx_per_amp[i, :, :]

    # Current-filament representation of plasma
    jA = plasma_current_density * dr * dz  # [A] plasma current in each mesh cell

    # Current-centroid
    rc = np.sum(jA * rmesh) / plasma_current  # [m]
    # zc = np.sum(jA * zmesh) / plasma_current  # [m] zc is never explicitly used

    # Plasma mesh-cell-filament current gradient
    djdr, djdz = gradient_order4(jA, rmesh, zmesh)  # [A/m] shape (nr, nz)
    #   The sensitivity to the centroid is opposite, because the centroid moves
    #   the current distribution the opposite direction by the same amount.
    djdrc = -djdr  # [A/m] eqn. 5 from (1)
    djdzc = -djdz  # [A/m]

    # Plasma mesh-cell-filament force gradient
    # Eqns. 8, 11 from (1)
    dfrdx_part = 2.0 * np.pi * rmesh * jA
    dfzdx_part = -2.0 * np.pi * rmesh * jA
    dfrdx = np.zeros((1, nx))
    dfzdx = np.zeros((1, nx))
    for i in range(0, nx):
        dfrdx[0, i] = np.sum(bzx_per_amp[i, :, :] * dfrdx_part)
        dfzdx[0, i] = np.sum(brx_per_amp[i, :, :] * dfzdx_part)

    # IxB forces on centroid
    # Eqns. 9, 10 from (1)
    dfrhoopdrc = MU_0 * ip**2 / (2.0 * rc)  # [N/m] scalar plasma self-loading
    dfrvacdrc = -2.0 * np.pi * np.sum(rmesh * djdr * bzvac)  # [N/m]
    dfrdrc = dfrhoopdrc + dfrvacdrc  # [N/m] total force gradient including conductor and plasma self-effect
    dfzdzc = 2.0 * np.pi * np.sum(rmesh * djdz * brvac)  # [N/m]
    #    Handle terms that may or may not be included
    dfrdzc: float  # [N/m]
    dfzdrc: float  # [N/m]
    if include_cross_terms:
        dfrdzc = float(-2.0 * np.pi * np.sum(rmesh * djdz * bzvac))
        dfzdrc = float(2.0 * np.pi * np.sum(rmesh * djdr * brvac))
    else:
        dfrdzc = 0.0
        dfzdrc = 0.0
    #    Solve for sensitivity of centroid location to conductor current
    #    Eqn. 6 from (1), modified if cross terms are included
    dfrzdx = np.array([dfrdx, dfzdx]).squeeze()  # [N/A], shape (2, nx)
    dfrzdrzc = np.array([[dfrdrc, dfrdzc], [dfzdrc, dfzdzc]]).T  # [N/m]
    drzcdx = np.linalg.inv(dfrzdrzc) @ dfrzdx  # [m/A]
    drcdx: NDArray = drzcdx[0, :]  # [m/A]
    dzcdx: NDArray = drzcdx[1, :]  # [m/A]

    # Plasma current distribution sensitivity to conductor current
    djdx = np.zeros((nx, nr, nz))  # [A/A], technically [dimensionless]
    for i in range(nx):
        djdx[i, :, :] = djdrc * drcdx[i] + djdzc * dzcdx[i]

    # Conductor mutual inductance modification due to plasma motion
    # Note that reshape and transpose do not reallocate, so this will be
    # relatively efficient memory-wise but may drive additional indexing
    # overhead due to using array views.
    xmat = psix_per_amp.reshape((nx, nrnz)) @ djdx.reshape((nx, nrnz)).T  # [H]

    # Unmodified mutual inductance matrix
    # fmt: off
    mmat = np.block(
        [
            [maa,     mas,    map_],
            [mas.T,   mss,    msp],
            [map_.T,  msp.T,  lp]]
    )  # [H], unmodified inductance matrix
    # fmt: on
    # Modified mutual inductance matrix
    lstar = mmat.copy()  # [H]
    lstar[:nx, :nx] += xmat

    # Adjust tables and matrices for inclusion/exclusion of plasma in the system
    if include_plasma_in_system:
        djdx = np.concatenate((djdx, plasma_current_density.reshape((1, nr, nz))), axis=0)  # [A/A]
        drcdx = np.block([drcdx, np.array(0.0)])  # [m/A]
        dzcdx = np.block([dzcdx, np.array(0.0)])  # [m/A]
        # NOTE: the reference prototype had psip_per_amp set to zero everywhere
        psip_per_amp = (plasma_poloidal_flux / ip).reshape((1, nr, nz))  # [Wb/A]
        psix_per_amp: NDArray = np.concatenate((psix_per_amp, psip_per_amp), axis=0)  # [Wb/A]
        rx = np.concatenate((np.diag(ra), np.diag(rs), np.array(rp).reshape((1,))), axis=0)  # [ohm]
    else:
        mmat = mmat[:nx, :nx]  # [H]
        lstar = lstar[:nx, :nx]  # [H]
        rx = np.concatenate((np.diag(ra), np.diag(rs)), axis=0)  # [ohm]

    # State-space system
    lstari = np.linalg.inv(lstar)  # [1/H]
    amat = -lstari @ np.diag(rx)  # [1/s], shape (nx + ?1, nx + ?1)
    bmat = lstari[:, :na]  # [1/H], shape (nx + ?1, na)

    # Mesh flux sensitivity
    #    Sensitivity of plasma flux to change in actuator flux, due to plasma motion
    #    Shape (nrnz, nx + ?1)
    nxmod = nx + int(include_plasma_in_system)  # Number of current-states, possibly including plasma
    dpsipladx = psi_mesh_per_amp.reshape((nrnz, nrnz)) @ djdx.reshape((nxmod, nrnz)).T  # [Wb/A]
    #    Sensitivity of total flux to change in actuator flux, including via plasma motion effect
    #    Shape (nrnrz, 1)
    dpsidx = dpsipladx + psix_per_amp.reshape((nxmod, nrnz)).T  # [Wb/A]
    dpsidrc = psi_mesh_per_amp.reshape((nrnz, nrnz)) @ djdrc.reshape((nrnz, 1))  # [Wb/A/m]
    dpsidzc = psi_mesh_per_amp.reshape((nrnz, nrnz)) @ djdzc.reshape((nrnz, 1))  # [Wb/A/m]

    # Scale eigenvalues
    d, v = np.linalg.eig(amat)  # Needed to extract gamma later, scaled or not
    if gamma_multiplier != 1.0:
        # Sort eigenvalues by magnitude of real part
        inds = np.flip(np.argsort(d.real))
        d = d[inds]
        v = v[:, inds]
        # If there is a vertical instability (which may not be present, depending on conditions)
        # then scale it
        if d[0] > 0.0:
            # Scale the eigenvalue that should correspond to vertical instability
            d[0] *= gamma_multiplier
            # Reconstruct `A` from the new eigenvalues.
            # https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Diagonalization_and_the_eigendecomposition
            # NOTE: the reference prototype uses non-unique right-side inverse,
            #       while this version uses a left-side inverse because the right-side inverse
            #       is not immediately available in numpy or scipy and would need to be constructed
            #       as a pseudoinverse
            amat = np.linalg.inv(v) @ (np.diag(d) @ v)
        else:
            # If this plasma is vertically-stable, leave `A` as-is
            pass
    else:
        # If we're not rescaling, leave `A` as-is
        pass

    # Extract the most-positive real eigenvalue of `A`
    # which should represent the vertical instability growth rate
    # if that instability is present
    gamma = float(np.max(d.real))

    # Ideal sensor responses
    bpols = device.poloidal_field_probes
    fflux = device.full_flux_loops
    pflux = device.partial_flux_loops
    nbpol = len(bpols)
    nfflux = len(fflux)
    npflux = len(pflux)
    cbpol = np.zeros((nxmod, nbpol))  # [T/A] C-matrix components
    cfflux = np.zeros((nxmod, nfflux))  # [Wb/A]
    cpflux = np.zeros((nxmod, npflux))  # [Wb/A]
    for i, dpsidxi in enumerate(dpsidx.T):
        # Calculate B-field tables locally to prevent actualizing them all at the same time
        dbrdx, dbzdx = calc_flux_density_from_flux(dpsidxi.reshape((nr, nz)), *device.meshes)
        # Response of each sensor to each conductor's current
        for j, s in enumerate(bpols):
            cbpol[i, j] = s.response(device.grids, dbrdx, dbzdx)
        for j, s in enumerate(fflux):
            cfflux[i, j] = s.response(device.grids, dpsidxi)
        for j, s in enumerate(pflux):
            cpflux[i, j] = s.response(device.grids, dbrdx, dbzdx)
    #    Combine C-matrix components into one matrix
    cmat = np.concatenate((cbpol, cfflux, cpflux), axis=1).T  # Various units

    # Pack output
    out = PlasmaLinearization(
        amat=amat,
        bmat=bmat,
        cmat=cmat,
        drcdx=drcdx,
        dzcdx=dzcdx,
        mmat=mmat,
        xmat=xmat,
        lstar=lstar,
        lstari=lstari,
        djdx=djdx,
        dpsidx=dpsidx,
        dpsipladx=dpsipladx,
        dpsivacdx=psix_per_amp,
        dpsidrc=dpsidrc,
        dpsidzc=dpsidzc,
        res=rx,
        gamma=gamma,
    )

    return out


@dataclass(frozen=True)
class PlasmaLinearization:
    """
    Linearization of the plasma current distribution response to actuator current.
    Includes system matrices and auxiliary outputs.

    The state-space system here is

    `xdot = Ax + Bu = dI/dt`
    where
        x: [A] conductor current vector (including plasma, if include_plasma_in_system is set)
        u: [V] actuator applied voltage
        A: [1/s] `-(Lstar^-1) @ resistance`, with plasma-modified inductance matrix Lstar
        B: [1/H] `Lstar^-1` trimmed to just actuator entries,
            inductive response to applied voltage on actuators

    and
    `Y = Cx + Du = V_sensors`
    where Y is the ideal integrated response on the sensors due to changing system current `x`,
    and D is zero due to sensors not responding directly to applied voltage on the actuators.

    The linearization also produces an estimate of the motion of the plasma current centroid in (r,z)
    with changing actuator current, which is useful in a similar system describing control of the
    plasma position using a subset of the same actuators.

    All fields should be treated as immutable, as they may include references to fields in a
    DeviceInductance instance. If there is a need to modify one of the returned arrays,
    first make a copy like `djdx_local = djdx.copy()` to avoid modifying the original.
    """

    # State-space matrices & current-centroid sensitivity
    amat: NDArray
    """
    [1/s], State-space dynamics A matrix.
    Formulated like `amat = -lstari * diag(res)`.
    Physically represents the system's internal inductive-resistive timescales.
    """
    bmat: NDArray
    """
    [1/H], State-space dynamics B matrix.
    Formulated like `bmat = lstari[actuator_entries]`.
    Physically represent the system's current response to applied voltage on the actuators.
    """
    cmat: NDArray
    """
    Various units, State-space dynamics C matrix.
    Formulated like [bpol, fflux, pflux] to combine all sensors' responses into one matrix.
    Physically represents the sensors' ideal integrated response to B- and flux- fields.
    """
    drcdx: NDArray
    """[m / A] with shape (1, ncond), Response of current centroid r to conducting elements (x)"""
    dzcdx: NDArray
    """[m / A] with shape (1, ncond), Response of current centroid z to conducting elements (x)"""

    # Inductance matrix components
    mmat: NDArray
    """
    [H] with shape (ncond + ?1, ncond + ?1), Mutual inductance matrix _without_ adjustment
    for coupling through plasma motion. 
    
    If plasma inductance components are included, this matrix includes the plasma self- and mutual-
    inductance components in the last row and column.
    """
    xmat: NDArray
    """
    [H] with shape (ncond + ?1, ncond + ?1), 
    The `x` matrix used in dynamics model (lstar = xmat + mmat).

    This is the first-order effective change in mutual inductance 
    between conductor elements caused by 
    plasma motion in response to changing conductor current.

    If plasma inductance components are included, this matrix is expanded to include a zero
    row and column for the plasma entry.
    """
    lstar: NDArray
    """
    [H] with shape (ncond + ?1, ncond + ?1), Plasma-modified mutual inductance.
    Formulated like `lstar = mmat + xmat`.
    """
    lstari: NDArray
    """[1/H] with shape (ncond + ?1, ncond + ?1), 
    Inverse of plasma-modified mutual inductance."""

    # Aux outputs
    djdx: NDArray
    """[A/A] with shape (nr*nz, ncond), 
    Response of plasma current distribution to conducting elements (x)"""

    dpsidx: NDArray
    """[Wb / A] with shape (nr*nz, ncond),
    Total response of flux distribution to conducting elements (x)"""
    dpsipladx: NDArray
    """[Wb / A] with shape (nr*nz, ncond),
    Plasma response of flux distribution to conducting elements (x)"""
    dpsivacdx: NDArray
    """
    [Wb / A] with shape (nr*nz, ncond), 
    Vacuum (total-less-plasma) response of flux distribution to conducting elements (x)
    """

    dpsidrc: NDArray
    """[Wb / m] with shape (nr*nz, 1), Response of flux distribution to current centroid r"""
    dpsidzc: NDArray
    """[Wb / m] with shape (nr*nz, 1), Response of flux distribution to current centroid z"""

    res: NDArray
    """
    [ohm] with shape (ncond + ?1), Resistance vector.
    If `include_plasma_in_system` is set, plasma resistance is included as the last entry.
    """
    gamma: float
    """
    [1/s] Usually represents the growth timescale of plasma vertical instability.
    This is largest positive real eigenvalue of `amat`.
    """
