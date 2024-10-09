from dataclasses import dataclass

import numpy as np
from omas import ODS

from cfsem import (
    self_inductance_lyle6,
    flux_circular_filament,
    self_inductance_annular_ring,
    self_inductance_circular_ring_wien,
)


@dataclass(frozen=True)
class CoilFilament:
    r: float
    """[m] radial location"""

    z: float
    """[m] z location"""

    n: float
    """[dimensionless] number of turns"""

    self_inductance: float
    """[H] scalar self-inductance of this filament"""


@dataclass(frozen=True)
class Coil:
    name: str
    """This name should match the name used in the device description ODS"""

    resistance: float
    """[ohm] total effective resistance of the coil; for superconducting coils, this will be small"""

    self_inductance: float
    """[H] total scalar self-inductance of this coil"""

    filaments: list[CoilFilament]
    """Discretized circular filaments describing the coil's winding pattern"""


def _extract_coils(description: ODS) -> list[Coil]:
    """
    Extract coil filamentization and, while full geometric info is available,
    calculate self-inductance of individual filaments and of the coil as a whole.

    Because some coils are not of rectangular cross-section, and even some coils of
    rectangular cross-section do not have evenly-distributed number of turns between
    different elements, the coil's self-inductance is calculated by using the singularity
    method to calculate the mutual inductance between each pair of elements, then replacing
    the singular self-field terms with the corresponding element's estimated self-inductance.

    An approximate calc is used for the self-inductance of individual elements, which
    can't use the singularity method. The existing method adequately handles
    rectangular-section elements, but approximates each non-rectangular element as a
    square section with the same area, which may introduce some error when handling elements
    of circular, annular, or other cross-sectional geometry. More detailed handling can be
    added later, giving higher resolution for such cases.

    Args:
        description: Device geometric info in the format produced by device_description

    Raises:
        ValueError: If an un-handled type of coil element cross-sectional geometry is encountered

    Returns:
        A list of coil objects, populated with reduced geometric info and estimated self-inductances.
    """

    coils: list[Coil] = []
    for ods_coil in description["pf_active.coil"].values():
        coil_name = ods_coil["name"]
        resistance = ods_coil["resistance"]
        coil_filaments: list[CoilFilament] = []

        # Process individual elements
        for coil_elem in ods_coil["element"].values():
            geom_type = coil_elem["geometry.geometry_type"]
            turns_with_sign = coil_elem["turns_with_sign"]  # [dimensionless]
            n = abs(turns_with_sign)  # Circuit definition is responsible for sign

            # Approximate the self-inductance of the individual elements
            # as rectangular sections, solid rings, or annular rings
            # depending on geometry type id.
            if geom_type == 5:
                # Annular section (or, with zero inner radius, solid circular)
                r = coil_elem["geometry.annulus.r"]  # [m]
                z = coil_elem["geometry.annulus.z"]  # [m]
                ri = coil_elem["geometry.annulus.radius_inner"]  # [m]
                ro = coil_elem["geometry.annulus.radius_outer"]  # [m]

                if ri > 1e-4:
                    elem_self_inductance = self_inductance_annular_ring(
                        r, ri, ro
                    )  # [H]
                else:
                    # Use solid ring calc for small inner radius to avoid div/0
                    elem_self_inductance = self_inductance_circular_ring_wien(
                        r, ro
                    )  # [H]

            elif geom_type == 2:
                # Solid rectangular section
                r = coil_elem["geometry.rectangle.r"]  # [m]
                z = coil_elem["geometry.rectangle.z"]  # [m]
                w = coil_elem["geometry.rectangle.width"]  # [m]
                h = coil_elem["geometry.rectangle.height"]  # [m]

                elem_self_inductance = self_inductance_lyle6(r, w, h, n)  # [H]
            else:
                raise ValueError(f"Unhandled coil element geometry type: {geom_type}")

            # Store the parts we need for calculating mutual inductances
            coil_filaments.append(
                CoilFilament(r=r, z=z, n=n, self_inductance=float(elem_self_inductance))
            )

        # Calculate self-inductance of the whole coil
        coil_self_inductance = 0.0  # [H]
        elem_rs = np.array([x.r for x in coil_filaments])  # [m]
        elem_zs = np.array([x.z for x in coil_filaments])  # [m]
        elem_ns = np.array([x.n for x in coil_filaments])  # [dimensionless]
        elem_self_inductances = np.array(
            [x.self_inductance for x in coil_filaments]
        )  # [H]
        nelem = len(elem_rs)
        for i in range(nelem):
            this_r = np.array([elem_rs[i]])  # [m]
            this_z = np.array([elem_zs[i]])  # [m]
            this_n = np.array([elem_ns[i]])  # [dimensionless]

            # Use one-to-many flux calc for speed
            contribs = elem_ns * flux_circular_filament(
                ifil=this_n,  # Unit current multiplied by number of turns
                rfil=this_r,
                zfil=this_z,
                rprime=np.array(elem_rs),
                zprime=np.array(elem_zs),
            )
            # Use precalcualted self-contribution which is otherwise singular and handled separately
            contribs[i] = elem_self_inductances[i]

            # Total contribution to coil self-inductance
            turn_contrib = np.sum(contribs)
            coil_self_inductance += float(turn_contrib)

        coil = Coil(
            name=coil_name,
            resistance=resistance,  # [ohm]
            self_inductance=coil_self_inductance,  # [H]
            filaments=coil_filaments,
        )
        coils.append(coil)

    return coils
