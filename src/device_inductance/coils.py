from dataclasses import dataclass
from functools import cached_property

import numpy as np
from cfsem import (
    flux_circular_filament,
    self_inductance_annular_ring,
    self_inductance_circular_ring_wien,
    self_inductance_lyle6,
)
from interpn import MulticubicRegular
from numpy.typing import NDArray
from omas import ODS
from shapely import Polygon

from device_inductance.local import LocalFields, local_fields


@dataclass(frozen=True)
class CoilFilament:
    """
    A discretized element of an axisymmetric magnet.
    Self-inductance is calculated based on conductor geometry.
    """

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
    """An axisymmetric magnet, which may not have a rectangular cross-section"""

    name: str
    """This name should match the name used in the device description ODS"""

    resistance: float
    """[ohm] total effective resistance of the coil; for superconducting coils, this will be small"""

    self_inductance: float
    """[H] total scalar self-inductance of this coil"""

    filaments: list[CoilFilament]
    """Discretized circular filaments describing the coil's winding pattern"""

    @cached_property
    def grids(self) -> tuple[NDArray, NDArray]:
        """Generate a grid that spans the winding pack, landing exactly on the windings if possible."""
        return self.local_fields.grids

    @cached_property
    def meshes(self) -> tuple[NDArray, NDArray]:
        """Generate a meshgrid that spans the winding pack, landing exactly on the windings if possible."""
        return self.local_fields.meshes

    @cached_property
    def local_fields(self) -> LocalFields:
        """
        Solve the local self-field flux and flux density per amp by mapping the
        coil section to a continuous current density distribution and solving the
        continuous flux field via 4th-order finite difference, then extracting the
        B-field from the flux field via 4th-order finite difference.

        Returns:
            Structure containing local field map components and interpolators
        """
        # Polygon section representation
        # As a heuristic, treat each filament as a rectangle with side length that is
        # half the smallest distance between two sequential filaments.
        # These polygons are only used if the windings do not fall on a regular grid
        # or are too close to r=0 to allow padding the grid to preserve boundary conditions.
        drs = np.diff(self.rs)
        dzs = np.diff(self.zs)
        w = np.min(np.linalg.norm((drs, dzs), axis=0)) / 2

        polygons = [
            Polygon.from_bounds(r - w / 2, z - w / 2, r + w / 2, z + w / 2)
            for r, z in zip(self.rs, self.zs, strict=True)
        ]

        # Point-source representation
        fil_rzn = (self.rs, self.zs, self.ns)

        # Local field solve
        return local_fields(fil_rzn, polygons)

    @cached_property
    def local_field_interpolators(
        self,
    ) -> tuple[MulticubicRegular, MulticubicRegular, MulticubicRegular]:
        """Build interpolators over the solved local fields, if available"""
        return (
            self.local_fields.psi_per_amp_interpolator,
            self.local_fields.br_per_amp_interpolator,
            self.local_fields.bz_per_amp_interpolator,
        )

    @cached_property
    def extent(self) -> tuple[float, float, float, float]:
        """[m] rmin, rmax, zmin, zmax extent of filament centers"""
        r = [f.r for f in self.filaments]
        z = [f.z for f in self.filaments]
        return min(r), max(r), min(z), max(z)

    @cached_property
    def rs(self) -> NDArray:
        """[m] Filament r-coordinates"""
        return np.array([f.r for f in self.filaments])

    @cached_property
    def zs(self) -> NDArray:
        """[m] Filament z-coordinates"""
        return np.array([f.z for f in self.filaments])

    @cached_property
    def ns(self) -> NDArray:
        """[dimensionless] Filament number of turns"""
        return np.array([f.n for f in self.filaments])


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
                    elem_self_inductance = self_inductance_annular_ring(r, ri, ro)  # [H]
                else:
                    # Use solid ring calc for small inner radius to avoid div/0
                    elem_self_inductance = self_inductance_circular_ring_wien(r, ro)  # [H]

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
            coil_filaments.append(CoilFilament(r=r, z=z, n=n, self_inductance=float(elem_self_inductance)))

        # Calculate self-inductance of the whole coil
        coil_self_inductance = 0.0  # [H]
        elem_rs = np.array([x.r for x in coil_filaments])  # [m]
        elem_zs = np.array([x.z for x in coil_filaments])  # [m]
        elem_ns = np.array([x.n for x in coil_filaments])  # [dimensionless]
        elem_self_inductances = np.array([x.self_inductance for x in coil_filaments])  # [H]
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
