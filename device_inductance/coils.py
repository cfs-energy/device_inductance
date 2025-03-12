from dataclasses import dataclass
from typing import Optional

from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from omas import ODS

from cfsem import (
    self_inductance_lyle6,
    flux_circular_filament,
    self_inductance_annular_ring,
    self_inductance_circular_ring_wien,
)

from .utils import solve_flux_axisymmetric, calc_flux_density_from_flux

from interpn import MulticubicRectilinear

from device_inductance.logging import log


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
    def grids(self) -> Optional[tuple[NDArray, NDArray]]:
        """Generate a set of regular r,z grids that span the coil winding pack centers exactly
        if possible, or None if the winding pack can't be represented exactly.

        Adds 4 grid cells of padding around the winding pack to deconflict the
        cells with nonzero current density from the boundary conditions of a
        flux solve.

        If only one unit cell is present on either axis, the grid will be
        expanded 1cm in either direction.
        """

        # Get coordinates with a unit cell
        unique_r = np.array(sorted(list(set([f.r for f in self.filaments]))))  # [m]
        unique_z = np.array(sorted(list(set([f.z for f in self.filaments]))))

        # Make sure there are enough unit cells to work with,
        # expanding dimensions if necessary
        if len(unique_r) == 1:
            r = unique_r[0]
            unique_r = [r - 1e-2, r, r + 1e-2]
        if len(unique_z) == 1:
            z = unique_z[0]
            unique_z = [z - 1e-2, z, z + 1e-2]
        if len(unique_r) < 2 or len(unique_z) < 2:
            log().error("Failed to expand coil grid dimensionality")
            return None

        # Check if the coordinates have regular spacing,
        # which is required to support the finite difference solve
        drs = np.diff(unique_r)
        drmean = np.mean(drs)
        if np.any(np.abs(drs - drmean) / drmean > 1e-4):
            log().warning(
                f"Coil {self.name} filaments are not on a regular grid; skipping grid for smooth self-field"
            )
            return None
        dzs = np.diff(unique_z)
        dzmean = np.mean(dzs)
        if np.any(np.abs(dzs - dzmean) / dzmean > 1e-4):
            log().warning(
                f"Coil {self.name} filaments are not on a regular grid; skipping grid for smooth self-field"
            )
            return None

        # Extend grids by a few cells outside the winding pack
        # 7 is the true minimum; 2x 4th-order finite difference patches
        # will be stacked to extract the flux then the flux density, which means
        # 7 cells see direct interaction with boundary conditions and must not
        # have nonzero current density to produce sane results; npad = 6 produces junk outputs.
        npad = 7
        nr = len(unique_r) + 2 * npad
        nz = len(unique_z) + 2 * npad
        r_pad = npad * drmean
        z_pad = npad * dzmean
        rgrid = np.linspace(unique_r[0] - r_pad, unique_r[-1] + r_pad, nr)
        zgrid = np.linspace(unique_z[0] - z_pad, unique_z[-1] + z_pad, nz)

        # Make sure the grid doesn't cross zero.
        # If this check becomes a problem, there is an alternate strategy
        # to double resolution and spread the coil's current density mapping
        # across more than one neighboring cell, but that is too much complexity
        # to implement proactively.
        if rgrid[0] < 0.0:
            log().error(f"Coil {self.name} grid for smooth self-field crossed r=0")
            return None

        return (rgrid, zgrid)

    @cached_property
    def meshes(self) -> Optional[tuple[NDArray, NDArray]]:
        """Generate a set of regular r,z meshes that span the coil winding pack centers exactly
        if possible, or None if the winding pack can't be represented exactly.

        Adds 4 grid cells of padding around the winding pack to deconflict the
        cells with nonzero current density from the boundary conditions of a
        flux solve.
        """
        grids = self.grids
        if grids is not None:
            rmesh, zmesh = np.meshgrid(*grids, indexing="ij")
            return (rmesh, zmesh)  # Unpack and repack for pyright...
        else:
            return None

    @cached_property
    def local_fields(self) -> Optional[tuple[NDArray, NDArray, NDArray]]:
        """
        Solve the local self-field flux and flux density per amp by mapping the
        coil section to a continuous current density distribution and solving the
        continuous flux field via 4th-order finite difference, then extracting the
        B-field from the flux field via 4th-order finite difference.

        Returns:
            (psi, br, bz) [Wb/A, T/A, T/A] 2D arrays of poloidal flux and flux density per amp of coil current
        """
        grids = self.grids
        meshes = self.meshes
        if grids is not None and meshes is not None:
            rgrid, zgrid = grids
            rmesh, zmesh = meshes
            dr = rgrid[1] - rgrid[0]  # [m]
            dz = zgrid[1] - zgrid[0]  # [m]
            area = dr * dz  # [m^2]

            # Map current density per amp
            jtor = np.zeros_like(meshes[0])  # [A-turns/m^2 / A]
            for f in self.filaments:
                # Get indices of location of this filament
                ri = np.argmin(np.abs(rgrid - f.r))
                zi = np.argmin(np.abs(zgrid - f.z))
                # Set current density for that unit cell
                # so that the total for the cell comes out to the
                # correct total current
                jtor[ri, zi] += f.n / area

            # Solve flux field
            psi = solve_flux_axisymmetric(grids, meshes, jtor)  # [Wb/A]

            # Extract flux density
            br, bz = calc_flux_density_from_flux(psi, rmesh, zmesh)  # [T/A]

            return psi, br, bz

        else:
            return None

    @cached_property
    def local_field_interpolators(
        self,
    ) -> Optional[
        tuple[MulticubicRectilinear, MulticubicRectilinear, MulticubicRectilinear]
    ]:
        """Build interpolators over the solved local fields, if available"""
        grids = self.grids
        local_fields = self.local_fields

        if grids is not None and local_fields is not None:
            psi, br, bz = local_fields
            grids = [x for x in grids]
            psi_interp = MulticubicRectilinear.new(grids, psi)
            br_interp = MulticubicRectilinear.new(grids, br)
            bz_interp = MulticubicRectilinear.new(grids, bz)

            return psi_interp, br_interp, bz_interp
        else:
            return None

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
