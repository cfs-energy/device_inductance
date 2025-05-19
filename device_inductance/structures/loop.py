"""Top-level discretization of conducting structure."""

from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import cfsem

from shapely import Polygon
from device_inductance import mesh

from .input import PassiveStructureInput
from .slicer import RadialSlicer
from .heuristics import poly_angle, unroundness, MAX_EDGE_LENGTH_M
from .filament import PassiveStructureFilament, _mesh_elem_to_fil


@dataclass(frozen=True)
class PassiveStructureLoop:
    """
    A logical chunk of structure material that may be the result of slicing a larger
    object into smaller pieces. Composed of one or more filaments.
    """

    # Inputs
    parent_name: str
    """Name of the input structure this was chunked from"""
    polygon: Polygon
    """Shape of the enclosing polygon before sub-discretization"""

    # Discretization results
    filaments: list[PassiveStructureFilament]  # After meshing

    @cached_property
    def rs(self) -> NDArray:
        """[m] Filament radial coordinates"""
        return np.array([f.r for f in self.filaments])

    @cached_property
    def zs(self) -> NDArray:
        """[m] Filament axial coordinates"""
        return np.array([f.z for f in self.filaments])

    @cached_property
    def ns(self) -> NDArray:
        """
        [dimensionless] (Fractional) number of turns of each filament, weighted according to their
        cross-sectional area as a fraction of this loop's total.
        """
        return np.array([f.polygon.area / self.polygon.area for f in self.filaments])

    @cached_property
    def resistance(self) -> float:
        """[ohm] Total loop resistance; effective parallel resistance over all filaments"""
        # This resistance calc treats the individual filaments as wired in parallel.
        resistance = 1.0 / sum([1.0 / f.resistance for f in self.filaments])
        return resistance  # [ohm]

    @cached_property
    def self_inductance(self) -> float:
        """[H] Self-inductance of this loop."""
        # Because the filaments within a chunk are assumed to be in parallel and isopotential on the section,
        # each one is accounted as only a fraction of a full turn - otherwise, the calculated inductance
        # would diverge as the discretization becomes finer.
        fil_frac_of_loop = self.ns
        self_inductance = 0.0  # [H]
        for i, f in enumerate(self.filaments):
            r = np.atleast_1d(f.r)
            z = np.atleast_1d(f.z)
            ref_current = fil_frac_of_loop[i]  # [A]
            mutuals = fil_frac_of_loop * cfsem.flux_circular_filament(
                ref_current, r, z, self.rs, self.zs
            )
            # Replace singularity with analytic estimate
            mutuals[i] = fil_frac_of_loop[i] ** 2 * f.self_inductance
            self_inductance += np.sum(mutuals)

        # This loop may be the result of discretizing a larger chunk of material,
        # in which case it does not represent a full loop by itself.
        # Because the self inductance is really the mutual inductance from self to self,
        # the fraction of loop needs to be accounted twice.
        # This overall fraction of loop is accounted in `self.ns` and does not need to be repeated here.
        self_inductance = float(self_inductance)

        return self_inductance  # [H]

    def mutual_inductance(self, other: PassiveStructureLoop) -> float:
        """
        [H] Mutual inductance between two loops.
        If `self` passed as `other`, the precalculated self-inductance is returned.
        """

        if id(other) == id(self):
            return self.self_inductance

        # Each loop's `ns` includes accounting of both filament number of turns and overall number of turns
        rzn1 = np.array([self.rs, self.zs, self.ns])  # [m], [m], [dimensionless]
        rzn2 = np.array([other.rs, other.zs, other.ns])

        m = cfsem.mutual_inductance_of_cylindrical_coils(rzn1, rzn2)

        return m  # [H]

    @classmethod
    def from_poly(
        cls,
        parent_name: str,
        polygon: Polygon,
        resistivity: float,
        max_edge_length_m: float = MAX_EDGE_LENGTH_M,
    ) -> PassiveStructureLoop:
        """
        Discretize a structure that is reasonably well-associated with a single centroid
        and may be the result of earlier sub-division of a larger structure, but may not be
        fully discretized yet.
        """
        # Subdivide by meshing
        sub_polygons: list[Polygon] = mesh._mesh_region(
            np.array(polygon.boundary.segmentize(max_edge_length_m).xy).T
        )

        # Make a filament from each mesh cell
        filaments = [
            _mesh_elem_to_fil(p, resistivity, parent_name) for p in sub_polygons
        ]

        # Call the collection of filaments a loop
        loop = PassiveStructureLoop(parent_name, polygon, filaments)

        return loop

    @classmethod
    def from_input(
        cls,
        inp: PassiveStructureInput,
        slicer: RadialSlicer,
        angle_thresh_deg: float = 20.0,
        unroundness_thresh: float = 2.0,
    ) -> list[PassiveStructureLoop]:
        """
        Make one or more PassiveStructureLoop from an input element.
        The input element may be subdivided into multiple loops
        if it subtends a large angle relative to the centroid
        and has a large perimeter-to-area ratio in the section plane.
        """
        # If something spans a large region around the centroid
        # AND it's not a conceptually solid block of material,
        # subdivide it so that we get adequate detail about current in different regions.
        angle_thresh_met = poly_angle(inp.polygon, slicer.centroid) > np.deg2rad(
            angle_thresh_deg
        )
        unroundness_thresh_met = unroundness(inp.polygon) > unroundness_thresh

        if angle_thresh_met and unroundness_thresh_met:
            # Slice into angular chunks
            chunks: list[Polygon] = slicer.slice(inp.polygon)
        else:
            # If the original takes up a small angular region or it's a solid block, use it as-is
            chunks: list[Polygon] = [inp.polygon]

        # Each chunk represents a fraction of one contiguous loop;
        # if we were to treat each chunk as a whole loop, the inductance of the system
        # would diverge with increasing discretization.
        # Each sub-loop's fraction of loop is weighted based on its section area.
        return [cls.from_poly(inp.parent_name, p, inp.resistivity) for p in chunks]
