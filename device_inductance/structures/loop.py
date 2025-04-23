from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import cfsem

from shapely import Polygon
import device_inductance

from .slicer import RadialSlicer
from .heuristics import poly_angle, unroundness

from . import MAX_EDGE_LENGTH_M


@dataclass(frozen=True)
class PassiveStructureLoop:
    """
    A logical chunk of structure material that may be the result of slicing a larger
    object into smaller pieces. Composed of one or more filaments.
    """

    # Inputs
    parent_name: str
    """Name of the input structure this was chunked from"""
    original_polygon: Polygon
    """Shape of the enclosing polygon before sub-discretization"""
    frac_of_loop: float
    """[dimensionless] What fraction of a full loop this represents; if the original input was chunked into
    `n` loops, this represents a `1/n` fraction of a loop."""

    # Discretization results
    filaments: list[device_inductance.PassiveStructureFilament]  # After meshing

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
        """[dimensionless] (Fractional) number of turns of each filament"""
        return np.ones_like(self.rs) / float(len(self.rs))

    @cached_property
    def resistance(self) -> float:
        """[ohm] Total loop resistance; effective parallel resistance over all filaments"""
        # Filament resistance already includes accounting of the area, which incorporates the effect
        # of self.frac_of_loop, so we don't need to bring that factor into the calc here.
        resistance = 1.0 / sum([1.0 / f.resistance for f in self.filaments])
        return resistance  # [ohm]

    @cached_property
    def self_inductance(self) -> float:
        """[H] Self-inductance with accounting for `frac_of_loop` for both sub-filaments and the loop as a whole."""
        # Because the filaments within a chunk are assumed to be in parallel and isopotential on the section,
        # each one is accounted as only a fraction of a full turn - otherwise, the calculated inductance
        # would diverge as the discretization becomes finer.
        nfils = len(self.rs)
        fil_frac_of_loop = np.atleast_1d([1.0 / float(nfils)])
        ref_current = np.ones((1,))  # [A]
        self_inductance = 0.0  # [H]
        for i, f in enumerate(self.filaments):
            r = np.atleast_1d(f.r)
            z = np.atleast_1d(f.z)
            mutuals = (fil_frac_of_loop**2) * cfsem.flux_circular_filament(
                ref_current, r, z, self.rs, self.zs
            )
            mutuals[i] = (
                fil_frac_of_loop[0] * f.self_inductance
            )  # Replace singularity with analytic estimate
            self_inductance += np.sum(mutuals)

        # This loop may be the result of discretizing a larger chunk of material,
        # in which case it does not represent a full loop by itself.
        # Because the self inductance is really the mutual inductance from self to self,
        # the fraction of loop needs to be accounted twice.
        self_inductance = (self.frac_of_loop**2) * float(self_inductance)

        return self_inductance  # [H]

    def mutual_inductance(self, other: PassiveStructureLoop) -> float:
        """
        [H] Mutual inductance between two loops, accounting for their `frac_of_loop` which may be
        non-unit if they were made by discretizing a larger loop.
        If `self` passed as `other`, the precalculated self-inductance is returned.
        """

        if id(other) == id(self):
            # Self-inductance already has `self.frac_of_loop` accounted
            return self.self_inductance

        rzn1 = np.array([self.rs, self.zs, self.ns])  # [m], [m], [dimensionless]
        rzn2 = np.array([other.rs, other.zs, other.ns])

        m = (
            self.frac_of_loop
            * other.frac_of_loop
            * cfsem.mutual_inductance_of_cylindrical_coils(rzn1, rzn2)
        )  # [H]

        return m  # [H]

    @classmethod
    def from_poly(
        cls,
        parent_name: str,
        polygon: Polygon,
        resistivity: float,
        frac_of_loop: float,
        max_edge_length_m: float = MAX_EDGE_LENGTH_M,
    ) -> PassiveStructureLoop:
        """
        Discretize a structure that is reasonably well-associated with a single centroid
        and may be the result of earlier sub-division of a larger structure, but may not be
        fully discretized yet.
        """
        # Subdivide by meshing
        sub_polygons: list[Polygon] = device_inductance.mesh._mesh_region(
            np.array(polygon.boundary.segmentize(max_edge_length_m).xy).T
        )

        # Make a filament from each mesh cell
        filaments = [
            device_inductance.structures._mesh_elem_to_fil(p, resistivity, parent_name)
            for p in sub_polygons
        ]

        # Call the collection of filaments a loop
        loop = PassiveStructureLoop(
            parent_name=parent_name,
            original_polygon=polygon,
            frac_of_loop=frac_of_loop,
            filaments=filaments,
        )

        return loop

    @classmethod
    def from_input(
        cls,
        parent_name: str,
        polygon: Polygon,
        resistivity: float,
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
        angle_thresh_met = poly_angle(polygon, slicer.centroid) > np.deg2rad(
            angle_thresh_deg
        )
        unroundness_thresh_met = unroundness(polygon) > unroundness_thresh

        if angle_thresh_met and unroundness_thresh_met:
            # Slice into angular chunks
            chunks: list[Polygon] = slicer.slice(polygon)
        else:
            # If the original takes up a small angular region or it's a solid block, use it as-is
            chunks: list[Polygon] = [polygon]

        # Each chunk represents a fraction of one contiguous loop;
        # if we were to treat each chunk as a whole loop, the inductance of the system
        # would diverge with increasing discretization
        frac_of_loop = 1.0 / float(len(chunks))  # [dimensionless]

        return [
            cls.from_poly(parent_name, p, resistivity, frac_of_loop) for p in chunks
        ]
