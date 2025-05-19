"""The lowest-level discretization of conducting structure."""

from dataclasses import dataclass

import numpy as np
from shapely import Polygon, centroid
from cfsem import self_inductance_lyle6


@dataclass(frozen=True)
class PassiveStructureFilament:
    """A chunk of a cylindrically-symmetric passive conductor"""

    parent_name: str
    """Name of structure that this is associated with"""
    r: float
    """[m] radial location of filament center"""
    z: float
    """[m] axial location of filament center"""
    area: float
    """[m^2] cross-section area"""
    resistance: float
    """Ohm] loop resistance"""
    self_inductance: float
    """[H] self-inductance assuming one full loop"""
    polygon: Polygon
    """[m] Outline of mesh element"""


def _mesh_elem_to_fil(
    mesh_elem: Polygon, resistivity: float, parent_name: str
) -> PassiveStructureFilament:
    """Convert mesh element polygon to a resistive filament"""
    # Will this implicitly close? Is the mesh data ordered properly?
    area = mesh_elem.area  # [m^2]

    # Mean r,z of nodes is only really correct centroid for a proper rectangle or rhombus
    # but it should be fairly close for well-behaved shapes
    c = centroid(mesh_elem)  # [m] mean of polygon vertices
    r, z = (c.x, c.y)  # [m]
    toroidal_circumference = 2.0 * np.pi * r  # [m]
    resistance = toroidal_circumference * resistivity / area  # [Ohm]

    # Approximate as square section to estimate self-inductance
    rs, zs = mesh_elem.exterior.xy
    w = max(rs) - min(rs)
    h = max(zs) - min(zs)
    self_inductance = self_inductance_lyle6(r, w, h, n=1.0)

    return PassiveStructureFilament(
        parent_name=parent_name,
        r=r,
        z=z,
        area=area,
        resistance=resistance,
        self_inductance=self_inductance,
        polygon=mesh_elem,
    )
