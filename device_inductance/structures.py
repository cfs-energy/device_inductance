from sys import intern
from dataclasses import dataclass

import numpy as np

from shapely import Polygon, centroid
from omas import ODS

from cfsem import self_inductance_lyle6
from device_inductance.mesh import _mesh_region
from device_inductance.utils import _progressbar
from device_inductance.logging import log


@dataclass(frozen=True)
class PassiveStructureFilament:
    """A chunk of a cylindrically-symmetric passive conductor"""
    parent_name: str  # Name of structure that this is associated with
    r: float  # [m]
    z: float  # [m]
    area: float  # [m^2] cross-section area
    resistance: float  # [Ohm] loop resistance
    self_inductance: float  # [H]
    polygon: Polygon  # Outline of mesh element


def _extract_structures(
    description: ODS, show_prog: bool = True
) -> list[PassiveStructureFilament]:
    """
    Extract and filamentize passive conducting structural elements
    by first meshing the 2D cross-sections to quads, then collapsing
    each quad to a singularity filament at the mean position of its vertices.

    Resistance and self-inductance are estimated by approximating each quad as
    a square of equivalent area, which may give significant error if quads of
    high aspect ratio or nonconvex shape are encountered. Because the meshing
    algorithm nominally guarantees convex elements and maximum element size is
    limited in order to keep spatial resolution, there is some hope that this
    approximation will not be too far off.

    Args:
        description: Device geometric info in the format produced by device_description
        show_prog: Display a terminal progressbar

    Returns:
        A list of passive filaments, populated with reduced geometric info and estimated self-inductances.
    """
    passive_filaments: list[PassiveStructureFilament] = []

    # Use block-representation to capture different effective resistivity in different regions
    # Inner wall, outer wall, ??
    items = description["wall.description_2d.0.vessel"]["unit"].items()
    if show_prog:
        items = _progressbar(items, "Passive structure meshes")
    for partial_name, wall_section in items:
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            # Unpack
            rs = wall_elem["outline.r"]
            zs = wall_elem["outline.z"]
            resistivity = wall_elem["resistivity"]  # [Ohm-m]

            # Mesh outline and approximate as filaments
            mesh = _mesh_region(np.array([x for x in zip(rs, zs)]))
            parent_name = intern(f"wall.description_2d.0.vessel.unit.{partial_name}")
            for mesh_elem in mesh:
                fil = _mesh_elem_to_fil(mesh_elem, resistivity, parent_name)
                if fil.resistance > 0.0:
                    passive_filaments.append(fil)
                else:
                    print("Skipped zero-resistance passive filament")

    # PF-related passive structures are stored separately.
    # These are vertical stability coil covers, stabilizer bars, etc
    items = description["pf_passive.loop"].items()
    if show_prog:
        items = _progressbar(items, "Passive coil meshes")
    for partial_name, passive_elem in items:
        # Unpack
        rs = passive_elem["element.0.geometry.outline.r"]
        zs = passive_elem["element.0.geometry.outline.z"]
        resistivity = passive_elem["resistivity"]

        # Mesh outline and approximate as filaments
        mesh = _mesh_region(np.array([x for x in zip(rs, zs)]))
        parent_name = intern(f"pf_passive.loop.{partial_name}")
        for mesh_elem in mesh:
            fil = _mesh_elem_to_fil(mesh_elem, resistivity, parent_name)
            if fil.resistance > 0.0:
                passive_filaments.append(fil)
            else:
                log().warning("Skipped zero-resistance passive filament")

    # Sort filaments to improve conditioning of model reduction
    passive_filaments = sorted(passive_filaments, key=lambda x: -x.self_inductance / x.resistance)

    return passive_filaments


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
    # TODO: think more about how to handle different geometries
    # when we're ready to split hairs
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
