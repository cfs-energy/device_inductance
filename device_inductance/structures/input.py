from dataclasses import dataclass
from omas import ODS
from shapely import Polygon

CONTACT_DETECTION_DISTANCE = 1e-4
"""[m] distance under which two passive structure objects are considered to be in electrical contact"""


@dataclass(frozen=True)
class PassiveStructureInput:
    """First level of structure discretization within device_inductance.
    Some discretization may have already been performed upstream."""

    parent_name: str
    """The name of the source of the info from the ODS input"""
    polygon: Polygon
    """[m] R-Z cross-section"""
    resistivity: float
    """[ohm-m] (effective) electrical resistivity of material."""


def _collect_structures(description: ODS) -> list[PassiveStructureInput]:
    """
    Combine all passive structure input geometries and resistivities into the same format,
    a polygon with a resistivity and a name.
    """

    # Collect structure polygons and resistivities
    structure_inputs: list[PassiveStructureInput] = []

    # Wall
    items = description["wall.description_2d.0.vessel"]["unit"].values()
    n_wall = 0  # Number of wall inputs
    for wall_section in items:
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            n_wall += 1
            name = wall_elem["name"]
            resistivity = wall_elem["resistivity"]  # [ohm-m]
            rs = wall_elem["outline.r"]  # [m]
            zs = wall_elem["outline.z"]  # [m]
            polygon = Polygon([x for x in zip(rs, zs)])

            structure_inputs.append(PassiveStructureInput(name, polygon, resistivity))

    # pf_passive
    # These entries may or may not be wall, depending on how the wall is defined.
    for i, passive_elem in enumerate(description["pf_passive.loop"].values()):
        name = f"{passive_elem['name']}_{i}"  # Original names may not be unique
        rs = passive_elem["element.0.geometry.outline.r"]
        zs = passive_elem["element.0.geometry.outline.z"]
        resistivity = passive_elem["resistivity"]
        polygon = Polygon([x for x in zip(rs, zs)])

        structure_inputs.append(PassiveStructureInput(name, polygon, resistivity))

    # Make sure the input polygons had a valid point ordering and aren't folded up
    for inp in structure_inputs:
        assert inp.polygon.is_valid

    # Make sure all inputs are accounted
    n_pf_passive = len(description["pf_passive.loop"].values())
    assert len(structure_inputs) == n_pf_passive + n_wall, (
        "Failed to account for some structure inputs, possibly due to duplicate names of wall sections"
    )

    return structure_inputs
