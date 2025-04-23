from dataclasses import dataclass
from omas import ODS
from shapely import Polygon


@dataclass(frozen=True)
class PassiveStructureInput:
    parent_name: str
    polygon: Polygon  # [m]
    resistivity: float  # [ohm]


def _collect_structures(description: ODS) -> list[PassiveStructureInput]:
    """
    Combine all passive structure input geometries and resistivities into the same format,
    a polygon with a resistivity and a name.
    """

    # Collect structure polygons and resistivities
    structure_inputs: tuple[str, Polygon, float] = []

    #  Wall
    items = description["wall.description_2d.0.vessel"]["unit"].values()
    for wall_section in items:
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            rs = wall_elem["outline.r"]
            zs = wall_elem["outline.z"]
            resistivity = wall_elem["resistivity"]  # [Ohm-m]

            structure_inputs.append(
                PassiveStructureInput(
                    wall_elem["name"], Polygon([x for x in zip(rs, zs)]), resistivity
                )
            )

    #  May or may not be wall, depending on how the wall is defined
    items = description["pf_passive.loop"].values()
    for passive_elem in items:
        # Unpack
        rs = passive_elem["element.0.geometry.outline.r"]
        zs = passive_elem["element.0.geometry.outline.z"]
        resistivity = passive_elem["resistivity"]

        structure_inputs.append(
            PassiveStructureInput(
                passive_elem["name"], Polygon([x for x in zip(rs, zs)]), resistivity
            )
        )

    return structure_inputs
