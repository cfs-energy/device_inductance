from dataclasses import dataclass
from omas import ODS
from shapely import Polygon


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
    frac_of_loop: float
    """
    [dimensionless] What fraction of a whole structure this represents; in the case of
    wall sections that are discretized into multiple elements, this represents each element's
    fraction of the section's total cross-sectional area.
    """


def _collect_structures(description: ODS) -> list[PassiveStructureInput]:
    """
    Combine all passive structure input geometries and resistivities into the same format,
    a polygon with a resistivity and a name.
    """

    # Collect structure polygons and resistivities
    structure_inputs: list[PassiveStructureInput] = []

    #  Wall
    items = description["wall.description_2d.0.vessel"]["unit"].values()
    for wall_section in items:
        section_polygons = {}
        section_area = 0.0  # [m^2]

        # First pass: extract polygons and find total area of this wall section
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            name = wall_elem["name"]
            rs = wall_elem["outline.r"]  # [m]
            zs = wall_elem["outline.z"]  # [m]
            polygon = Polygon([x for x in zip(rs, zs)])
            section_area += polygon.area  # [m^2]
            section_polygons[name] = polygon  # [m]

        # Second pass: calculate frac_of_loop for each element & finalize element inputs
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            name = wall_elem["name"]
            resistivity = wall_elem["resistivity"]  # [Ohm-m]
            polygon = section_polygons[name]  # [m]

            # This method prevents the total inductance of a wall section from diverging
            # with changing upstream discretization.
            #
            # Ideally, we'd have a mechanism to account for section elements that have different
            # resistivity, but there isn't one right choice of representation of that
            # inhomogeneous system as a single homogeneous system; instead, we use the simplest
            # and most predictable method, which may produce some error in extreme cases
            # (for example, a wall section with a large element with infinite resistivity
            # alongside a small element with small resistivity). Actual geometries are expected
            # to stay far from such extremes.
            frac_of_loop = polygon.area / section_area  # [dimensionless]

            structure_inputs.append(
                PassiveStructureInput(name, polygon, resistivity, frac_of_loop)
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
                passive_elem["name"],
                Polygon([x for x in zip(rs, zs)]),
                resistivity,
                frac_of_loop=1.0,
            )
        )

    # Make sure the input polygons had a valid point ordering and aren't folded up
    for inp in structure_inputs:
        assert inp.polygon.is_valid

    return structure_inputs
