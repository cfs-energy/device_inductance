from dataclasses import dataclass
from omas import ODS
from shapely import Polygon
import networkx as nx

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

    # Wall
    items = description["wall.description_2d.0.vessel"]["unit"].values()
    n_wall = 0  # Number of wall inputs
    for wall_section in items:
        section_polygons = {}
        section_area = 0.0  # [m^2]

        # First pass: extract polygons and find total area of this wall section
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            n_wall += 1
            name = wall_elem["name"]
            assert name not in section_polygons.keys(), (
                f"Duplicate wall section name detected: {name}"
            )
            rs = wall_elem["outline.r"]  # [m]
            zs = wall_elem["outline.z"]  # [m]
            polygon = Polygon([x for x in zip(rs, zs)])
            section_area += polygon.area  # [m^2]
            section_polygons[name] = polygon  # [m]

        # Second pass: calculate frac_of_loop for each element & finalize element inputs
        for wall_elem in wall_section["element"].values():  # Segments of each wall
            name = wall_elem["name"]
            resistivity = wall_elem["resistivity"]  # [ohm-m]
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

    # pf_passive
    # These entries may or may not be wall, depending on how the wall is defined.
    #
    # Some of these may contact each other an must be treated as a single turn
    # if their section is electrically continuous.

    # First pass: generate polygons
    elem_polygons: dict[str, Polygon] = {}
    for i, passive_elem in enumerate(description["pf_passive.loop"].values()):
        name = f"{passive_elem['name']}_{i}"
        rs = passive_elem["element.0.geometry.outline.r"]
        zs = passive_elem["element.0.geometry.outline.z"]
        polygon = Polygon([x for x in zip(rs, zs)])
        elem_polygons[name] = polygon

    # Second pass: group elements by contact
    # Develop full N^2 adjacency graph that may have cycles
    g = nx.Graph()
    for name in elem_polygons.keys():
        g.add_node(name)

    for name1, polygon1 in elem_polygons.items():
        for name2, polygon2 in elem_polygons.items():
            if polygon1.distance(polygon2) < CONTACT_DETECTION_DISTANCE:
                # `intersects` does not work well here - fails to detect some valid contacts
                g.add_edge(name1, name2)

    # Extract connected subgraphs (groups of elements that are in electrical contact)
    components: list[nx.Graph] = [
        g.subgraph(c).copy() for c in nx.connected_components(g)
    ]
    groups = [[*c.nodes] for c in components]

    # Reverse map from element names to group and accumulate section
    # area of each group
    group_assignments = {}
    group_areas = []
    for i, group in enumerate(groups):
        group_area = 0.0  # [m^2]
        for name in group:
            group_assignments[name] = i
            group_area += elem_polygons[name].area
        group_areas.append(group_area)

    # Second pass: assign frac_of_loop based on connected groups
    for i, passive_elem in enumerate(description["pf_passive.loop"].values()):
        name = f"{passive_elem['name']}_{i}"
        rs = passive_elem["element.0.geometry.outline.r"]
        zs = passive_elem["element.0.geometry.outline.z"]
        resistivity = passive_elem["resistivity"]
        polygon = elem_polygons[name]
        group_index = group_assignments[name]
        group_area = group_areas[group_index]

        structure_inputs.append(
            PassiveStructureInput(
                passive_elem["name"],
                Polygon([x for x in zip(rs, zs)]),
                resistivity,
                frac_of_loop=polygon.area / group_area,
            )
        )

    # Make sure the input polygons had a valid point ordering and aren't folded up
    for inp in structure_inputs:
        assert inp.polygon.is_valid

    # Make sure all inputs are accounted
    n_pf_passive = len(description["pf_passive.loop"].values())
    assert len(structure_inputs) == n_pf_passive + n_wall, (
        "Failed to account for some structure inputs, possibly due to duplicate names of wall sections"
    )

    return structure_inputs
