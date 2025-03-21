"""
Method for discretizing cross-section outlines of conducting structure.
Mainly copied from Christoph Hasse's work on early device description adapters for MEQ.
"""

import itertools
from enum import Enum, auto

from shapely import Polygon

import gmsh
import numpy as np
from numpy.typing import NDArray


class MeshMode(Enum):
    """Meshing Mode"""

    Quad = auto()
    QuasiStructuredQuad = auto()
    Triangular = auto()


def _mesh_region(
    rz: NDArray,
    min_length: float = 0.1,
    max_length: float = 0.3,
    mesh_mode: MeshMode = MeshMode.QuasiStructuredQuad,
) -> list[Polygon]:
    """
    Generate a structured 2D mesh from an outline.

    Args:
        rz_in: [m] (Nx2) Region outline coords, interleaved
        min_length: [m] target minimum edge length in mesh. Defaults to 0.1.
        max_length: [m] target maximum edge length in mesh. Defaults to 0.3.
        mesh_mode: algorithm for meshing. Defaults to MeshMode.QuasiStructuredQuad.

    Raises:
        ValueError: If an unhandled MeshMode is provided

    Returns:
        Mesh element polygons, with number of vertices depending on method
    """
    # Notes from original implementation:
    # use_qsq enable the quasi structured quadrilaterl meshing
    # produces mesh that has elements more square like but seems to not
    # generate very reproducible meshes. so for testing it's better to disable
    # we need to close polygon in gmsh by reusing the same gmsh point so no need for the last closing point
    if np.all(np.isclose(rz[0], rz[-1])):
        rz = rz[:-1]

    gmsh.initialize()

    pts = [gmsh.model.geo.add_point(p[0], p[1], 0) for p in rz]

    lines = [gmsh.model.geo.add_line(s, e) for (s, e) in itertools.pairwise(pts)]
    # and the closing one
    lines.append(gmsh.model.geo.add_line(pts[-1], pts[0]))
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.Smoothing", 100)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_length)

    # Set up for nearly-deterministic meshing
    gmsh.option.setNumber("Mesh.RandomSeed", 12394871234)
    #  Note this random factor is smaller than recommended, but
    #  larger values result in behavior so non-repeatable that it causes
    #  frequent unit test failures
    gmsh.option.setNumber("Mesh.RandomFactor", 1e-60)
    gmsh.option.setNumber("Mesh.RandomFactor3D", 1e-60)
    gmsh.option.setNumber('General.NumThreads', 1)

    if mesh_mode == MeshMode.Triangular:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
    elif mesh_mode == MeshMode.QuasiStructuredQuad:
        # we don't want to be left with any triangles! (see gmsh tutorial 11)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 11)
        gmsh.option.setNumber("Mesh.QuadqsSizemapMethod", 0)
    elif mesh_mode == MeshMode.Quad:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    else:
        raise ValueError(f"Unknown MeshMode {mesh_mode}")

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)

    n_nodes = 3 if mesh_mode == MeshMode.Triangular else 4

    # elem_node_tags, holds n_nodes (4 if quada or 3 if triangle) unique numerical identifiers
    # that identify the nodes that make up a quadrilateral/triangular mesh element
    # the vector is originally n_nodes * N long
    _, _, elem_node_tags = gmsh.model.mesh.get_elements(2, surf)
    # reshape into N x n_nodes, and substract 1 from the node identifier (see below)
    elem_node_tags = np.asarray(elem_node_tags[0]).reshape((-1, n_nodes)) - 1

    # To get the coordinates for an element we need to look up the coordinates
    # for each node from the nodes storage.

    # node_tags are numerical unique identifiers from 1...N
    # node_coords is a vector of 3N length holding x,y,z for each node
    node_tags, node_coords, _ = gmsh.model.mesh.get_nodes(2, surf, includeBoundary=True)
    node_tags = np.asarray(node_tags)
    # turn it into N x 2 vector
    node_coords = np.asarray(node_coords).reshape(-1, 3)[:, :2]  # only need X,Y (R,Z)
    # we substract 1 because then we can use 0... N-1 as index into the sorted array
    node_tags -= 1  # will be an idx
    node_sorting = np.argsort(node_tags)

    # ensure that we really have indices from 0...N-1
    assert np.all(node_tags[node_sorting] == np.arange(len(node_tags)))
    # now our coordinates are sorted according to that array.
    # So looking up the coordinates of node  id=5 is node_coords[5]
    node_coords = node_coords[node_sorting]

    # this is now an array of shape (N, n_nodes, 2)
    # N elements, defined by n_nodes sets of (R,Z) coordinate pairs
    mesh_element_coords = node_coords[elem_node_tags]

    gmsh.finalize()

    mesh_polygons = [Polygon(x) for x in mesh_element_coords]

    return mesh_polygons
