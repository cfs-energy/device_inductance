from itertools import chain

from omas import ODS
from shapely import Polygon

from device_inductance.utils import _progressbar

from .input import _collect_structures
from .slicer import RadialSlicer
from .loop import PassiveStructureLoop


def _extract_structures(
    description: ODS,
    limiter: Polygon,
    n_slices: int,
    extent: tuple[float, float, float, float],
    show_prog: bool = True,
) -> list[PassiveStructureLoop]:
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
    # Collect wall and pf_passive structures into same format
    structure_inputs = _collect_structures(description)

    # Make a slicer that will chunk inputs radially if needed (usually just for the VV),
    # prioritizing protection of the iteraction between the structures and the plasma centroid.
    slicer = RadialSlicer(n_slices, extent, (limiter.centroid.x, limiter.centroid.y))

    # Chunk, mesh, and process each input.
    # The slow part here is running gmsh on each chunk.
    structure_chunks: list[list[PassiveStructureLoop]] = []
    items = structure_inputs
    items = _progressbar(items, suffix="Structures discretized") if show_prog else items
    for inp in items:
        structure_chunks.append(PassiveStructureLoop.from_input(inp, slicer=slicer))

    # Flatten
    structure_loops: list[PassiveStructureLoop] = list(chain(*structure_chunks))

    return structure_loops
