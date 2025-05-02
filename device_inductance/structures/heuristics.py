import numpy as np
from numpy.typing import NDArray

from shapely import Polygon

MAX_EDGE_LENGTH_M = 0.1
"""Default maximum length of an edge in the lowest-level discretization"""


def winding_number(
    path: NDArray, centroid: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculate winding number about x, y, and z axes, w.r.t. the filament centroid.
    This estimates the angle subtended by each segment of the coil path, which is
    2*pi radians per full revolution, and may be either positive or negative depending
    on the orientation of the coil winding path.

    The choice of sign convention is arbitrary, as these results exist to compare to each other,
    and for no other purpose.
    """

    # path = xyz  # [m]
    n = path.shape[1]

    # centroid = np.mean(path, axis=1)  # [m]
    x_winding = np.zeros(n)  # [rad]
    y_winding = np.zeros(n)
    z_winding = np.zeros(n)

    for i in range(n - 1):
        # Translate the target points to center on the centroid
        first_point = path[:, i] - centroid  # [m]
        second_point = path[:, i + 1] - centroid

        # Extract angle subtended by the two points about each axis
        cross_1_2 = np.cross(first_point, second_point)
        dot_1_2 = np.dot(first_point, second_point)
        x_winding[i + 1] = x_winding[i] + np.atan2(cross_1_2[0], dot_1_2)
        y_winding[i + 1] = y_winding[i] + np.atan2(cross_1_2[1], dot_1_2)
        z_winding[i + 1] = z_winding[i] + np.atan2(cross_1_2[2], dot_1_2)

    return x_winding, y_winding, z_winding  # [rad]


def unroundness(p: Polygon) -> float:
    """
    [dimensionless] ratio of perimeter to perimeter of a circle with the same area.
    Gives a heuristic estimate for how un-like a circle the shape is.
    """
    perimeter = p.boundary.length  # [m]
    circle_perimeter = 2.0 * np.pi * np.sqrt(p.area / np.pi)  # [m]
    return perimeter / circle_perimeter  # [dimensionless]


def poly_angle(
    p: Polygon,
    centroid: tuple[float, float],
    max_edge_length_m: float = MAX_EDGE_LENGTH_M,
) -> float:
    """
    Get peak winding number in [rad] of the polygon boundary w.r.t. the centroid.
    This is not necessarily the same as the maximum angle subtended by the polygon,
    but is a good heuristic for it, and runs in O(N) instead of O(N^2).
    """
    # Expand to 3D and segmentize to avoid missing big sections
    pr, pz = p.boundary.segmentize(max_edge_length_m).xy
    xyz = np.array([pr, np.zeros_like(pr), pz])
    centroid_arr = np.array([centroid[0], 0.0, centroid[1]])

    # Get aggregate winding number w.r.t. centroid along polygon boundary
    _wx, wy, _wz = winding_number(xyz, centroid_arr)

    return np.max(np.abs(wy))  # [rad]
