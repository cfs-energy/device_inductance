"""Tool for cutting a polyogon into pie-slice chunks"""

from itertools import product

import numpy as np
from shapely import Polygon, GeometryCollection, MultiPolygon


class RadialSlicer:
    """Tool for cutting a polyogon into pie-slice chunks"""

    centroid: tuple[float, float]
    """Centroid used to generate the slices"""

    polygons: list[Polygon]
    """Contiguous but non-overlapping slices spanning the provided extent"""

    def __init__(
        self,
        n_slices: int,
        extent: tuple[float, float, float, float],
        centroid: tuple[float, float],
    ):
        # Unpack
        self.centroid = centroid
        rmid, zmid = centroid
        centroid_arr = np.array(centroid)

        # Choose a pie radius
        #   Actualize all the points at the corners of the extent
        extent_points = [
            np.array(x) for x in product([extent[0], extent[1]], [extent[2], extent[3]])
        ]
        #   Find the largest radius from the centroid to any of the points
        extent_radii = np.linalg.norm(
            np.array([x - centroid_arr for x in extent_points]), axis=1
        )
        pie_radius = 1.05 * np.max(extent_radii)

        pie_angles = np.linspace(0.0, 2.0 * np.pi, n_slices + 1)
        pie_r = rmid + pie_radius * np.cos(pie_angles)
        pie_z = zmid + pie_radius * np.sin(pie_angles)
        pie_rz = [x for x in zip(pie_r, pie_z)]
        mids = [(rmid, zmid)] * n_slices
        #  Sections start at the midpoint, go to two points on the circle, then end back at the midpoint
        pie_sections = zip(mids, pie_rz[:-1], pie_rz[1:], mids)
        pie_slices = [Polygon(s) for s in pie_sections]

        self.polygons = pie_slices

    def slice(self, p: Polygon) -> list[Polygon]:
        """Cut polygon `p` into as many as `n_slices` new polygons or as few as 1 polygon, if no cuts are needed."""

        # Do the intersections - this can balloon into a jumble of outputs
        intersections = [p.intersection(s) for s in self.polygons]

        # Remove edge and vertex intersections
        new_polygons = [g for g in intersections if isinstance(g, Polygon)]

        for g in intersections:
            # It's possible to have one polygon intersect more than once,
            # at an edge, at a point, etc. and we have to handle all cases.
            if isinstance(g, GeometryCollection) or isinstance(g, MultiPolygon):
                new_polygons.extend([x for x in g.geoms if isinstance(x, Polygon)])

        # Remove any empty polygons; these are common
        new_polygons = [x for x in new_polygons if not x.is_empty]

        # Make sure the area adds up
        area_orig = p.area
        area_new = sum([x.area for x in new_polygons])
        assert abs(1.0 - abs(area_new / area_orig)) < 1e-3

        return new_polygons
