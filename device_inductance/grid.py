from typing import NamedTuple


class GridSpec(NamedTuple):
    """Exact specification of regular grid,
    as an alternative to approximate minimum extent"""

    r0: float
    """[m] start of r-grid"""
    nr: int
    """Number of points in r-grid"""
    z0: float
    """[m] Start of z-grid"""
    nz: int
    """Number of points in z-grid"""


Extent = tuple[float, float, float, float]
"""[m] rmin, rmax, zmin, zmax rectangular bounds"""


Resolution = tuple[float, float]
"""[m] spatial resolution of computational grid"""