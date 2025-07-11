"""Grid input definitions"""

GridSpec = tuple[float, int, float, int]
"""[m] rmin, nr, zmin, nz exact specification of regular grid,
as an alternative to approximate minimum extent"""


Extent = tuple[float, float, float, float]
"""[m] rmin, rmax, zmin, zmax rectangular bounds"""


Resolution = tuple[float, float]
"""[m] dr, dz spatial resolution of computational grid"""
