"""Non-axisymmetric coils (EFCC, etc)"""

from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class Coil3D:
    path: tuple[NDArray, NDArray, NDArray]
    self_inductance: float
    resistance: float
    turns: float
