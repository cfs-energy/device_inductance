import numpy as np
from pytest import approx

import device_inductance
from . import typical_outputs, typical_outputs_many_slices  # Required fixture

__all__ = ["typical_outputs", "typical_outputs_many_slices"]


def test_structure_invariants(
    typical_outputs: device_inductance.TypicalOutputs,
    typical_outputs_many_slices: device_inductance.TypicalOutputs,
):
    """
    Check that the total self-inductance, total resistance, and mode eigenvalues
    are not changing much with changing discretization coarseness.
    """
    d1 = typical_outputs.device
    d2 = typical_outputs_many_slices.device

    # Make sure they both end up with the same total section area
    area1 = np.sum([s.polygon.area for s in d1.structures])
    area2 = np.sum([s.polygon.area for s in d2.structures])

    assert area1 == approx(area2, rel=1e-3), "Section area changed with discretization"

    # Make sure the total stored energy for a given current density on the cross-section
    # is not changing significantly with changing discretization
    i1 = np.array([s.polygon.area for s in d1.structures])  # [A] Reference current for 1 A/m^2 current density
    i2 = np.array([s.polygon.area for s in d2.structures])
    m1 = d1.structure_mutual_inductances  # [H] mutual inductance matrix between structure loops
    m2 = d2.structure_mutual_inductances

    e1 = 0.5 * i1.T @ m1 @ i1  # [J] stored energy for 1A/m^2 on the section
    e2 = 0.5 * i2.T @ m2 @ i2

    assert e1 == approx(e2, rel=1e-3), "Stored energy changed with discretization"

    # Make sure the parallel resistance is not changing significantly with changing discretization
    r1 = 1.0 / np.sum(1.0 / np.diag(d1.structure_resistances))  # [ohm]
    r2 = 1.0 / np.sum(1.0 / np.diag(d2.structure_resistances))  # [ohm]
    assert r1 == approx(r2, rel=1e-3), "Section resistance changed with discretization"

    # Make sure the top few structure model reduction eigenvalues are not changing much...
    eigd1 = d1.structure_mode_eigenvalues[:40]
    eigd1 = np.abs(eigd1 / eigd1[0])  # Normalize
    eigd2 = d2.structure_mode_eigenvalues[:40]
    eigd2 = np.abs(eigd2 / eigd2[0])

    assert np.allclose(eigd1, eigd2, rtol=0.01, atol=0.02), "Eigenvalue profile changed with discretization"
