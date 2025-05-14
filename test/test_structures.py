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

    # Make sure the total self inductance is not changing significantly with changing discretization
    l1 = np.sum(d1.structure_mutual_inductances)  # [H]
    l2 = np.sum(d2.structure_mutual_inductances)  # [H]
    assert l1 == approx(l2, rel=1e-3)

    # Make sure the parallel resistance is not changing significantly with changing discretization
    r1 = 1.0 / np.sum(1.0 / np.diag(d1.structure_resistances))  # [ohm]
    r2 = 1.0 / np.sum(1.0 / np.diag(d2.structure_resistances))  # [ohm]
    assert r1 == approx(r2, rel=1e-3)

    # Make sure the top few structure model reduction eigenvalues are not changing much...
    eigd1 = d1.structure_mode_eigenvalues
    eigd1 = np.abs(eigd1 / eigd1[0])  # Normalize
    eigd2 = d2.structure_mode_eigenvalues
    eigd2 = np.abs(eigd2 / eigd2[0])

    assert np.allclose(eigd1, eigd2, rtol=0.1, atol=0.02)
