import numpy as np

import device_inductance

from pytest import approx

from cfsem import body_force_density_circular_filament_cartesian
from . import typical_outputs, typical_outputs_stabilized_eigenmode  # Required fixture

__all__ = ["typical_outputs", "typical_outputs_stabilized_eigenmode"]


def test_coil_coil_forces(typical_outputs: device_inductance.TypicalOutputs):
    """Spot-check an entry in the coil force matrix by capitalizing on the equivalence of
    integrating I*dL and J*dV"""

    device = typical_outputs.device
    coils = device.coils

    for i in range(len(coils)):
        for j in range(len(coils)):
            if i == j:
                continue

            coila = coils[i]
            coilb = coils[j]

            fab_mat_r, fab_mat_z = device.coil_coil_forces[0][i, j], device.coil_coil_forces[1][i, j]  # [N/A^2]
            
            ra, za, na = coila.rs, coila.zs, coila.ns
            rb, zb, nb = coilb.rs, coilb.zs, coilb.ns
            zero = np.zeros_like(rb)
            # Replacing J with I*dL gives body force instead of body force density
            fab_jxb_r, fab_jxb_y, fab_jxb_z  = body_force_density_circular_filament_cartesian(na, ra, za, obs=(rb, zero, zb), j=(zero, 2.0 * np.pi * rb * nb, zero))  # [N/A^2]

            assert sum(fab_jxb_y) == 0.0
            # We see a combination of interp error here, as well as some filamentization error
            # for coil pairs that are very close together, because the BFD calc uses the filament B-field
            # which suffers a bit in the near-field compared to the smooth self-field patch used in the
            # tables
            assert fab_mat_r == approx(sum(fab_jxb_r), rel=2e-2, abs=3e-6)
            assert fab_mat_z == approx(sum(fab_jxb_z), rel=2e-2, abs=3e-6)
