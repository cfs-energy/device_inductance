import numpy as np

import device_inductance

from pytest import approx

from . import typical_outputs  # Required fixture

__all__ = ["typical_outputs"]

TESTFUNC = lambda r: 1.0 / (2.0 * np.pi * r)  # noqa: E731
"""Arbitrary function for dummy field"""


def test_full_flux_loops(typical_outputs: device_inductance.TypicalOutputs):
    """Make sure the full flux loops sample the input field at a point"""
    grids = typical_outputs.grids
    rmesh, _ = typical_outputs.meshes

    testmesh = TESTFUNC(rmesh)

    assert len(typical_outputs.device.full_flux_loops) > 0
    
    for ffloop in typical_outputs.device.full_flux_loops:
        assert ffloop.response(grids, testmesh) == approx(TESTFUNC(ffloop.r), rel=1e-3)


def test_partial_flux_loops(typical_outputs: device_inductance.TypicalOutputs):
    """Make sure the partial flux loops return the integral of B*dA on their enclosed surface"""
    grids = typical_outputs.grids
    rmesh, _ = typical_outputs.meshes

    testmesh = TESTFUNC(rmesh)

    assert len(typical_outputs.device.partial_flux_loops) > 0

    unit_vector = np.array(
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]
    )  # direction of test field
    mag_rel = np.sqrt(2.0)  # total field magnitude relative to 1.0 / (2.0 * pi * r)
    for pfloop in typical_outputs.device.partial_flux_loops:
        # Use 1/r tendency of TESTFUNC to make flux integral easy
        dr = pfloop.r1 - pfloop.r0
        dz = pfloop.z1 - pfloop.z0
        dvec = np.array([dz, -dr])  # Perpendicular normal vector

        expected = np.dot(unit_vector, dvec) * pfloop.loop_frac * mag_rel

        response = pfloop.response(grids, testmesh, testmesh)

        assert response == approx(expected, rel=1e-3)


def test_poloidal_field_probes(typical_outputs: device_inductance.TypicalOutputs):
    """Make sure the pickup coils return the sum of projected components of the B-field on their axis"""
    grids = typical_outputs.grids
    rmesh, zmesh = typical_outputs.meshes

    assert len(typical_outputs.device.poloidal_field_probes) > 0

    # Field varying on R
    testmesh = TESTFUNC(rmesh)
    for bpcoil in typical_outputs.device.poloidal_field_probes:
        expected = TESTFUNC(bpcoil.r) * bpcoil.unit_vector[0]
        response = bpcoil.response(grids, testmesh, np.zeros_like(testmesh))
        assert response == approx(expected, rel=1e-3)

    # Field varying on Z
    testmesh = zmesh  # Avoid div/0 with testfunc
    for bpcoil in typical_outputs.device.poloidal_field_probes:
        expected = bpcoil.z * bpcoil.unit_vector[1]
        response = bpcoil.response(grids, np.zeros_like(testmesh), testmesh)
        assert response == approx(expected, rel=1e-3)

    # Field varying on R and Z
    for bpcoil in typical_outputs.device.poloidal_field_probes:
        expected = (
            bpcoil.z * bpcoil.unit_vector[1]
            + TESTFUNC(bpcoil.r) * bpcoil.unit_vector[0]
        )
        response = bpcoil.response(grids, TESTFUNC(rmesh), zmesh)
        assert response == approx(expected, rel=1e-3)
