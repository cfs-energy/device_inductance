import numpy as np

import device_inductance

from pytest import approx

from . import typical_outputs  # Required fixture

__all__ = ["typical_outputs"]


def test_circuits(typical_outputs: device_inductance.TypicalOutputs):
    """Circuit mutual inductances and tables are derived from coils,
    which are tested for physical consistency, so these tests
    focus on the differences between the coils and circuits
    instead of revisiting all the coil tests"""

    device = typical_outputs.device
    coils = device.coils
    circuits = device.circuits

    # Make sure all the coils are in a circuit exactly once
    all_coils_in_circuits = sum(
        [[c[0] for c in circuit.coils] for circuit in circuits], start=[]
    )

    for i in range(len(coils)):
        assert (
            i in all_coils_in_circuits
        ), f"Coil {i}, {device.get_coil_names()[i]}, missing from circuits"

    assert (
        len(set(all_coils_in_circuits)) == device.n_coils
    ), "Extra coils or missing coils in circuits"

    # Make sure the total self-inductance of the circuit system is positive
    # Even if every circuit is wound in reverse, the total should still be positive,
    # and we'll never have a system that bootstraps its own current to infinity.
    # That said, not every row or column will have a positive sum, only the whole matrix.
    assert (
        np.sum(device.circuit_mutual_inductances) > 0.0
    ), "Nonphysical circuit mutual inductance matrix"

    assert (
        np.sum(device.circuit_structure_mutual_inductances) > 0.0
    ), "Nonphysical circuit-structure mutual inductance matrix"

    #    Circuit-mode mutual inductances are really part of a larger matrix
    #    and don't necessarily have a positive sum on their own
    circuit_mode_total_inductance = 0.0
    circuit_mode_total_inductance += 2.0 * np.sum(
        device.circuit_structure_mode_mutual_inductances
    )
    circuit_mode_total_inductance += np.sum(device.circuit_mutual_inductances)
    circuit_mode_total_inductance += np.sum(device.structure_mode_mutual_inductances)
    assert (
        circuit_mode_total_inductance > 0.0
    ), "Nonphysical circuit mutual inductance matrix"

    # All circuit self-inductances should be positive for the same reason
    assert np.all(
        np.diag(device.circuit_mutual_inductances) > 0.0
    ), "Bad circuit self-inductance"

    # Circuit system total inductance should be less than or equal to coils,
    # since reversing any coils in a circuit will reduce total inductance
    assert np.sum(device.circuit_mutual_inductances) <= np.sum(
        device.coil_mutual_inductances
    ), "Nonphysical circuit mutual inductance matrix"

    # Circuits' resistances should match the sum of coils, since it's all series segments
    assert np.sum(device.circuit_resistances) == approx(np.sum(
        device.coil_resistances
    ), rel=1e-4), "Circuit resistance does not match sum of coils"

    # Make sure the circuit tables sums match the coils with signs
    psi_circ = device.circuit_flux_tables
    br_circ, bz_circ = device.circuit_flux_density_tables
    psi_c = device.coil_flux_tables
    br_c, bz_c = device.coil_flux_density_tables
    for i, circuit in enumerate(device.circuits):
        circuit_flux_sum = np.sum(psi_circ[i])
        circuit_br_sum = np.sum(br_circ[i])
        circuit_bz_sum = np.sum(bz_circ[i])

        coil_flux_sum = sum([sign * np.sum(psi_c[j]) for j, sign in circuit.coils])
        coil_br_sum = sum([sign * np.sum(br_c[j]) for j, sign in circuit.coils])
        coil_bz_sum = sum([sign * np.sum(bz_c[j]) for j, sign in circuit.coils])

        assert circuit_flux_sum == approx(
            coil_flux_sum, rel=1e-6
        ), "Circuit table does not match coils"
        assert circuit_br_sum == approx(
            coil_br_sum, rel=1e-6
        ), "Circuit table does not match coils"
        assert circuit_bz_sum == approx(
            coil_bz_sum, rel=1e-6
        ), "Circuit table does not match coils"
