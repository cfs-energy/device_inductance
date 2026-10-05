import numpy as np


def test_cfsem14_force_pose_mapping(monkeypatch):
    """Check source centers and orientations for unequal source and target sizes.

    Signed, fractional turn counts exercise current weighting in both force
    paths. Compare the resulting forces with the axisymmetric field calculation
    to check that the explicit source geometry preserves the same result.
    """
    from types import SimpleNamespace

    from cfsem import flux_density_circular_filament
    from device_inductance import forces
    from device_inductance.coils import Coil, CoilFilament

    def coil(name, rows):
        result = Coil(name, 0.0, 0.0, [CoilFilament(r, z, n, 0.0) for r, z, n in rows])
        # This test exercises mutual forces, without a self-field table solve.
        result.__dict__["local_fields"] = None
        return result

    source = coil("source", [(1.1, -0.3, 0.75), (1.3, 0.2, -0.25)])
    target = coil("target", [(1.8, -0.4, -0.4), (1.9, 0.1, 1.2), (2.0, 0.5, 0.6)])
    backend = forces.body_force_density_circular_filament_cartesian
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return backend(**kwargs)

    monkeypatch.setattr(forces, "body_force_density_circular_filament_cartesian", capture)
    grids = (np.linspace(0.5, 2.5, 8), np.linspace(-1.0, 1.0, 9))
    tables = (np.zeros((2, 8, 9)), np.zeros((2, 8, 9)))
    fr, fz = forces._calc_coil_coil_forces([source, target], grids, tables, show_prog=False)
    structure = SimpleNamespace(rs=source.rs, zs=source.zs, ns=source.ns)
    sfr, sfz = forces._calc_structure_coil_forces([target], [structure], show_prog=False)

    for call, src, obs in zip(calls, (source, target, source), (target, source, target), strict=True):
        np.testing.assert_array_equal(call["ifil"], src.ns)
        np.testing.assert_array_equal(call["rfil"], src.rs)
        np.testing.assert_array_equal(call["loc"], (np.zeros(len(src.rs)), np.zeros(len(src.rs)), src.zs))
        np.testing.assert_array_equal(
            call["normal"], (np.zeros(len(src.rs)), np.zeros(len(src.rs)), np.ones(len(src.rs)))
        )
        np.testing.assert_array_equal(call["obs"], (obs.rs, np.zeros(len(obs.rs)), obs.zs))
        np.testing.assert_array_equal(
            call["j"], (np.zeros(len(obs.rs)), 2 * np.pi * obs.rs * obs.ns, np.zeros(len(obs.rs)))
        )
        assert call["wire_radius"] is None
    assert calls[-1]["par"] is False

    br, bz = flux_density_circular_filament(source.ns, source.rs, source.zs, target.rs, target.zs)
    lengths = 2 * np.pi * target.rs * target.ns  # [m] signed azimuthal length per terminal ampere
    # Both public CFSEM paths describe the same axisymmetric source. Allow only rounding.
    np.testing.assert_allclose(
        [fr[0, 1], fz[0, 1]], [sum(lengths * bz), sum(-lengths * br)], rtol=1e-12, atol=1e-18
    )
    np.testing.assert_allclose([sfr[0, 0], sfz[0, 0]], [fr[0, 1], fz[0, 1]], rtol=1e-12, atol=1e-18)
