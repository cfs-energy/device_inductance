import numpy as np
import pytest

from device_inductance.model_reduction import eigenmode_reduction


@pytest.mark.parametrize("max_neig", [None, 0, 1, 2, 3, 100])
def test_eigenmode_reduction_preserves_ordered_eigenpairs(max_neig):
    """Retain the longest L/R timescales with their matching current modes."""
    # The coupled pair has timescales 3 s and 1 s; the isolated loop has 5 s.
    m = np.array([[2.0, 2.0, 0.0], [2.0, 8.0, 0.0], [0.0, 0.0, 10.0]])  # [H]
    r = np.diag([1.0, 4.0, 2.0])  # [ohm]
    expected = np.array([-5.0, -3.0, -1.0])[:max_neig]  # [s]

    d, tuv, neig = eigenmode_reduction(m, r, max_neig)

    assert neig == len(expected)
    assert tuv.shape == (3, neig)
    assert np.isrealobj(d) and np.isrealobj(tuv)
    np.testing.assert_allclose(d, expected, rtol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(tuv, axis=0), 1.0, rtol=1e-13)
    np.testing.assert_allclose(-m @ tuv, (r @ tuv) * d, rtol=1e-13, atol=1e-14)

    if neig == 3:
        # A complete current basis must preserve the response to applied voltage.
        currents = np.array([0.25, -0.5, 0.75])  # [A]
        voltages = np.array([1.0, 2.0, -1.0])  # [V]
        forcing = voltages - r @ currents
        full_rate = np.linalg.solve(m, forcing)  # [A/s]
        mode_rate = np.linalg.solve(tuv.T @ m @ tuv, tuv.T @ forcing)
        np.testing.assert_allclose(tuv @ mode_rate, full_rate, rtol=1e-13, atol=1e-14)


def test_eigenmode_reduction_roundoff_at_repeated_timescale():
    """Roundoff in reciprocal mutuals must not create complex current modes."""
    # Two uncoupled loops have the same 2 s timescale. Opposite rounding errors
    # in their nominally zero mutual terms split a general eigensolve into a
    # complex pair, even though the reciprocal physical system has real modes.
    m = np.array([[2.0, 1e-16], [-1e-16, 8.0]])  # [H]
    r = np.diag([1.0, 4.0])  # [ohm]

    d, tuv, neig = eigenmode_reduction(m, r, None)

    assert neig == 2
    assert np.isrealobj(d) and np.isrealobj(tuv)
    assert np.linalg.matrix_rank(tuv) == 2
    np.testing.assert_allclose(d, [-2.0, -2.0], rtol=1e-13)
    np.testing.assert_allclose(-m @ tuv, (r @ tuv) * d, rtol=1e-13, atol=1e-14)
    modal_r = tuv.T @ r @ tuv
    np.testing.assert_allclose(modal_r, np.diag(np.diag(modal_r)), atol=1e-14)


def test_eigenmode_reduction_rejects_nonreciprocal_inductance():
    m = np.array([[2.0, 0.5], [1.0, 2.0]])
    with pytest.raises(ValueError, match="symmetric"):
        eigenmode_reduction(m, np.eye(2), None)


@pytest.mark.parametrize("resistance", [0.0, -1.0])
def test_eigenmode_reduction_requires_positive_resistance(resistance):
    with pytest.raises(np.linalg.LinAlgError):
        eigenmode_reduction(np.eye(2), np.diag([1.0, resistance]), None)
