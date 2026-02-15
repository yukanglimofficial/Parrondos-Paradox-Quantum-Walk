import numpy as np

from dtqw.noise import kraus_pd, apply_kraus_to_density


def test_pd_channel_scales_off_diagonals_and_keeps_diagonals():
    # Choose a valid density matrix with nonzero coherence
    p = 0.30
    rho = np.array(
        [
            [0.25, 0.20 + 0.10j],
            [0.20 - 0.10j, 0.75],
        ],
        dtype=np.complex128,
    )

    rho2 = apply_kraus_to_density(rho, kraus_pd(p))

    # Diagonals unchanged
    np.testing.assert_allclose(rho2[0, 0], rho[0, 0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(rho2[1, 1], rho[1, 1], atol=1e-12, rtol=0.0)

    # Off-diagonals scaled by (1 - p)
    np.testing.assert_allclose(rho2[0, 1], (1.0 - p) * rho[0, 1], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(rho2[1, 0], (1.0 - p) * rho[1, 0], atol=1e-12, rtol=0.0)

    # Trace preserved
    np.testing.assert_allclose(np.trace(rho2), np.trace(rho), atol=1e-12, rtol=0.0)
