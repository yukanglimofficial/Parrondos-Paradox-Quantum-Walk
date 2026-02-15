import numpy as np

from dtqw.simulate import run_sequence_unitary


def test_unitary_probability_conserved_to_1e12():
    T = 50
    phi = 0.0
    coin_params = {
        "A": (0.0, 45.0, 0.0),  # balanced rotation coin (degrees)
    }

    out = run_sequence_unitary(sequence="A", T=T, phi=phi, coin_params=coin_params)
    psi_t = out["psi_t"]

    for t in range(T + 1):
        total_prob = float(np.sum(np.abs(psi_t[t]) ** 2))
        np.testing.assert_allclose(total_prob, 1.0, atol=1e-12, rtol=0.0)
