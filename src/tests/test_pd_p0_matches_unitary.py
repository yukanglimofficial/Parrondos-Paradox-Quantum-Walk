import numpy as np

from dtqw.metrics import prob_from_psi_t
from dtqw.noise import run_sequence_pd_trajectories
from dtqw.simulate import run_sequence_unitary


def test_pd_p0_matches_unitary_distribution():
    T = 20
    phi = 0.7
    coin_params = {"A": (0.0, 45.0, 0.0), "B": (0.0, 45.0, 0.0)}
    seq = "ABB"

    # Unitary baseline
    out_u = run_sequence_unitary(sequence=seq, T=T, phi=phi, coin_params=coin_params)
    P_u = prob_from_psi_t(out_u["psi_t"])

    # PD trajectories with p=0 should be identical (K0=I, others zero)
    out_pd = run_sequence_pd_trajectories(
        sequence=seq,
        T=T,
        phi=phi,
        p=0.0,
        coin_params=coin_params,
        N_traj=5,   # doesn't matter; all trajectories identical when p=0
        seed=123,
    )
    P_pd = out_pd["P_t"]

    np.testing.assert_allclose(P_pd, P_u, atol=1e-12, rtol=0.0)
