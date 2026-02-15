from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from dtqw.coins import su2_coin, apply_coin
from dtqw.defect import apply_origin_phase_defect
from dtqw.shift import apply_conditional_shift
from dtqw.simulate import default_init_state


def kraus_pd(p: float) -> List[NDArray[np.complex128]]:
    """
    Phase damping / dephasing channel (coin-only) with PROJECT p definition:

      rho_01 -> (1 - p) * rho_01
      rho_00, rho_11 unchanged

    A Kraus set that implements this is:
      K0 = sqrt(1-p) * I
      K1 = sqrt(p) * |0><0|
      K2 = sqrt(p) * |1><1|

    Notes:
    - This is NOT the Pauli phase-flip channel.
    - p must be in [0, 1].
    """
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1]; got p={p}")

    s0 = np.sqrt(1.0 - p)
    sp = np.sqrt(p)

    K0 = s0 * np.eye(2, dtype=np.complex128)
    K1 = np.array([[sp, 0.0], [0.0, 0.0]], dtype=np.complex128)
    K2 = np.array([[0.0, 0.0], [0.0, sp]], dtype=np.complex128)
    return [K0, K1, K2]


def apply_kraus_to_density(
    rho: NDArray[np.complex128],
    Ks: List[NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """
    Deterministic Kraus-sum application to a 2x2 density matrix:
      rho' = sum_k K_k rho K_k^dagger
    Used for tests and sanity checks.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (2, 2):
        raise ValueError(f"rho must have shape (2,2); got {rho.shape}")

    out = np.zeros_like(rho)
    for K in Ks:
        K = np.asarray(K, dtype=np.complex128)
        if K.shape != (2, 2):
            raise ValueError(f"Each K must be (2,2); got {K.shape}")
        out += K @ rho @ K.conj().T
    return out


def apply_kraus_sample(
    psi: NDArray[np.complex128],
    Ks: List[NDArray[np.complex128]],
    rng: np.random.Generator,
) -> Tuple[NDArray[np.complex128], int]:
    """
    Trajectory (quantum-jump) sampling of a Kraus channel acting on the COIN only.

    Given statevector psi with shape (2, Npos), compute:
      psi_k = K_k @ psi
      p_k = ||psi_k||^2

    Then sample k ~ Categorical(p_k), and return:
      psi <- psi_k / sqrt(p_k)

    Returns:
      (psi_new, k_index)
    """
    psi = np.asarray(psi, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi must have shape (2, Npos); got {psi.shape}")

    psi_list: List[NDArray[np.complex128]] = []
    p_list: List[float] = []

    for K in Ks:
        psi_k = np.asarray(K, dtype=np.complex128) @ psi
        pk = float(np.sum(np.abs(psi_k) ** 2))
        psi_list.append(psi_k)
        p_list.append(pk)

    p_raw = np.asarray(p_list, dtype=float)
    total = float(np.sum(p_raw))
    if not (total > 0.0):
        raise RuntimeError("All Kraus probabilities are zero; input state likely has zero norm.")

    p_norm = p_raw / total
    k = int(rng.choice(len(Ks), p=p_norm))

    psi_k = psi_list[k]
    pk = p_raw[k]
    if not (pk > 0.0):
        # This should not occur if sampling was correct, but guard anyway.
        raise RuntimeError("Sampled a zero-probability Kraus outcome (numerical issue).")

    psi_new = psi_k / np.sqrt(pk)
    return psi_new, k


def run_sequence_pd_trajectories(
    sequence,
    T: int,
    phi: float,
    p: float,
    coin_params: dict,
    N_traj: int,
    seed: int = 0,
    init_state: NDArray[np.complex128] | None = None,
) -> dict:
    """
    Noisy DTQW using PD (phase damping) on the COIN only via trajectory sampling.

    Per bootstrap plan confirmatory order:
      coin -> defect -> coin-noise (PD) -> shift

    Returns:
      dict with:
        x: positions (2T+1,)
        P_t: mean position distribution over trajectories, shape (T+1, 2T+1)
    """
    T = int(T)
    if T < 0:
        raise ValueError("T must be nonnegative")
    N_traj = int(N_traj)
    if N_traj <= 0:
        raise ValueError("N_traj must be positive")

    if isinstance(sequence, str):
        seq = list(sequence)
    else:
        seq = list(sequence)
    if len(seq) == 0:
        raise ValueError("sequence must be non-empty")

    Npos = 2 * T + 1
    x = np.arange(-T, T + 1, dtype=int)

    Ks = kraus_pd(p)
    rng = np.random.default_rng(int(seed))

    P_acc = np.zeros((T + 1, Npos), dtype=np.float64)

    for _ in range(N_traj):
        if init_state is None:
            psi = default_init_state(T)
        else:
            psi = np.asarray(init_state, dtype=np.complex128)
            if psi.shape != (2, Npos):
                raise ValueError(f"init_state must have shape (2, {Npos}); got {psi.shape}")
            psi = psi.copy()

        # t=0
        P_acc[0] += np.sum(np.abs(psi) ** 2, axis=0).real

        for t in range(1, T + 1):
            key = seq[(t - 1) % len(seq)]
            if key not in coin_params:
                raise KeyError(f"coin_params missing key {key!r}")

            alpha, beta, gamma = coin_params[key]
            C = su2_coin(alpha, beta, gamma, degrees=True)

            psi = apply_coin(psi, C)
            psi = apply_origin_phase_defect(psi, phi)
            psi, _k = apply_kraus_sample(psi, Ks, rng)
            psi = apply_conditional_shift(psi)

            P_acc[t] += np.sum(np.abs(psi) ** 2, axis=0).real

    P_mean = P_acc / float(N_traj)
    return {"x": x, "P_t": P_mean}
