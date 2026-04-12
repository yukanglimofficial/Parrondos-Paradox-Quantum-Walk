from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dtqw.coins import su2_coin, apply_coin
from dtqw.defect import apply_origin_phase_defect
from dtqw.shift import apply_conditional_shift


def default_init_state(T: int) -> NDArray[np.complex128]:
    """
    Default initial state (Jan et al. 2020):
      |Psi(0)> = (|0> - i|1>)/sqrt(2) tensor |x=0>

    Returned as a statevector with shape (2, 2T+1).
    """
    Npos = 2 * T + 1
    psi = np.zeros((2, Npos), dtype=np.complex128)
    psi[:, T] = np.array([1.0, -1j], dtype=np.complex128) / np.sqrt(2.0)
    return psi


def run_sequence_unitary(
    sequence,
    T: int,
    phi: float,
    coin_params: dict,
    init_state: NDArray[np.complex128] | None = None,
) -> dict:
    """
    Run a noiseless coined DTQW for T steps using a periodic coin sequence.

    Per bootstrap plan, per-step order is:
      coin -> defect -> shift

    Args:
      sequence: e.g. "ABB" or ["A","B","B"] (cycled over T steps)
      T: number of steps
      phi: defect phase at x=0 applied every step
      coin_params: mapping like {"A": (alpha,beta,gamma), "B": (...)}.
                  Angles interpreted as degrees by su2_coin().
      init_state: optional psi0 with shape (2, 2T+1). If None, uses default_init_state(T).

    Returns:
      dict with:
        x: integer positions array shape (2T+1,)
        psi_t: complex array shape (T+1, 2, 2T+1) including t=0
    """
    T = int(T)
    if T < 0:
        raise ValueError("T must be nonnegative")

    if isinstance(sequence, str):
        seq = list(sequence)
    else:
        seq = list(sequence)

    if len(seq) == 0:
        raise ValueError("sequence must be non-empty")

    Npos = 2 * T + 1
    x = np.arange(-T, T + 1, dtype=int)

    if init_state is None:
        psi = default_init_state(T)
    else:
        psi = np.asarray(init_state, dtype=np.complex128)
        if psi.shape != (2, Npos):
            raise ValueError(f"init_state must have shape (2, {Npos}); got {psi.shape}")
        psi = psi.copy()

    psi_t = np.zeros((T + 1, 2, Npos), dtype=np.complex128)
    psi_t[0] = psi

    for t in range(1, T + 1):
        key = seq[(t - 1) % len(seq)]
        if key not in coin_params:
            raise KeyError(f"coin_params missing key {key!r}")

        alpha, beta, gamma = coin_params[key]
        C = su2_coin(alpha, beta, gamma, degrees=True)

        psi = apply_coin(psi, C)
        psi = apply_origin_phase_defect(psi, phi)
        psi = apply_conditional_shift(psi)

        psi_t[t] = psi

    return {"x": x, "psi_t": psi_t}
