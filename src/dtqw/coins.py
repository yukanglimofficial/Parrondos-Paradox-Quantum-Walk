from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def su2_coin(alpha: float, beta: float, gamma: float, degrees: bool = True) -> NDArray[np.complex128]:
    """
    SU(2) coin rotation operator R(alpha, beta, gamma) as used by Jan et al. (AQT 2020).

    If degrees=True (default), alpha/beta/gamma are interpreted as degrees and converted
    to radians before calling sin/cos.

    R(alpha, beta, gamma) =
        [[ exp(i*alpha) * cos(beta),  -exp(-i*gamma) * sin(beta) ],
         [ exp(i*gamma) * sin(beta),   exp(-i*alpha) * cos(beta) ]]

    Returns:
        (2,2) complex128 matrix
    """
    if degrees:
        a = np.deg2rad(alpha)
        b = np.deg2rad(beta)
        g = np.deg2rad(gamma)
    else:
        a = float(alpha)
        b = float(beta)
        g = float(gamma)

    c = np.cos(b)
    s = np.sin(b)

    C = np.array(
        [
            [np.exp(1j * a) * c, -np.exp(-1j * g) * s],
            [np.exp(1j * g) * s,  np.exp(-1j * a) * c],
        ],
        dtype=np.complex128,
    )
    return C


def apply_coin(psi: NDArray[np.complex128], C: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Apply a 2x2 coin matrix C to the coin axis of psi.

    psi must have shape (2, Npos).
    """
    psi = np.asarray(psi)
    C = np.asarray(C)

    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi must have shape (2, Npos); got {psi.shape}")
    if C.shape != (2, 2):
        raise ValueError(f"C must have shape (2, 2); got {C.shape}")

    return C @ psi
