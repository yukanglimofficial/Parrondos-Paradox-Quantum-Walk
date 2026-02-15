from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_origin_phase_defect(psi: NDArray[np.complex128], phi: float) -> NDArray[np.complex128]:
    """
    Apply a single-point phase defect at the origin x=0.

    We assume the position axis represents x in [-T, ..., +T] (odd length Npos),
    so the origin is the center index: idx0 = Npos // 2.

    The defect multiplies BOTH coin components at x=0 by exp(i*phi).
    """
    psi = np.asarray(psi)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi must have shape (2, Npos); got {psi.shape}")

    Npos = psi.shape[1]
    if Npos % 2 != 1:
        raise ValueError(f"Npos must be odd so x=0 is representable; got Npos={Npos}")

    out = psi.copy()
    idx0 = Npos // 2
    out[:, idx0] *= np.exp(1j * float(phi))
    return out
