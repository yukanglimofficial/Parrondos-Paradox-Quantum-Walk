from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_conditional_shift(psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Conditional shift for a 1D coined DTQW.

    Project convention (bootstrap_plan reference implementation):
      - coin state |0> shifts RIGHT: x -> x + 1
      - coin state |1> shifts LEFT:  x -> x - 1

    Guardrail: MUST NOT wrap around (no circular np.roll).
    Implemented via slicing with boundary fill = 0.
    """
    psi = np.asarray(psi)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError(f"psi must have shape (2, Npos); got {psi.shape}")

    out = np.zeros_like(psi)

    # |0> moves right: output index j+1 gets input index j
    out[0, 1:] = psi[0, :-1]

    # |1> moves left: output index j-1 gets input index j
    out[1, :-1] = psi[1, 1:]

    return out
