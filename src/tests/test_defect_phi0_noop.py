import numpy as np

from dtqw.defect import apply_origin_phase_defect


def test_defect_phi0_noop():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(2, 11)) + 1j * rng.normal(size=(2, 11))
    psi = psi.astype(np.complex128)

    out = apply_origin_phase_defect(psi, phi=0.0)

    np.testing.assert_allclose(out, psi, atol=0.0, rtol=0.0)
