import numpy as np

from dtqw.shift import apply_conditional_shift


def test_shift_no_wraparound_edges_drop():
    """
    If someone mistakenly uses np.roll, amplitudes at the edges would wrap to the opposite side.
    This test makes that bug visible.

    Convention: |0> moves RIGHT, |1> moves LEFT.
    """
    psi = np.zeros((2, 5), dtype=np.complex128)

    # Amplitudes on edges that would wrap under np.roll:
    psi[0, -1] = 1.0 + 0j  # |0> at right edge would wrap to index 0 under roll(+1)
    psi[1, 0]  = 1.0 + 0j  # |1> at left edge would wrap to last index under roll(-1)

    out = apply_conditional_shift(psi)

    # Must NOT wrap:
    assert out[0, 0] == 0.0 + 0j
    assert out[1, -1] == 0.0 + 0j


def test_shift_moves_interior_correctly_and_conserves_prob():
    psi = np.zeros((2, 5), dtype=np.complex128)
    psi[0, 2] = 1.0 + 0j  # coin 0 at center should move right
    psi[1, 2] = 2.0 + 0j  # coin 1 at center should move left

    out = apply_conditional_shift(psi)

    assert out[0, 3] == 1.0 + 0j
    assert out[1, 1] == 2.0 + 0j

    # Probability conserved for this interior-only state
    assert np.isclose(np.sum(np.abs(out) ** 2), 1.0**2 + 2.0**2)
