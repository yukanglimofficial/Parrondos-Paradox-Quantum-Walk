import numpy as np

from dtqw.metrics import (
    P0bar,
    compute_metrics_from_P_t,
    deltaP,
    fit_v_fit,
    ipr,
    mean_x,
    sigma_x,
    w_loc,
)


def test_metrics_delta_at_origin():
    # x = [-2, -1, 0, 1, 2]
    x = np.arange(-2, 3, dtype=int)
    P_t = np.zeros((6, x.size), dtype=float)
    P_t[:, 2] = 1.0  # all mass at x=0 for all times

    mx = mean_x(P_t, x)
    dP = deltaP(P_t, x)
    sig = sigma_x(P_t, x)
    ip = ipr(P_t)

    assert np.allclose(mx, 0.0)
    assert np.allclose(dP, 0.0)
    assert np.allclose(sig, 0.0)
    assert np.allclose(ip, 1.0)  # delta distribution => sum P^2 = 1

    assert w_loc(P_t, x, x0=0, T0=0, T1=5) == 1.0
    assert P0bar(P_t, x, T0=0, T1=5) == 1.0


def test_fit_v_fit_exact_line():
    t = np.arange(10, dtype=float)
    y = 2.0 * t + 1.0
    slope, intercept = fit_v_fit(y, T0=3)
    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, 1.0)


def test_compute_metrics_wrapper_runs():
    x = np.arange(-3, 4, dtype=int)
    P_t = np.zeros((5, x.size), dtype=float)
    # Put mass at x=+1 for all times -> mean=+1, deltaP=1, sigma=0
    P_t[:, np.where(x == 1)[0][0]] = 1.0

    m = compute_metrics_from_P_t(P_t, x, T0=2, x0_loc=1)
    assert np.allclose(m["x_mean_t"], 1.0)
    assert np.allclose(m["deltaP_t"], 1.0)
    assert np.allclose(m["sigma_t"], 0.0)
