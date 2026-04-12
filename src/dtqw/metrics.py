from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def prob_from_psi_t(psi_t: NDArray[np.complex128]) -> FloatArray:
    """
    Convert psi_t (T+1, 2, Npos) to P_t (T+1, Npos) by summing coin probabilities.
    """
    psi_t = np.asarray(psi_t, dtype=np.complex128)
    if psi_t.ndim != 3 or psi_t.shape[1] != 2:
        raise ValueError(f"psi_t must have shape (T+1, 2, Npos); got {psi_t.shape}")
    P_t = np.sum(np.abs(psi_t) ** 2, axis=1).real.astype(np.float64)
    return P_t


def _as_2d_P_t(P_t: NDArray[Any]) -> Tuple[FloatArray, bool]:
    """
    Return (P2d, was_1d). P2d is always (Tn, Npos).
    """
    P = np.asarray(P_t, dtype=np.float64)
    if P.ndim == 1:
        return P[None, :], True
    if P.ndim == 2:
        return P, False
    raise ValueError(f"P_t must have shape (Npos,) or (T+1, Npos); got {P.shape}")


def mean_x(P_t: NDArray[Any], x: NDArray[Any]) -> FloatArray:
    """
    Compute <x>(t) = sum_x x * P(x,t).
    """
    P2d, was_1d = _as_2d_P_t(P_t)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1D; got {x.shape}")
    if P2d.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch: P_t has Npos={P2d.shape[1]} but x has {x.shape[0]}")

    mx = np.sum(P2d * x[None, :], axis=1)
    return mx[0:1] if was_1d else mx


def deltaP(P_t: NDArray[Any], x: NDArray[Any]) -> FloatArray:
    """
    deltaP(t) = P_R(t) - P_L(t) where:
      P_R = sum_{x>0} P(x,t)
      P_L = sum_{x<0} P(x,t)
    """
    P2d, was_1d = _as_2d_P_t(P_t)
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if P2d.shape[1] != x.shape[0]:
        raise ValueError("Shape mismatch between P_t and x")

    mR = x > 0
    mL = x < 0
    PR = np.sum(P2d[:, mR], axis=1)
    PL = np.sum(P2d[:, mL], axis=1)
    dP = PR - PL
    return dP[0:1] if was_1d else dP


def sigma_x(P_t: NDArray[Any], x: NDArray[Any]) -> FloatArray:
    """
    sigma(t) = sqrt(<x^2>(t) - <x>(t)^2)
    """
    P2d, was_1d = _as_2d_P_t(P_t)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if P2d.shape[1] != x.shape[0]:
        raise ValueError("Shape mismatch between P_t and x")

    mx = np.sum(P2d * x[None, :], axis=1)
    m2 = np.sum(P2d * (x[None, :] ** 2), axis=1)
    var = m2 - mx**2

    # Numerical guard: clamp tiny negative values to 0
    var = np.where(var < 0.0, np.maximum(var, -1e-15), var)
    var = np.maximum(var, 0.0)

    sig = np.sqrt(var)
    return sig[0:1] if was_1d else sig


def ipr(P_t: NDArray[Any]) -> FloatArray:
    """
    IPR(t) = sum_x P(x,t)^2
    (Higher IPR generally indicates more localization.)
    """
    P2d, was_1d = _as_2d_P_t(P_t)
    out = np.sum(P2d**2, axis=1)
    return out[0:1] if was_1d else out


def w_loc(P_t: NDArray[Any], x: NDArray[Any], x0: int, T0: int, T1: int | None = None) -> float:
    """
    w_loc = mean_t sum_{|x| <= x0} P(x,t) over t in [T0, T1] (inclusive).
    """
    P2d, _ = _as_2d_P_t(P_t)
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1 or P2d.shape[1] != x.shape[0]:
        raise ValueError("Shape mismatch between P_t and x")

    if T1 is None:
        T1 = P2d.shape[0] - 1

    T0 = int(T0)
    T1 = int(T1)
    if not (0 <= T0 <= T1 < P2d.shape[0]):
        raise ValueError(f"Invalid time window: T0={T0}, T1={T1}, Tn={P2d.shape[0]}")

    m = np.abs(x) <= int(x0)
    w_t = np.sum(P2d[:, m], axis=1)
    return float(np.mean(w_t[T0 : T1 + 1]))


def P0bar(P_t: NDArray[Any], x: NDArray[Any], T0: int, T1: int | None = None) -> float:
    """
    P0bar = mean_t P(x=0, t) over t in [T0, T1] (inclusive).
    """
    P2d, _ = _as_2d_P_t(P_t)
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1 or P2d.shape[1] != x.shape[0]:
        raise ValueError("Shape mismatch between P_t and x")

    idx0 = np.where(x == 0)[0]
    if idx0.size != 1:
        raise ValueError("x must contain exactly one 0 position")
    i0 = int(idx0[0])

    if T1 is None:
        T1 = P2d.shape[0] - 1

    T0 = int(T0)
    T1 = int(T1)
    if not (0 <= T0 <= T1 < P2d.shape[0]):
        raise ValueError(f"Invalid time window: T0={T0}, T1={T1}, Tn={P2d.shape[0]}")

    p0_t = P2d[:, i0]
    return float(np.mean(p0_t[T0 : T1 + 1]))


def fit_v_fit(x_mean_t: NDArray[Any], T0: int) -> Tuple[float, float]:
    """
    Fit drift velocity v_fit as the OLS slope of <x>(t) over late window t in [T0, T].
    Returns (slope, intercept) for y = slope * t + intercept.
    """
    y = np.asarray(x_mean_t, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("x_mean_t must be 1D")

    T0 = int(T0)
    Tn = y.shape[0]
    if not (0 <= T0 < Tn):
        raise ValueError(f"Invalid T0={T0} for series length {Tn}")

    t = np.arange(Tn, dtype=np.float64)
    t = t[T0:]
    y = y[T0:]

    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))

    denom = float(np.sum((t - t_mean) ** 2))
    if denom == 0.0:
        # Window has length 1
        return 0.0, float(y[0])

    slope = float(np.sum((t - t_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * t_mean)
    return slope, intercept


def compute_metrics_from_P_t(
    P_t: NDArray[Any],
    x: NDArray[Any],
    T0: int,
    x0_loc: int,
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute the standard metric set from P_t and x.
    """
    P2d, _ = _as_2d_P_t(P_t)
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 1 or P2d.shape[1] != x.shape[0]:
        raise ValueError("Shape mismatch between P_t and x")

    mx = mean_x(P2d, x).astype(np.float64)
    dP = deltaP(P2d, x).astype(np.float64)
    sig = sigma_x(P2d, x).astype(np.float64)
    ipr_t = ipr(P2d).astype(np.float64)

    v_fit, v_int = fit_v_fit(mx, T0=T0)
    w = w_loc(P2d, x, x0=int(x0_loc), T0=int(T0), T1=None)
    p0 = P0bar(P2d, x, T0=int(T0), T1=None)

    return {
        "x_mean_t": mx,
        "deltaP_t": dP,
        "sigma_t": sig,
        "ipr_t": ipr_t,
        "v_fit": float(v_fit),
        "v_intercept": float(v_int),
        "w_loc": float(w),
        "P0bar": float(p0),
        "T0": int(T0),
        "x0_loc": int(x0_loc),
    }
