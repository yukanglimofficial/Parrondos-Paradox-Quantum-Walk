# Math appendix notes

## Proposition 1 (phase damping CPTP map)

Let
$$K_0=\sqrt{1-p}\,I,\qquad K_1=\sqrt{p}\,|0\rangle\langle 0|,\qquad K_2=\sqrt{p}\,|1\rangle\langle 1|.$$
Then $\sum_k K_k^\dagger K_k = I$, so the map $\mathcal E_p(\rho)=\sum_k K_k\rho K_k^\dagger$ is CPTP. For
$$\rho=\begin{pmatrix}a&c\\ c^\ast & b\end{pmatrix},$$
direct multiplication gives
$$\mathcal E_p(\rho)=\begin{pmatrix}a&(1-p)c\\ (1-p)c^\ast & b\end{pmatrix}.$$

## Proposition 2 (phi-irrelevance after complete dephasing)

In the fully classicalized $Z_{cp}$ null, all off-diagonal phases are removed each step, so the origin defect contributes only a phase factor with no effect on classical position probabilities. Therefore the $p=1$ classicalized null is $\phi$-invariant.

## Lemma 3 (bias and drift can disagree)

A simple counterexample: place probability $0.51$ at $x=+1$ and probability $0.49$ at $x=-100$. Then $\Delta P=0.51-0.49>0$, but $\langle x\rangle=0.51-49<0$. Hence right-heavy bias does not imply positive mean-position drift.

## Proposition 4 (finite-horizon primary transport interval at p=0)

At fixed-window horizon T=6000, the exact follow-up estimates a nonzero primary PP interval approximately [0.551688, 0.564684]·pi with width 0.012996·pi.

State this explicitly as an exact finite-horizon proposition, not as an infinite-time theorem.

## Wording caution

Use phrases such as 'exact finite-horizon', 'long-horizon follow-up', and 'finite-volume Floquet proxy'. Do not overstate these computations as closed-form infinite-time proofs.
