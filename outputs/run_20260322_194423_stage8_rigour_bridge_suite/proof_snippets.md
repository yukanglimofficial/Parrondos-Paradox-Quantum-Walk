# Stage-8 proof snippets

## Proposition 1 (phase damping is CPTP)

With Kraus operators
$$K_0=\sqrt{1-p}\,I,\qquad K_1=\sqrt{p}\,|0\rangle\langle 0|,\qquad K_2=\sqrt{p}\,|1\rangle\langle 1|,$$
we have $\sum_k K_k^\dagger K_k = I$, so $\mathcal E_p(\rho)=\sum_k K_k\rho K_k^\dagger$ is CPTP. For $\rho=\begin{pmatrix}a&c\\ c^\ast&b\end{pmatrix}$ one gets
$$\mathcal E_p(\rho)=\begin{pmatrix}a&(1-p)c\\ (1-p)c^\ast&b\end{pmatrix}.$$

## Proposition 2 ($\phi$ is irrelevant at complete dephasing)

At $p=1$, every step removes all off-diagonal coin coherences before the shift statistics are read. The origin defect multiplies amplitudes at $x=0$ by a common phase factor $e^{i\phi}$, which leaves diagonal probabilities unchanged. Hence the fully classicalized null is $\phi$-invariant.

## Lemma 3 (bias and drift can disagree)

Take a distribution with probability $0.51$ at $x=+1$ and probability $0.49$ at $x=-100$. Then $\Delta P=0.51-0.49>0$ but $\langle x\rangle=0.51-49<0$. Therefore right-heavy bias does not imply positive mean-position drift.

## Proposition 4 (finite-horizon primary interval, exact follow-up)

At fixed-window horizon $T=12000$, the exact follow-up estimates a nonzero primary PP interval $[0.551798,\,0.564652]\pi$ with width 0.012854$\pi$.

State this explicitly as a **finite-horizon exact proposition**, not as an infinite-time theorem.

## Proposition 5 (localized-band / transport-band distinction)

The refined maximum-advantage point has extrapolated ABB drift 0.108154 and extrapolated $w_{\mathrm{loc},3}$ 0.579486, while the PP-primary midpoint has extrapolated ABB drift 0.087049 and extrapolated $w_{\mathrm{loc},3}$ 0.083203. This separates the strongest-drift localized band from the narrower transport interval.
On the largest tested ring ($L=300$), the top-localized-mode overlap is 0.608498 at the refined maximum but 0.140105 at the PP-primary midpoint, supporting the bound-state/localization interpretation of the high-advantage band.

## Wording guardrail

Use phrases such as **exact finite-horizon**, **long-horizon noiseless follow-up**, and **finite-volume Floquet proxy**. Avoid claiming a closed-form infinite-time theorem unless you actually prove one.