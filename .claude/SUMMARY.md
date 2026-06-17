# Structured GP Regression for Radio Telescope Data — Implementation Summary

Handoff notes for a Claude Code implementation session.

For CLAUDE: do not edit this doc. It is read-only.

**Stack decision:** PyTorch + GPyTorch + `linear_operator` (PyTorch-only; we evaluated
JAX options — COLA, Lineax — but chose the mature PyTorch stack for now).

---

## 1. Problem

- 2D data `X` of shape `(N_times, N_freqs)`, e.g. `(256, 256)` (must scale toward `1024²`).
- Goal: GP regression / Wiener filtering / gap-filling with a **structured, mildly
  non-stationary, heteroscedastic-noise** covariance, plus hyperparameter optimization.
- Naive dense covariance is `N×N` with `N = N_times·N_freqs` (4.3e9 entries at 256², ~10¹²
  at 1024²) — never form it. Everything is matrix-free / structured.

---

## 2. Covariance model (building blocks)

**Unrolling convention:** row-major, time slow / freq fast. The separable signal
covariance is then a Kronecker product

$$K = T \otimes F$$

with `T` the `(N_t, N_t)` time core and `F` the `(N_f, N_f)` freq core.
Matvec via the matricization identity (never densify):

$$(T \otimes F)\,\mathrm{vec}(X) = \mathrm{vec}(T\,X\,F) \quad\Rightarrow\quad \texttt{T @ Xmat @ F}.$$

**Non-stationarity via diagonal congruence (`HCH`):** each core is a stationary core
modulated by a diagonal amplitude envelope:

$$F = f\,G\,f, \qquad T = t\,P\,t,$$

with `f`, `t` diagonal envelopes and `G`, `P` stationary. `HCH` models non-stationary
**variance** (amplitude breathing), *not* non-stationary lengthscale/spectral content.
It is unconditionally PSD (congruence), so "mild" is about identifiability, not validity.
Keep envelopes smooth/low-parameter or they are unidentifiable vs. the cores.

**Noise:** heteroscedastic, diagonal `D = diag(vec(Nvar))`, where the variance *field*
`Nvar` (shape `(N_t, N_f)`) is low-rank: `Nvar = U @ S @ V.T`. The low rank is a
**storage + preconditioner-seed** win, NOT an operator-rank win — `D` is full-rank
diagonal. Equivalently `D = Σ_a s_a · diag(u_a) ⊗ diag(v_a)` (sum of Kronecker-of-diagonals).

**Key congruence identities** (used in both paths). With `H = diag(vec(h))`, `C` stationary:

$$(HCH + D)^{-1} = H^{-1}(C + \tilde D)^{-1}H^{-1}, \qquad \tilde D = \mathrm{diag}(\mathrm{vec}(\,Nvar / h^2\,))\ \text{(still diagonal)}.$$

$$\log|HCH + D| = 2\sum_{ij}\log|h_{ij}| + \log|C + \tilde D|.$$

So the non-stationary solve reduces to a stationary solve bracketed by two O(N) diagonal
scalings: prescale `ỹ = y/h`, solve `(C + D̃)β = ỹ`, postscale `α = β/h`.

---

## 3. The two non-separability paths

The cross-axis dependence (frequency covariance varying with time, and vice versa) lands in
one of two places. **Determine which physically before coding** — they cost very differently.

### Path (i): non-separability is in the diagonal envelopes

`f` varies with time and `t` with frequency, but only through *amplitude*; the stationary
cores `G`, `P` stay global. Everything collapses into one diagonal envelope over the grid:

$$K = H\,(P \otimes G)\,H, \qquad H = \mathrm{diag}(\mathrm{vec}(h)),\ h_{ij} = t_{ij}\,f_{ij}.$$

(`h` is a general 2D field, not a rank-1 outer product — that's what the cross-dependence buys.)

**Procedure (exact reduction):**
1. Eigendecompose `P`, `G` once (`(256,256)` `eigh`, ms).
2. Prescale `ỹ = y / h`.
3. Solve `(P⊗G + D̃)β = ỹ` with preconditioned CG; preconditioner `(P⊗G + σ̄²I)⁻¹` from the
   joint eigendecomposition (project / divide by `λ^P_a λ^G_b + σ̄²` / back-project).
   `σ̄² = mean(D̃)`. Mild noise spread ⇒ few iterations.
4. Postscale `α = β / h`.
5. Log-det: `2 Σ log|h_ij| + Σ_ab log(λ^P_a λ^G_b + σ̄²)` (exact split).

This is the cheap/exact path. Cost ≈ stationary separable solve + O(N) scalings.

### Path (ii): non-separability is in the cores

`G = G(t)` and `P = P(ν)` — the correlation *structure* (not just amplitude) reshapes across
the orthogonal axis. No single `P⊗G`. Model as a short sum of Kronecker terms (LMC /
first-order Taylor expansion of the drift):

$$K = H\Big[\sum_{m=0}^{M} P_m \otimes G_m\Big]H,$$

`m=0` the bulk, `m≥1` the mild-drift modes (`M = 1–3` for mild drift).

**Procedure (preconditioned CG over the drift):**
1. Eigendecompose the **bulk** `P_0`, `G_0` once → preconditioner.
2. Matvec = sum of `M+1` Kronecker matvecs + diagonal scalings + noise multiply,
   each O(N^{1.5}).
3. Preconditioned CG with the bulk separable term as preconditioner. CG now iterates over
   **both** the drift correction and the noise; iteration count scales with drift magnitude.
4. Log-det has no clean split → stochastic Lanczos quadrature (SLQ).

**Exact escape hatch:** if the cores only rescale eigenvalues in a *fixed* basis (frequencies
stable, amplitudes drift), transform to that basis once and the problem decouples into
independent 1D-in-Y GPs — no CG.

---

## 4. GPyTorch / `linear_operator` mapping

| Concept | Class |
|---|---|
| Stationary Kronecker core `P⊗G` | `KroneckerProductLinearOperator(P, G)` — exact eig `solve`/`logdet`/`inv_quad_logdet` |
| `P⊗G + diag` | `KroneckerProductAddedDiagLinearOperator` (auto from `Kron + DiagLinearOperator(d)`) |
| Sum of Kronecker (path ii) | `SumKroneckerLinearOperator(...)` — *verify constructor signature*; or `SumLinearOperator([KroneckerProductLinearOperator(Pm, Gm) ...])` |
| Diagonal envelope / noise | `DiagLinearOperator(...)`, `AddedDiagLinearOperator` |
| Marginal likelihood | `op.inv_quad_logdet(rhs, logdet=True)` (fused `yᵀK⁻¹y` + `log|K|`) |

**Path (i) sketch (operate at the `linear_operator` level for full control):**

```python
import math, torch
from linear_operator.operators import KroneckerProductLinearOperator, DiagLinearOperator

def neg_log_mll(params, y, Nt, Nf):
    P, G, h, Nvar = build_from_params(params)      # P:(Nt,Nt) G:(Nf,Nf) h,Nvar:(Nt,Nf)
    hvec  = h.reshape(-1)
    ytil  = (y.reshape(Nt, Nf) / h).reshape(-1)    # H^{-1} y
    dtil  = (Nvar / h**2).reshape(-1)              # H^{-1} D H^{-1}, stays diagonal
    A     = KroneckerProductLinearOperator(P, G) + DiagLinearOperator(dtil)
    inv_quad, logdet = A.inv_quad_logdet(ytil.unsqueeze(-1), logdet=True)
    logdet_h = 2.0 * hvec.abs().log().sum()        # Jacobian of the H^{-1} rescaling
    N = Nt * Nf
    return 0.5 * (inv_quad + logdet + logdet_h + N * math.log(2 * math.pi))

# optimize params with torch.autograd + Adam/LBFGS
# posterior mean: alpha = (A^{-1} ytil) / hvec ; then mean = (P⊗G or K) @ (alpha grid)
```

**Path (ii):** swap the `KroneckerProductLinearOperator(P, G)` for the sum-of-Kronecker
operator over the `(P_m, G_m)` list; everything else is identical.

**You can also wrap this as a custom `gpytorch.kernels.Kernel`** whose `forward` returns the
structured `LinearOperator`, then use `ExactGP` + `ExactMarginalLogLikelihood`. Operating
directly on the operator (as above) gives more control over the `HCH` congruence and the
heteroscedastic noise; pick whichever fits the codebase.

---

## 5. Critical GPyTorch caveats / gotchas

- **Heteroscedastic noise forces the iterative path in BOTH paths.** The exact Kronecker
  closed form in `KroneckerProductAddedDiagLinearOperator` (the `λ^P_a λ^G_b + σ²` spectral
  shift) is valid **only for a constant (homoscedastic) added diagonal**. Your low-rank
  `Nvar` makes `D̃` non-constant, so the solve drops to CG + SLQ log-det. The envelope
  congruence stays exact/free; it's the noise that forces iteration.
- **Default preconditioner is generic pivoted-Cholesky (rank `max_preconditioner_size=15`),
  not your structured Kronecker-eigendecomposition preconditioner.** This is the one place
  GPyTorch won't hand you your performance edge. For best CG convergence, inject the
  structured preconditioner (the `(P⊗G + σ̄²I)⁻¹` projector). `min_preconditioning_size`
  default 2000 — below that, no preconditioner.
- **Cholesky fallback is available and is the default for small N.** `max_cholesky_size`
  default **800**: matrices ≤ 800 use jittered Cholesky (`psd_safe_cholesky`). Force exact
  Cholesky at any size with `gpytorch.settings.fast_computations(covar_root_decomposition=False,
  log_prob=False, solves=False)`. Use this to **validate** the iterative+structured path
  against exact at small N (conformance test).
- **`fast_pred_var` defaults to `False`** — predictive variances are exact (Cholesky-root)
  by default; opt *in* to LOVE (`gpytorch.settings.fast_pred_var()`) for fast predictive vars.
- **SLQ defaults:** `num_trace_samples=10`, `max_lanczos_quadrature_iterations=20`,
  `cg_tolerance` loose for probe solves. SLQ log-det is **stochastic** — see §6 caveat.

---

## 6. Hyperparameter optimization

Marginal-likelihood gradient decomposes into a cheap and an expensive term:

$$\partial_i \mathcal L = \tfrac12\big[\underbrace{-\,\alpha^\top(\partial_i K)\alpha}_{\text{cheap: needs only }\alpha=K^{-1}y} + \underbrace{\mathrm{tr}(K^{-1}\partial_i K)}_{\text{expensive: SLQ}}\big].$$

When already near the optimum and only minimally adjusting:

1. **Profile out scale & noise (exact, do always).** Write `K = σ_f²(R + ηI)`, `η = σ_n²/σ_f²`.
   Then `σ̂_f² = (1/N) yᵀ(R+ηI)⁻¹y` closed form, and `η` is a 1-D search reusing one
   eigendecomposition: `log|R+ηI| = Σ_ab log(ρ_ab + η)`. Removes 2 params from the loop.
2. **Freeze + linearize the complexity term (scenario ii).** Compute `log|K|` and its
   trace-gradient `g_h` once at θ₀; linearize. Then optimize a surrogate where only the
   cheap exact data-fit term (and its gradient `-αᵀ∂Kα`) is re-evaluated. Error O(‖θ-θ₀‖²).
3. **One Fisher / Gauss-Newton step:** `δθ = -ℐ⁻¹g`, `ℐ_ij = ½ tr(K⁻¹∂_iK K⁻¹∂_jK)`.
   `ℐ` is `p×p` (tiny), PSD, no line search. Computed once → single corrective step.
   (This is the Fisher-matrix machinery turned inward on hyperparameter space.)
- **In scenario (i)**, the complexity gradient is the closed form `Σ ∂log(λ^P λ^G + σ̄²)` —
  cheap and exact, so skip the freezing and just do one Fisher step.
- **Caveat:** near the optimum the true gradient is small, so SLQ trace noise can dominate
  and the optimizer jitters. Either freeze the trace (above) or **fix the Hutchinson probe
  vectors** across evaluations so the stochastic error cancels in differences
  (GPyTorch `deterministic_probes`). Don't run noisy SGD at the minimum.

---

## 7. Complex-valued data (phase information)

Complex data needs **two** second-order matrices: Hermitian covariance `C = E[zzᴴ]` and
pseudo-covariance `C̃ = E[zzᵀ]`. The pseudo-covariance is the proper/improper diagnostic.

- **Proper (circular), `C̃ = 0`:** single Hermitian `N×N` kernel captures the full coupling
  (cross-covariance is `Im(C)`). Complex linear algebra — ~2× cheaper than naive real
  stacking, same storage as the (uncoupled) separate-real/imag approach. All structured
  machinery ports (`eigh` on Hermitian, complex Kronecker, complex `HCH`).
- **Our case — "single phase offset" = rectilinear / maximally noncircular.** The signal is
  a real process under a fixed rotation, `z = e^{iφ} u`, `u` real. Then:
  - `C_xx = cos²φ · R`, `C_yy = sin²φ · R` — **unequal** (ratio `cot²φ`), same shape `R`.
  - `C_xy = ½ sin2φ · R` — symmetric, nonzero. `C̃ = e^{2iφ} R` (maximally improper).
  - Note: `C_xx = C_yy` at φ=45° does **not** imply proper. Test properness via `|C̃|`, not
    equal diagonal blocks.
  - **Payoff:** the signal has only `N` real DOF. De-rotate `u = Re(e^{-iφ} z)` and run a
    single **real** `N×N` GP — cheaper than everything (¼ of naive stacking).
  - Circular thermal noise stays circular under de-rotation ⇒ the de-rotated **imaginary
    channel is signal-free noise**: discard it, or use it as a clean noise/φ estimator.
    `φ` = phase of `C̃` / 2, or maximize real-channel signal power; can be a hyperparameter.
- **Decision step:** estimate `C̃` from the data; near-zero ⇒ proper (single Hermitian GP);
  magnitude near `|C|` ⇒ rectilinear (de-rotate → single real GP).

---

## 8. Kernel choice for the signal (superposition of sinusoids)

Designing the kernel = designing the PSD (Bochner). For a superposition of tones with
different frequencies/amplitudes:
- **Spectral mixture kernel** (Wilson & Adams 2013): PSD = sum of Gaussian bumps →
  `k(τ) = Σ_q w_q exp(-2π²v_q τ²) cos(2π μ_q τ)`. Each component: frequency `μ_q`, power `w_q`,
  bandwidth/coherence `v_q`. Initialize from the empirical periodogram (the marginal
  likelihood is multimodal; kernel learning ≡ spectral estimation via the Whittle likelihood).
- For 1D / decoupled-mode pieces, **celerite/SHO** terms give O(N) solves (Foreman-Mackey).
- A single exp-sine-squared periodic kernel = one fundamental + all harmonics (not one tone) —
  fine only if the tones are harmonically related.
- Sharp spectral lines extrapolate oscillations *through* gaps (vs. Matérn reverting to mean).

---

## 9. Implementation checklist

1. [ ] Operator builders: stationary cores `P`, `G` (parametrized kernels); envelope `h`
       (smooth, low-parameter); noise field `Nvar = U S Vᵀ`.
2. [ ] Matrix-free matvec `T @ Xmat @ F` and the `HCH` pre/post-scaling.
3. [ ] Path (i): `KroneckerProductLinearOperator + DiagLinearOperator(d̃)`; marginal
       likelihood via `inv_quad_logdet` + the `2 Σ log|h|` Jacobian term.
4. [ ] Structured preconditioner `(P⊗G + σ̄²I)⁻¹` (eigendecomposition projector) — inject it;
       do not rely on default pivoted-Cholesky.
5. [ ] Conformance test: small-N comparison of structured+iterative vs. exact Cholesky
       (`fast_computations(... =False)` / raise `max_cholesky_size`). Check solve residual,
       log-det vs. `slogdet`, and gradients (finite-difference).
6. [ ] Hyperparameter opt: profile scale/noise; Fisher step (scenario i) or
       freeze-linearize surrogate (scenario ii); fix probe vectors near the optimum.
7. [ ] Path (ii) only if cores genuinely drift: `SumKroneckerLinearOperator` (verify ctor),
       bulk-term preconditioner, SLQ log-det.
8. [ ] Complex data: estimate `C̃`; if rectilinear, de-rotate to a single real GP and treat
       the imaginary channel as a noise reference.
9. [ ] Predictive variances: opt into LOVE (`fast_pred_var`) only if needed.

---

## 10. References

- Saatçi (2011 thesis) — Kronecker / grid GPs.
- Gardner, Pleiss, Bindel, Weinberger, Wilson (2018) — GPyTorch / BBMM (mBCG, SLQ log-det).
- Pleiss et al. (2018) — LOVE (fast predictive variances).
- Wilson & Adams (2013) — spectral mixture kernels.
- Álvarez, Rosasco, Lawrence (2012) — kernels for vector-valued functions (LMC / coregionalization).
- Gibbs (1997); Paciorek & Schervish (2004) — nonstationary kernels (lengthscale variation).
- Foreman-Mackey et al. (2017) — celerite (O(N) quasiseparable 1D).
- Schreier & Scharf, *Statistical Signal Processing of Complex-Valued Data* — proper/improper,
  augmented/widely-linear, rectilinear signals.
