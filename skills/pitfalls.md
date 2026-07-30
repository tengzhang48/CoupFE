# CoupFE pitfalls

Codified, broken-control-tested failure modes. Each is something a plausible,
confident implementation gets wrong and that ordinary tests (compile, smoke,
CS-vs-FD consistency) do **not** catch. This file evolves with the code.

## Complex-step safety (the differentiable path)

The residual is complex-stepped, so it must be **analytic** in the perturbed
variables. The compiler/author must avoid, in the differentiable path:

- `abs(z)`, `max`, `min`, `sign`, clipping, saturation on a perturbed value;
- `real(z)` extraction before the residual is complete;
- comparisons of a complex perturbation; a branch whose active side changes under
  the perturbation (freeze the branch from the **real** base point and replay it);
- a **conjugating** contraction. In Fortran `dot_product` conjugates its first
  complex argument — use an explicit `sum(a*b)`. The analytic continuation of
  `sqrt(x·x)` is `sum(z*z)`, never `sum(conjg(z)*z)`.
- eigen/SVD/polar routines that use conjugation; repeated eigenvalues.

A passing CS-vs-FD check does **not** prove analyticity — it differentiates the
wrong-but-consistent code faithfully. Gate with an independent oracle.

## Spectral functions — use ITERATION, not `eig` (the matrix-backend rule)

For matrix functions (`logm`, `sqrtm`, `expm`, `polar`) **use the iterative matrix
backend (`matrix_backend='iterative'`) as the default.** It never forms
eigenvectors, so it is **robust to repeated eigenvalues and complex-step-safe**.
This is why Hencky-strain plasticity (`E = ½ logm(C)`) is fine even at degenerate
states.

- **The `eig` backend leaks complex-step perturbations at near-diagonal states.**
  The exact bug found in the lab's Anand rock model: the `eig33z` near-diagonal
  guard returned `V = I` for the tiny CS off-diagonals, **zeroing the shear block
  of the tangent** (a 1.2% discrepancy that *looked* like constitutive
  eigenvector ambiguity but was a code leak). Lowering the guard threshold only
  moves the failure — the robust fix is the **iterative backend**. Use
  `matrix_backend="eig"` only *deliberately* (compatibility / a controlled
  performance comparison).
- **Eigenvector degeneracy at repeated eigenvalues is a constitutive ambiguity,
  not a code bug — fix at the model level, not in `eig`.** For models that
  genuinely need principal *directions* (e.g. spectral slip systems), any
  orthonormal basis of the degenerate eigenspace is valid → different response.
  So `tensor.eig` reconstruction at non-diagonal repeated eigenvalues is
  *intentionally* a limitation (an `xfail`) — reach for the iterative matrix
  function instead.
- **Test whether `eig` actually misbehaves at the failing state before blaming
  degeneracy.** In one session the degenerate states were finite in both `eig`
  and `eig33z`; the NaN was a return-map issue elsewhere. Confirm the culprit at
  the exact state.
- **The framework verify's hardcoded FD step (`1e-7`) is too small for spectral
  elements — it false-fails a *correct* complex-step tangent.** Iterative
  `logm`/`expm`/`polar` carry an internal noise floor (~1e-12); a `1e-7` central
  FD amplifies it to ~1e-5, so `run_verification` reports rel ~7e-6 and fails.
  The CS tangent (what the kernel uses) is exact: a FD-step sweep is a textbook
  **U-curve** bottoming at rel ~1e-8 around `eps ≈ 3e-5`. **Diagnose the U-curve;
  do not loosen the tolerance.** Until the framework exposes an `fd_eps` knob,
  verify locally at the resolvable step. The private finite-strain-J2 case that
  exposed this issue is withheld from the public artifact pending provenance.

The CoupFE regression tests that mark this limitation use the wording:

> "eig reconstruction at non-diagonal repeated eigenvalues is intentionally a
> limitation — use the iterative matrix backend; degeneracy is constitutive, not
> a code bug."

## Code-generation and weak-form traps

These traps were found in `abaqus_ufl_lab` and are carried into the CoupFE
`coupfe.codegen` port. They are **code-generation** failures: `verify()` and
compile may pass while the generated `.for` is wrong.

### `z**2` NaN-at-0 trap

`z**2` (integer power) codegens to complex `z ** DCMPLX(2.0)` = `exp(2*log z)`,
which is **NaN at `z = 0`** (`log 0 = -inf`). For a square of any quantity that
can be zero — e.g. `(X[0] - R0)**2` where a Gauss point may sit at `R0` — write
the explicit product `dx*dx`, not `dx**2`. `verify()` and a direct probe at a
generic state will NOT catch this; only an input where the base is exactly 0
(a centroid Gauss point, a symmetric coordinate) trips it. See
`tests/test_z2_nan_at_zero.py`.

### Vector `state_vars` miscompiles — carry orientation as 3×3 structural tensors

A length-3 **vector** `state_vars` entry (e.g. a fiber direction `f0=[1,0,0]`) is
traced as a **scalar**: the emitted Fortran does `f0 = f0_old` with a rank-0 LHS
and a rank-1 RHS → `Incompatible ranks 0 and 1 in assignment`, and any
`dyad(f0,f0)` / `f0 @ C @ f0` downstream → `Inconsistent ranks`. `verify()` (pure
Python, where numpy vectors work) does **not** catch it — only the kernel build does.
**Fix: store orientation as 3×3 STRUCTURAL TENSORS** (the working `Fp`-matrix path):
keep `ff = f0⊗f0`, `fssym = sym(f0⊗s0)` as `np.zeros((3,3))` state; take invariants by
contraction (`I4f = trace(C@ff)`) and stresses directly (`S = Ta*ff`); return them
**unchanged** from `stress_PK1` so they survive `commit`. (A non-symmetric tensor state
must round-trip column-major — see the STATEV-layout rule below.)

### Coupled-material codegen gotchas (fields + state) — all `verify()`-blind

Found building the mixed u-p element; each passes a Python `verify()` but breaks the
**generated Fortran**:
- **Fortran is CASE-INSENSITIVE.** A local var collides with a same-letter field/state —
  naming the PK1 stress `P` clashes with the pressure field `p`
  (`Symbol 'p' already has basic type`). Rename the local (`PK1`); likewise avoid `J`/`j`,
  `N`/`n` clashes.
- **Put FIELDS right after `F`, before the state vars, in the material-method signature.**
  `stress_PK1(self, F, p, ff_old, …, dt)` compiles; moving `p` after the state vars makes
  the generated subroutine's definition and call-site disagree on arg rank
  (`Rank mismatch in argument 'p'`).
- **`verify`'s FD comparison FALSE-FAILS on field/param derivatives at extreme scales.**
  `d(stress)/dp = J·F⁻ᵀ` is exact but FD cancels at a large field value (`p~1e3`);
  `d(resid)/dp = −1/κ` has a tiny norm (`κ=1e6`) so its *relative* error inflates. The
  complex-step kernel tangent is exact — pass `tol=1e-5` or a custom CS-vs-analytic check,
  don't chase it (same family as the spectral-element FD false-fail above).

### F-bar `(J̄/J)^(1/d)` NaN-at-J̄≤0 trap (a missing FEASIBILITY guard)

F-bar rescales by `alpha = (J̄/J)^(1/d)` with `J̄` the **centroid** Jacobian. A fractional power of a
possibly-**negative** base is a silent NaN: if the centroid `J̄` crosses 0 (centroid inversion under
bending/shear) → `(negative)^(1/2)` → **NaN** — *even when every Gauss-point J is still positive*. It is
**F-bar-specific** (a standard element uses each GP's own J and can't hit the centroid mode — verified: the
standard element ran clean where F-bar NaN'd). The exponent is correct; what's missing is a **feasibility
guard**. Fix: a **`J>0` / `J̄>0` line search** (prevent the Newton step from driving `J̄≤0`; the **bulk
analogue of the contact CCD**) or, interim, a NaN-safe fallback (detect `J̄≤0` → signal a step cut). General
rule: **contact and bulk both need feasibility guards, not just energies** — contact keeps `gap>0`
(CCD/ACCD), bulk keeps `J>0` (J-bound line search / strain limiting); ppf has both, which is why it's robust
and an energy-only port NaNs at the boundary. **Diagnostic:** a NaN whose residual localizes to the bulk
batch (a `bulk_nan` split in the assemble) + an element with `J̄≤0` = this mode. **Localize the residual
(bulk vs contact) BEFORE sweeping loads/BCs** — one run says which operator; don't sweep lid/gravity/walls/ν.
(2026-06-26 lesson.)

### Finite-strain tensor-return shape trap

When a finite-strain UMAT/UEL material method returns a tensor, prefer assigning
`det(F)` to a scalar temporary before using it in the returned tensor expression:

```python
# Historically risky in finite-strain UMAT/UEL generation
return G * (F - inv(F).T) + K * log(det(F)) * inv(F).T
```

This formula is mathematically correct. It historically exposed a
component-wise tensor-return codegen bug that emitted invalid Fortran like:

```fortran
LOG(det33z(F(ii,jj)))
```

`det33z` needs the full `3x3` tensor, not a scalar component. The runtime symptom
is often an all-NaN element residual/tangent even for a simple undeformed element.

Use explicit scalar/tensor temporaries for readability and easier generated-code
inspection:

```python
J = det(F)
FinvT = inv(F).T
return G * (F - FinvT) + K * log(J) * FinvT
```

Regression tests now cover the direct-return `det(F)` case, but after
`generate_element(...)` / `generate_uel(...)` it is still worth grepping the
`.for` file for impossible patterns such as `det33z(F(ii,jj))`. If any matrix
helper receives a component-indexed tensor argument, treat that as a
 code-generation shape bug, not a mechanics or solver-convergence issue.

### `sym` is unsupported in generated code

`coupfe.codegen.core.tensor.sym` exists for the Python reference path, but the
Fortran translator does **not** support it. An unsupported name raises
"Unsupported function: sym" at generate time (`verify()` is unaffected). Write
`0.5 * (A + A.T)` instead.

The supported tensor helpers for generated methods are: `det`, `inv`, `log`,
`exp`, `sqrt`, `trace`, `dev`, `eye`, `dyad`, `sym3`, `sqrtm`, `logm`, `expm`.
`tanh` is also unsupported — build it from `exp` if you need it.

### `self._helper()` — branch on `.real`

A `self._helper()` is translated to its own Fortran subroutine. The helper body
must use translator-supported idioms: branch with **statement-form**
`if x.real > 0.0:` (maps to `IF (DBLE(x) .GT. 0)`), not a `hasattr`/ternary;
Python floats already carry `.real`.

- Branch on `.real` when a branch is unavoidable. Do not branch on the imaginary
  perturbation.
- A branch on `.real` is **invariant** under the imaginary complex-step
  perturbation, so an elastic/plastic (yield) switch is *not* a complex-step
  discontinuity — do not blame the branch for a tangent NaN.
- A shared helper is emitted once; if it is called from both a tensor-context and
  a scalar-context with the same argument, the translator may see multiple
  argument-kind signatures. Keep helpers type-consistent.

### State-variable rules for UEL `stress_PK1`

Three rules, each a real compile/codegen failure found building stateful
elements:

1. **Return the new state as a dict literal in the `return`:**
   `return P, {'ep': ep_new, 'Fp': Fp_new}`. Assigning it first
   (`state = {...}; return P, state`) makes the translator hit a bare `Dict`
   statement → "Unsupported expression: Dict".
2. **`stress_PK1` must list every declared state var's `_old` in its Python
   signature** — even one it recomputes and never reads (e.g. `hydro_old` for a
   `sigma_h` state). The codegen adds all state `_old` args to the subroutine; a
   missing one is passed REAL into a COMPLEX arg → "Type mismatch ... passed
   REAL(8) to COMPLEX(8)" at compile, not at generate.
3. **A tensor state var (`eps_p`, `Fp`) is typed as a tensor only inside
   `stress_PK1`; in any other method it is mis-inferred as scalar.** A
   `self._helper()` called from both `stress_PK1` and a scalar-equation method
   with that tensor arg fails: "called with multiple argument kind signatures".
   Work around by computing the needed scalar in `stress_PK1` and passing it to
   the scalar equation as a scalar state var. This also keeps the J2 return map
   out of the scalar equation.
4. **Tensor state is stored COLUMN-MAJOR in the flat per-GP STATEV** — the kernel
   reads/writes `idx = offset + j*3 + i` (`umat_gen`). Pack/unpack with
   `ravel('F')` / `reshape((3,3), order='F')`, and the reference oracle
   (`reference_assembly`) must match. **A symmetric tensor state (`epsp`, `Cp`,
   strain) is order-invariant (`M == Mᵀ`), so a layout mismatch is invisible — it
   passes every test.** The trip condition is a **conjunction**: declared as a 3×3 **tensor** AND
   **non-symmetric in value**. `lce_quad4`'s `Fv` is non-symmetric (`Fv12 ≠ Fv21`) but stored as
   **scalars** (`Fv11..Fv21`) → skips the reshape, dodges it too. `Fp` is the first with both: the
   oracle reads the transpose → ~1e-4 stress error (machine-zero in the *committed state*, which
   hides it). Real latent `reference_assembly` bug, fixed 2026-06-25 when `j2_fefp_uel` became the
   first element to exercise the path. **Gate any stored-tensor element with a non-symmetric
   `Fp_old`** (a diagonal/symmetric one — or a scalar-packed tensor — is a vacuous test of the layout).

### Native vs Abaqus sign convention

CoupFE now has both backends:

- **Native** (`generate_element(..., backend='native')`): the emitted subroutine
  is `coupfe_element_rk`. It returns the weak-form residual `R` and the
  consistent tangent `K = ∂R/∂u`. Newton solves `K δu = −R`. This is the
  standalone CoupFE path.
- **Abaqus UEL** (`generate_uel(...)`, an alias for
  `generate_element(..., backend='abaqus_uel')`): the emitted subroutine is
  `UEL`. Abaqus defines `RHS = −R` and `AMATRX = −dRHS/dU`, so the generated
  code assembles into `RHS` and `AMATRX`. This is the Abaqus-export path.

`CompiledElement` consumes either kernel and presents the same `(R, K)` sign
convention to the rest of CoupFE. When you read a generated `.for`, know which
backend it came from: `R(row) = ...` is native; `RHS(row,1) = ...` is Abaqus.
Do not mix the two sign conventions in a single mental model.

### Coupled-field / transport flux sign trap (the `OperatorSignWarning` rule)

For scalar UEL equations the generated tuple convention is:

```text
R += storage * N - flux . grad(N)
```

Here `flux` is a code-generation name for the coefficient of `grad(test)`, not
necessarily the physical flux. The shortest safe pipeline is:

1. write the weak-form term you want;
2. match it to `storage * test - flux . grad(test)`;
3. return that coefficient directly.

If the paper gives only a strong form, derive the weak form once and then use
the same direct matching. Avoid translating through "physical flux" and then
translating again into the tuple API; that extra mapping is where most sign
mistakes have entered.

Examples:

- Ordinary diffusion with `storage = c_dot`: return the physical flux
  `-D * grad_c` as the code `flux`.
- AT2 damage with `storage = Gc/ell*d - 2*(1-d)*H`: the desired weak form
  contains `+Gc*ell*grad(d).grad(test)`, so the generated method must return
  `flux = -self.Gc * self.ell * grad_d`.

`OperatorSignWarning` is promoted to an error in `pyproject.toml`. A flipped
flux makes the single-field operator indefinite at the resolving scale; the
warning catches it during `verify()`. It is a screening tool, not a proof — add a
non-homogeneous gradient-sign check for every new damage/phase-field/transport
UEL. See `tests/test_phase_flux_sign_trap.py`.

## State protocol

- Every residual/tangent evaluation starts from `state_committed`.
- No complex-step column may see the state produced by a previous column.
- Line-search trial residuals never commit.
- Commit **once**, by a final real evaluation, after the global solve accepts.
- A tangent that depends on evaluation order = a leaked-state bug.

## Coupled convergence gate (the "converges but wrong transient" trap)

A coupled solver (u-μ, u-A, u-T, u-c-φ) can converge every step yet drift from the
reference because a single global ‖R‖ is **momentum-dominated** (the transport
block coefficients are ~1e-13 in SI). The Newton stops ~2 iters early on the weak
field; the under-resolution **compounds** (matches early, drifts late).

- **Fingerprint:** early match, late drift. Physics/BC errors diverge from t=0.
- A tiny `|R_weak|` is **not** convergence when its coefficients are tiny — divide
  by the block scale to get the *solution* error.
- **Fix:** field-wise convergence — gate each field on its own characteristic
  scale (what Abaqus does by default via `*CONTROLS` Ra/Ca).
- **Iter count discriminates:** 3–5 iters/step ⇒ the tangent is fine, the gate is
  the bug; 15–40 ⇒ the tangent converges slowly.
- For **iterative** solvers, also non-dimensionalize the weak form (e.g. μ by RT;
  the per-field diagonal ratio can be ~1e17 and is fatal to CG/GMRES) — derived
  scales, never a magic constant.

## Block-scale disparity (catch it before you solve)

Before running a coupled transient, check the per-field block-scale disparity off
the element tangent alone (the `assert_coupled_field_scale_balance` gate, to be
migrated with the harness). It fires on a badly-scaled operator (e.g. a gel's
~5e17 u-vs-μ ratio) **before** a multi-day under-resolution hunt.

## Reverting your OWN debug edit in a shared tree sweeps another agent's WIP

`git checkout <file>` / `git restore <file>` is **whole-file** — it cannot tell
your temporary instrumentation from a co-worker's uncommitted changes in the same
file. A file-level restore once discarded another agent's uncommitted research
work; it was not recoverable because `git fsck --lost-found` can find staged or
committed blobs, not arbitrary working-tree text. **Revert your own temporary edits
surgically by deleting exactly your lines**, never with a whole-file checkout,
when another agent may have work in progress in the tree.

## Contact (a separate operator, current configuration)

- **Do not complex-step through the active-set switch / search.** Freeze the active set,
  the closest feature, and stick/slip from the **real** iterate; complex-step (or use an
  analytic tangent) only the *smooth* branch. Complex-stepping the `min`/`if g<0`/
  closest-feature logic gives invalid or zero derivatives.
- **Derive from the energy, so signs are unambiguous.** Convention: gap `> 0` = separated
  (no force), `< 0` = penetration; the normal points **out of the obstacle into the body**;
  the contact residual is the energy gradient `∂Π_c/∂u`. Gate equal-and-opposite reaction.
- **N-body node-to-segment contact needs CONSISTENT edge winding per body.** The 2D barrier's
  signed gap `d=(e×r)/L` is `>0` only when the secondary is on the **left** of edge `a→b`, so
  every body's boundary loop must be oriented the same way (outside-on-left). It's a *two-body*
  operator (one oriented primary surface); use it for **N-body mutual contact** (union of all
  bodies' edges) and an inconsistently-wound edge gives a neighbour node a **negative** gap ⇒
  the barrier reads a *deep penetration* ⇒ a huge **spurious force at the undeformed rest
  state** ⇒ divergence. **Diagnose with `|R_contact|` at U=0 (must be 0).** Orient the edges
  (`cross(b−a, mid−centroid)>0`); the opt-in `body_id=` (exclude all same-body edges, not just
  the 1-ring) is the belt-and-suspenders. (`tests/test_multibody_contact.py`.)
- **Barrier/penalty stiffness from a POSITIVE proxy**, never the raw (possibly indefinite)
  bulk tangent — a material-only / positive-projected `K_proxy`.
- **A barrier needs CCD** — a Newton step can cross the surface despite the barrier. Plain
  penalty tolerates a small penetration ~ load/k (k→∞ ⇒ penetration→0).
- **`ppf_norm=True` changes what `κ` means; `elastic_op` completes the recipe.** With the
  default cubic, `κ` is a shape coefficient: the force is `−κ(d̂−d)² n`. With `ppf_norm=True`
  the shape carries `2/d̂`, so `κ` is a true stiffness (force/length). To match ppf fully,
  pass the bulk operator(s) as `elastic_op` so `s` includes `nᵀK_elastn` on top of `κ+M/d²`.
  Do not blindly reuse the same numerical `κ` when toggling the flag.
- **Use `kinematic=True` for prescribed-motion obstacles.** The effective barrier gap is
  floored at `constraint_tol*dhat`, keeping the repulsive force bounded and preventing the
  Newton solve from sticking to a moving wall (the ppf `kinematic` floor/sphere flag).
- **`thickness` turns an obstacle into a pass-through shell.** Nodes that have penetrated
  deeper than `thickness` are ignored, so a body can tunnel through instead of being trapped
  by a deep-penetration barrier force.
- **Feasibility is PAIRED: contact `gap>0` (CCD) ⟷ bulk `J>0`.** A contact CCD alone is not enough — the
  same Newton step can invert a bulk element (F-bar NaNs at centroid `J̄≤0`, see the codegen traps). An
  implicit contact solve needs *both* feasibility line searches (CCD + a J-bound / strain limit), the way
  ppf pairs ACCD with strain limiting. Energy (barrier / element) ≠ feasibility.
- **Contact is current-config** (gap/normal/projection are spatial) even though the bulk is
  reference/total-Lagrangian — the contract lets them coexist; never convert the bulk.
- **Distributed contact ≠ the bulk halo** (spatial proximity, dynamic sparsity, global CCD
  min-reduce) — keep the production distributed solver contact-extensible (separable
  `K_bulk`/`K_contact`, or matrix-free; see `docs/dev/contact.md`).

## Validation, not consistency

Consistency checks (CS-vs-FD, compile, smoke) are blind to convention errors, a
wrong-but-consistent residual, and anti-diffusive sign flips. Gate physical
properties with **independent** oracles: complex-step vs analytic tangent,
block-definiteness / diffusive-flux sign, patch/energy tests, cross-backend
(Abaqus) parity. A check is only trusted once it **fails** on a reintroduced bug
(a "broken control").

## Contact-benchmark traps

- **A small contact patch is starved on a uniform mesh.** The Hertz patch (`b~2.5` on an
  `L=50` block) gets only a few nodes uniformly, so the contact stiffness and load **drift
  with refinement** (don't converge). **Grade the mesh** at the contact (`sinh`,
  `parabolic_block_mesh(beta=...)`); load/half-width/pressure then converge. It fails
  *silently* as a non-converging load, not an error.
- **`max|x|` over active nodes over-reads the half-width** — it includes near-edge nodes with
  spurious ~0 reactions, and the bias *grows* with refinement. Fit the semi-ellipse
  (`fit_semiellipse`: `p²` linear in `x²`) instead, and check its R².
- **Path-dependent friction needs the oracle's loading history.** If the analytic
  solution assumes full normal loading followed by tangential loading, reproduce
  that sequence incrementally; a one-shot solve is a different experiment.
- **Compare like with like.** Match material, interface, and dimensional assumptions
  to the oracle before calling a stable deviation a bug. Use a controlled parameter
  sweep to test the proposed cause instead of widening the tolerance.
- **`np.broadcast_to(scalar, (nc,))` is a stride-0, size-1-buffer view** — passed to compiled
  code that reads `nc` contiguous doubles, it reads out of bounds. Use a contiguous array
  (`np.ascontiguousarray(...)`).

## Adaptive dynamics + barrier + moving obstacle: six failure modes from `ring_compress`

The dynamic-relaxation version of the ring compression example
(`examples/ring_compress/reproduce_dynamics.py`) hit a cluster of interacting
failure modes that are easy to reproduce in any barrier-dynamics problem with a
moving rigid obstacle. The example ships as RESEARCH; its historical external
comparison lacks a retained raw run and is not release-validation evidence.

1. **`maxit` must be passed explicitly to `newton_solve` inside `solve_dynamics_adaptive`.**
   The driver accepts `maxit` but originally forwarded only `**newton_kw`; `newton_solve`
   then used its default `maxit=60`. The step statistics looked fine, but the barrier
   case silently did far more work than intended and the cost exploded. **Fix:**
   `newton_solve(..., maxit=maxit, **newton_kw)`.

2. **A moving obstacle must be visible to the predictor CCD over the whole step, not
   only at the end position.** The predictor CCD runs *before* the first residual of the
   step. Updating `obs.p = motion(t)` inside `max_step` is better than nothing, but it still
   evaluates the gap at a single end position — a node can tunnel mid-step and end
   separated. **Fix:** use a time-aware obstacle (`MovingHalfSpace(position(t), normal)` /
   `MovingSphere(center(t), R)`). `_call_max_step` now forwards both `t` and `dt`; the
   barrier's `max_step` evaluates the obstacle at `t − dt + α·dt` during the safety
   bisection, bounding the actual relative motion. The `_MovingObstacleContact` wrapper is
   kept only to update `obs.p` for the penalty path.

3. **Barrier contact nodes must start with `gap > 0`.** If any contact node lies exactly
   on the obstacle at `t=0`, the CCD bound is zero on the first step and `M/d²` is
   evaluated at `d≈0`. The Newton step freezes (`alpha ≈ 2⁻³⁰`) and the force overflows.
   **Fix:** offset both plates by `±dhat` in the barrier setup so every node starts
   separated.

4. **`M/d²` must be regularized for tiny positive gaps.** The CCD bound keeps `d>0`, but
   it can let `d` approach machine precision. A raw `mass / d²` term overflows and
   poisons the residual/tangent. **Fix:** floor `d²` in `rigid_barrier_eval`
   (`np.maximum(d², 1e-24)`), both in the normal force and in the friction stiffness.

5. **The CCD bound itself must not collapse to zero.** When a node is already very close
   to the obstacle, the bisection safety net in `max_step` can back `alpha` down to
   underflow. **Fix:** introduce a `gap_floor` in the CCD estimate and safety net, and
   clamp the returned `alpha` to a tiny positive floor in the driver.

6. **Dynamic relaxation needs the STRUCTURAL TIMESCALE, a HOLD, and a KE-gated sample —
   not a nominally "slow" ramp.** (This point previously blamed
   `solve_dynamics` itself; that was wrong — see the dated investigation in
   `docs/lessons_learned.md`.) **Fix:** compute `omega_1` first; ramp in stages;
   hold with damping selected from that timescale; sample only after a stated
   kinetic-energy threshold. At a settled state `v -> 0`, the `αM·v` force
   vanishes. Any external comparison still requires the missing input,
   environment, raw output, and rerun gate.

## Fixed load increments with penalty contact can follow a collapse branch

A quasistatic compression problem that converges with Abaqus may collapse in CoupFE if you
use `solve_increments` with a **fixed, large load increment**. The penalty normal force is
small (proportional to penetration), so the return-map friction cap is small; a large
increment lets the contact nodes slide along the obstacle and the body folds onto an
asymmetric, low-reaction path. The symptom is a tiny active contact set and a reaction near
zero.

**Don't fix it by adding symmetry BCs** — that masks the problem and changes the physics.
The right fix is **adaptive load stepping**: cut the increment when Newton stalls and grow it
when convergence is easy. This mirrors the increment-control idea without
claiming cross-engine parity. See `examples/ring_compress/reproduce.py` and
`skills/contact.md` (`RigidContact` quasistatic compression section).

## A joint (R, K) kernel must not be evaluated through two separate operator calls

The compiled element kernel (`element_rk_batch`, driving the generated `SUBROUTINE UEL`)
returns the residual AND the tangent in ONE call — exactly like the Abaqus `UEL` returns
`RHS` and `AMATRX` together (`abaqus_ufl.fe` follows this: `assemble` → `R, K` in one pass).
CoupFE's `Operator` contract splits evaluation into `residual()` and `tangent()`, and Newton
calls both at the same `U` — so a naive implementation runs the kernel TWICE per iteration
and discards half each time (measured ~40% of a compiled-element solve's runtime).

`ElementGroup` fixes this with a one-entry `_rk` cache (`fuse_rk=True` by default): the paired
residual/tangent at one iterate share a single kernel call. Correctness guards that make it
safe to leave ON: the key includes `props` (callers mutate them per step, e.g. active stress),
`commit()` clears the cache (committed-state change), and line-search trials at `U+αdU` miss.
**Broken control:** `tests/test_element_group.py::test_rk_fusion_invalidates_on_prop_change`
asserts a prop change between residual and tangent does not serve a stale K.

Turn fusion OFF (`fuse_rk=False` / `COUPFE_FUSE_RK=0`) only for schemes that deliberately
mutate props/state BETWEEN a residual and its paired tangent, residual-only/matrix-free loops,
or debugging. If you write a NEW driver, prefer evaluating R and K together (one kernel call)
over a residual-then-tangent pair — the joint evaluation is the kernel's native shape.
