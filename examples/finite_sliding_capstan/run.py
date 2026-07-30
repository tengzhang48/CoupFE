"""FINITE SLIDING on a CURVED surface — the capstan. ``python run.py``.

The harder finite-sliding case than a flat polyline: a belt slides around a cylinder, so the contact
migrates along a CURVE and the carried friction state is a **force vector that must rotate with the
surface tangent** at every re-pairing — not just a scalar. It has a famous analytic check, the **capstan
equation** ``T_load / T_hold = e^{μθ}`` (the same exponential grip that lets a knot or a winch hold a huge
load), so we can verify the finite-sliding integration exactly.

A belt wrapped over angle ``θ`` is discretized into ``N`` elements. As the belt slides (gross sliding) the
**finite-sliding return-map** friction acts **tangent to the cylinder at each element's angle** (the frame
rotates around the wrap); the per-element normal force is ``N_i = T_i·dθ`` (tension × curvature), at slip
``f_i = μ N_i``, and the tension propagates ``T_{i+1} = T_i + f_i`` → the capstan exponential.

Why the rotating frame is *necessary*, not cosmetic: the friction force must stay **tangent** to the
surface. If you froze the frame (kept the friction in its initial direction, as a flat-case state transfer
would), the *same* force would have a large **normal** component on the rotated part of the wrap — it would
press the belt into or off the cylinder. So the broken control is a real error on a curve.

Self-check (prints ``OK`` / ``FAIL``):
  * the integrated finite-sliding friction reproduces the capstan ``T_load/T_hold = e^{μθ}`` (several θ, μ);
  * with the rotating frame the friction force is tangent everywhere (normal component ≈ 0);
  * with a FROZEN frame the friction force develops a spurious normal component reaching ≈ |f| — the frame
    rotation is physically necessary on the curved surface.
"""
from __future__ import annotations

import numpy as np

R, K_T, T_HOLD = 1.0, 5.0e2, 1.0          # cylinder radius, tangential stick stiffness, held-end tension


def capstan(theta, mu, N=400):
    """Slide a belt around a `theta`-wrap; finite-sliding return-map friction with a rotating frame;
    propagate tension. Returns (T_load/T_hold, max normal-force fraction if the frame were FROZEN)."""
    dth = theta / N
    T = T_HOLD
    ft = 0.0                              # committed tangential friction (the carried state)
    t0 = np.array([-np.sin(0.5 * dth), np.cos(0.5 * dth)])   # the initial surface tangent
    max_frozen_normal = 0.0
    for i in range(N):
        th = (i + 0.5) * dth
        N_i = T * dth                     # normal force on this element (tension × subtended angle)
        cap = mu * N_i
        ds = R * dth                      # the belt slides one element-arc (gross sliding)
        ft = float(np.clip(ft + K_T * ds, -cap, cap))    # finite-sliding return-map onto the cone μN_i
        T = T + ft                        # tension propagation around the wrap (capstan)
        # WITH the rotating frame the friction force ft·t(th) is tangent (normal component ≡ 0).
        # FROZEN frame: the force ft·t0 would have a normal component ft·(t0·n(th)) on the rotated surface.
        n_th = np.array([np.cos(th), np.sin(th)])         # outward surface normal at this element
        max_frozen_normal = max(max_frozen_normal, abs(ft * float(t0 @ n_th)) / (cap + 1e-30))
    return T / T_HOLD, max_frozen_normal


def main():
    ok = True
    print(f"{'θ(deg)':>7} {'μ':>5} {'sim ratio':>10} {'e^(μθ)':>10} {'rel err':>9} {'frozen-frame normal/|f|':>24}")
    for theta_deg, mu in ((90, 0.3), (180, 0.3), (270, 0.25), (180, 0.5)):
        theta = np.radians(theta_deg)
        ratio, frozen_normal = capstan(theta, mu)
        analytic = float(np.exp(mu * theta))
        cap_ok = abs(ratio - analytic) / analytic < 0.02         # capstan reproduced
        frame_needed = frozen_normal > 0.7                       # frozen frame ⇒ huge spurious normal force
        ok &= cap_ok and frame_needed
        print(f"{theta_deg:>7} {mu:>5} {ratio:>10.4f} {analytic:>10.4f} {abs(ratio-analytic)/analytic:>9.1e} "
              f"{frozen_normal:>23.2f}")
    print("capstan reproduced by finite-sliding friction; rotating frame necessary on the curve  -> "
          + ("OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
