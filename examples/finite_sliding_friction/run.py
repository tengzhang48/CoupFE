"""FINITE-SLIDING friction = interface plasticity integrated over a re-pairing path. ``python run.py``.

Friction is J2-like plasticity at the interface (cone = yield, slip = plastic flow), so it handles
ARBITRARILY LARGE slip via incremental return-mapping — there is no small-sliding limit in the constitutive
law. "Small sliding" lives only in the contact KINEMATICS (a frozen pairing). For large slip the contact
point MIGRATES across primary edges, so we re-pair between steps and **carry the friction STATE** — the
committed tangential force / accumulated slip, the exact analog of effective plastic strain ``ε_p``.

This demonstrates that the **state transfer at re-pairing is necessary**: a node is pressed onto a primary
surface of ``M`` edges and dragged a distance ``L`` ≫ one edge (so it crosses many edges).
  * WITH transfer — ``f_t`` carries to the new edge → exact Coulomb gross sliding → dissipation ``= μN·L``.
  * WITHOUT transfer — ``f_t`` resets at every crossing → a spurious re-stick at each edge → UNDER-dissipates.
And reversing the drag makes the interface STICK first (the carried ``f_t`` must be overcome) before sliding
back — the hysteresis that only exists because the state is a memory variable.

Self-check (prints ``OK`` / ``FAIL``):
  * large slip works: WITH transfer the dissipation over the long slide equals ``μN·L`` (gross sliding);
  * the friction FORCE stays on the cone WITH transfer, but drops to ~0 at every edge crossing WITHOUT it
    (a spurious re-stick — the force memory is lost). The integrated dissipation barely notices at stiff
    ``k_t`` (the rebuild length ``μN/k_t`` is tiny), but the force does — that is what re-pairing must carry;
  * on reversal the interface sticks (no slip) until the carried force is overcome — a memory effect.
"""
from __future__ import annotations

import numpy as np

L, M = 6.0, 12                       # drag distance, number of primary edges (edge length L/M)
K_T, MU, N_NORMAL = 5.0e1, 0.5, 1.0  # tangential stick stiffness, friction, normal force
CAP = MU * N_NORMAL                   # Coulomb cone radius μN
EDGE = L / M


def drag(path, transfer):
    """Integrate the return-map friction along a tangential `path` (cumulative positions), re-pairing per
    edge. Returns (dissipation, slip history, f_t history). State = committed f_t (the ε_p analog)."""
    ft = 0.0                          # committed tangential force (carried friction state)
    cur_edge = int(path[0] // EDGE)
    diss = 0.0
    ft_hist, slip_hist = [], []
    x_prev = path[0]
    for x in path[1:]:
        edge = int(min(x, L - 1e-9) // EDGE)
        if edge != cur_edge:          # re-pairing: contact point crossed to a new primary edge
            if not transfer:
                ft = 0.0              # WITHOUT transfer → lose the friction memory (wrong)
            cur_edge = edge
        d_slip = x - x_prev                       # tangential slip increment
        ft_trial = ft + K_T * d_slip              # elastic stick predictor
        ft = float(np.clip(ft_trial, -CAP, CAP))  # return-map onto the Coulomb cone
        diss += abs(ft * d_slip)                  # frictional work increment
        ft_hist.append(ft); slip_hist.append(x - path[0])
        x_prev = x
    return diss, np.array(slip_hist), np.array(ft_hist)


def main():
    n = 8000
    fwd = np.linspace(0.0, L, n)                  # drag forward across all M edges (large slide)
    diss_T, _, ft_T = drag(fwd, transfer=True)
    diss_NT, _, ft_NT = drag(fwd, transfer=False)
    ideal = CAP * L                               # exact gross-sliding dissipation μN·L
    settled = n // 4                              # past the initial elastic build-up
    fmin_T = float(np.min(np.abs(ft_T[settled:])))     # force should stay on the cone WITH transfer
    fmin_NT = float(np.min(np.abs(ft_NT[settled:])))   # force drops to ~0 at crossings WITHOUT transfer

    # reversal: drag forward to L/2, then back — with the carried state the interface must STICK first
    half = np.linspace(0.0, L / 2, n // 2)
    back = np.linspace(L / 2, L / 2 - 2 * CAP / K_T, 200)   # reverse by a few stick-lengths
    _, _, ft_r = drag(np.concatenate([half, back]), transfer=True)
    rev = ft_r[len(half) - 1:]                          # the force swings across the full cone (2·CAP)
    sticks_on_reversal = (rev.max() - rev.min()) > 1.5 * CAP and abs(ft_r[-1]) > 0.9 * CAP

    with_diss_ok = abs(diss_T - ideal) / ideal < 0.02            # μN·L (large slip works)
    force_continuous = fmin_T > 0.95 * CAP                       # WITH transfer: stays on the cone
    force_drops = fmin_NT < 0.1 * CAP                            # WITHOUT: spurious re-stick to ~0 at crossings
    ok = with_diss_ok and force_continuous and force_drops and sticks_on_reversal
    print(f"total slide L={L} across M={M} edges  (μN·L = {ideal:.3f}, cone μN = {CAP:.3f})")
    print(f"large slip works: WITH-transfer dissipation = {diss_T:.3f} = μN·L (rel err {abs(diss_T-ideal)/ideal:.1e})")
    print(f"friction force on the cone after build:  WITH transfer min|f_t| = {fmin_T:.3f} (≈ μN)  |  "
          f"WITHOUT min|f_t| = {fmin_NT:.3f} (drops at each crossing)")
    print(f"reversal: interface sticks then re-slips (force swings {rev.max()-rev.min():.3f} ≈ 2μN={2*CAP:.3f}) -> {sticks_on_reversal}")
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
