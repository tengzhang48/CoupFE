"""RESEARCH tire solve and von-Mises post-processing.

This reports peak location and contact-/rim-band stress summaries and writes a
deformed-mesh VTK. No exact GetFEM source, parameters, screenshot provenance,
or retained CoupFE output is available, so the resulting pattern is a local
diagnostic—not an external comparison or validation gate.
"""
from __future__ import annotations

import numpy as np

from examples.tire_contact.run import solve_tire, G, K_BULK, R, R_IN, R_OUT
from examples.tire_contact.mesh import _cross_section_rho
from examples.tire_contact.vonmises import element_vonmises, write_vtk


def main(n_steps=60, grav=None, vtk="examples/tire_contact/tire_vonmises.vtk"):
    kw = {} if grav is None else dict(grav=grav)
    r = solve_tire(n_steps=n_steps, verbose=True, **kw)
    nodes, elems, U, z_ground = r["nodes"], r["elems"], r["U"], r["z_ground"]

    vm = element_vonmises(nodes, elems, U, G, K_BULK)

    pos = nodes + U.reshape(len(nodes), 3)
    cent = pos[elems].mean(axis=1)                      # deformed element centroids
    zc = cent[:, 2]
    # contact region = elements whose deformed centroid is in the bottom band (near the floor)
    contact_band = zc < (z_ground + 0.20)
    # rim region = elements whose (reference) cross-section ρ is near r_in
    rho_c = _cross_section_rho(nodes[elems].mean(axis=1), R)
    rim_band = rho_c < (R_IN + 0.05)

    print(f"  von Mises: max = {vm.max():.4f} at deformed z = {zc[vm.argmax()]:+.3f} "
          f"(floor z = {z_ground:+.3f})")
    print(f"  contact-band mean / global mean = {vm[contact_band].mean():.4f} / {vm.mean():.4f} "
          f"= {vm[contact_band].mean()/max(vm.mean(),1e-30):.2f}x")
    print(f"  rim-band mean     / global mean = {vm[rim_band].mean():.4f} / {vm.mean():.4f} "
          f"= {vm[rim_band].mean()/max(vm.mean(),1e-30):.2f}x")

    write_vtk(vtk, nodes, elems, U, vm)
    print(f"  wrote {vtk}  (deformed tire + von Mises, open in ParaView)")

    # qualitative success: peak near the floor, and the contact band is stressed above average
    peak_at_contact = zc[vm.argmax()] < (z_ground + 0.30)
    concentrates = vm[contact_band].mean() > vm.mean()
    ok = bool(r["min_gap"] > 0 and r["in_contact"] > 0 and peak_at_contact and concentrates)
    print("OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
