"""Optional ppf-contact-solver side of a box-on-floor rerun recipe.

This is the upstream-side runner paired with `coupfe_box_on_floor.py`. It needs the
ppf-contact-solver (ZOZO, Apache-2.0) built + a CUDA GPU, so it is NOT part of
the CoupFE test suite. Run it from the ppf
repo root::

    PYTHONPATH=/path/to/ppf-contact-solver python ppf_reference.py <theta_deg> <mu> [frames]

A tet box rests on an invisible wall (rigid half-space) under gravity tilted by
θ, with ppf smoothed friction `μ`.  Reports the box centroid x (slide) and the
minimum vertex y (penetration).

No upstream output/environment is retained in this repository. Capture the
exact ppf revision, CUDA stack, command, parameters, and output together with a
CoupFE rerun before citing a comparison.
"""
import sys

import numpy as np

from frontend import App  # ppf frontend (run from the ppf repo root)


def run(theta_deg, mu, frames=120):
    app = App.create(f"cf_xcheck_{int(theta_deg)}_{int(mu*100)}")
    V, F, T = app.mesh.tet_box(1.0, 1.0, 1.0)
    app.asset.add.tet("box", V, F, T)
    scene = app.scene.create()
    scene.add.invisible.wall([0, 0, 0], [0, 1, 0])          # rigid floor y=0
    box = scene.add("box").at(0.0, 0.55, 0.0)               # bottom ~0.05 above floor
    box.param.set("model", "snhk").set("poiss-rat", 0.45).set("friction", mu)
    scene = scene.build()
    session = app.session.create(scene)
    th = np.radians(theta_deg)
    g = 9.8
    (session.param.set("dt", 0.01).set("frames", frames)
     .set("gravity", [g*np.sin(th), -g*np.cos(th), 0.0]).set("friction-mode", "max"))
    session = session.build()
    session.start(blocking=True)
    verts, frame = session.get.vertex()
    cx = float(verts[:, 0].mean())
    min_y = float(verts[:, 1].min())
    print(f"RESULT theta={theta_deg:.0f} mu={mu} frame={frame} "
          f"centroid_x={cx:+.4f} min_y={min_y:+.4f}")
    return cx, min_y


if __name__ == "__main__":
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    run(theta, mu, frames)
