"""The CoupFE model-setup pipeline (P) — the declarative front door, end to end.

Run:  PYTHONPATH=. mamba run -n coupfe python examples/model_pipeline/run.py

This is the whole point of P: a real refined-mesh + neo-Hookean + rigid-contact +
load-stepped solve in ~6 declarative lines, instead of hand-wiring the mesh view,
the compiled element, the contact operator, the BC DOF dicts, and the driver.
"""

import numpy as np

from coupfe import Model, NeoHookean
from coupfe.operators.contact import HalfSpace


def main():
    m = Model.structured(4, 4, 1.0, 1.0)          # unit block, 16 Quad4
    m.refine(levels=1)                            # → 64 elements (labels/geometry propagate)
    m.material("block", NeoHookean(G=1.0, K=10.0))
    m.fix("left", x=0.0)                          # roller on the left edge
    m.prescribe("top", y=-0.05)                   # press the top down
    m.contact("bottom", HalfSpace([0.0, -0.01], [0.0, 1.0]), k=1.0e4)

    res = m.solve(steps=4)                        # load-stepped Newton + line search

    ybot = res.position("bottom")[:, 1]
    print(f"elements           : {m.view.n_elem}")
    print(f"DOFs               : {m.view.ndof}")
    print(f"converged          : {res.converged}")
    print(f"Newton iters total : {res.iters}")
    print(f"max |displacement| : {np.abs(res.U).max():.4f}")
    print(f"min bottom y (plane at -0.0100) : {ybot.min():.4f}")
    assert res.converged and ybot.min() > -0.012
    print("OK — declarative pipeline solved.")


if __name__ == "__main__":
    main()
