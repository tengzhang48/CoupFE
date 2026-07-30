"""Compiled (f2py) element kernel — CoupFE's fast assembly path.

A single residual definition is emitted once as a self-contained Fortran ``.for``
(its consistent tangent derived *inside* the kernel by complex step — one residual,
derived tangent).  :func:`build_element_kernel` f2py-compiles that ``.for`` together
with a thin driver wrapper into an importable module; :class:`CompiledElement`
drives it: one batched call evaluates every element in the group and returns
per-element ``(R, K)`` plus the updated state, so a state commit is a direct write
(no separate recover step).

This is a clean-room port of the research lab's compiled-element backend.  The lab
derived element shape/DOF/state sizes by introspecting a ``WeakForm``; CoupFE has no
weak-form layer at runtime, so those sizes are passed **explicitly** — the kernel is
just compiled Fortran with a known ABI.  CoupFE never imports the lab.

Sign convention: kernels may be emitted as Abaqus UELs or as CoupFE native kernels.
The Abaqus UEL convention is ``RHS = -R`` and ``AMATRX = -dRHS/dU``; the native
kernel already returns the weak-form ``R`` and ``K = dR/dU``.  This adapter keeps the
rest of CoupFE on the standard convention ``R`` with ``K = dR/dU``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def _initial_svars_from_schema(n_elem, n_gp, svars_size, schema):
    """Build a committed-state buffer from the declared state schema.

    The flat per-element state array is ordered per Gauss point, with each
    GP block laid out according to the schema offsets.  The result has shape
    (n_elem, svars_size) even when n_gp * per_gp_size < svars_size, because
    the runtime may pad the buffer for f2py shape compatibility.
    """
    svars = np.zeros((n_elem, svars_size), dtype=float)
    if not schema:
        return svars
    per_gp = sum(entry['size'] for entry in schema.values())
    for e in range(n_elem):
        for gp in range(n_gp):
            base = gp * per_gp
            for name, entry in schema.items():
                init = np.asarray(entry['init'], dtype=float)
                off = base + entry['offset']
                if entry['size'] == 1:
                    svars[e, off] = float(init)
                else:
                    svars[e, off:off + entry['size']] = init.ravel()
    return svars

# f2py wrappers that adapt each backend ABI to a flat callable.  Vendored into
# CoupFE so the runtime has no path dependency on the research lab.
_DRIVE_UEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "drive_uel.f90"))
_DRIVE_NATIVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "drive_native.f90"))


def build_element_kernel(for_path, module_name, workdir=None, backend=None):
    """f2py-compile an element ``.for`` + driver wrapper into a module.

    ``for_path`` is a self-contained Fortran kernel (material + helpers).  The
    backend is auto-detected from the source if not supplied:
    ``SUBROUTINE coupfe_element_rk`` → native, ``SUBROUTINE UEL`` → Abaqus UEL.
    The returned module exposes ``drive_native`` / ``drive_native_batch`` or
    ``drive_uel`` / ``drive_uel_batch`` accordingly.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="coupfe_uel_")
    os.makedirs(workdir, exist_ok=True)
    forname = os.path.basename(for_path)

    if backend is None:
        with open(for_path, encoding="utf-8", errors="replace") as f:
            src_head = f.read(4096)
        src_upper = src_head.upper()
        if "SUBROUTINE COUPFE_ELEMENT_RK" in src_upper:
            backend = "native"
        elif "SUBROUTINE UEL" in src_upper:
            backend = "abaqus_uel"
        else:
            raise ValueError(
                f"Cannot detect backend from {for_path!r}: no recognised "
                "subroutine signature. Pass backend='native' or 'abaqus_uel'.")

    if backend == "native":
        driver = _DRIVE_NATIVE
        entry_points = ("drive_native", "drive_native_batch")
    elif backend == "abaqus_uel":
        driver = _DRIVE_UEL
        entry_points = ("drive_uel", "drive_uel_batch")
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'native' or 'abaqus_uel'.")

    shutil.copy(driver, workdir)
    # Sanitize the generated .for to pure ASCII: its comments can contain a non-ASCII
    # char (e.g. a UTF-8 em-dash), which f2py's crackfortran fails to decode under a
    # C/ascii locale (as inside mpiexec).  Non-ASCII appears only in comments, so
    # replacing it is harmless and makes the build locale-independent.
    with open(for_path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    with open(os.path.join(workdir, forname), "w",
              encoding="ascii", errors="replace") as f:
        f.write(src)

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    only = ["only:"] + list(entry_points) + [":"]
    # Force the meson f2py backend on EVERY Python version. On 3.12+ it is the only
    # backend (numpy.distutils is gone); on <=3.11 f2py still defaults to the legacy
    # numpy.distutils backend, which breaks against modern setuptools
    # (``ModuleNotFoundError: distutils.msvccompiler``) and fails the build on e.g. CI's
    # Python 3.9. meson + ninja are already required (the ``runtime`` extra), so forcing
    # the backend just makes the build identical and locale-independent across versions.
    subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "-c", "--backend", "meson",
         os.path.basename(driver), forname, "-m", module_name] + only,
        cwd=workdir, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    return importlib.import_module(module_name)


class CompiledElement:
    """Drive a compiled (f2py) element kernel over a batch of like elements.

    ``module``: from :func:`build_element_kernel`.

    The element's shape is passed explicitly (no weak-form introspection):

    * ``props``: the material property vector the kernel reads (e.g. ``(G, K)``).
    * ``dof_per_node``: DOFs the element writes per node (2 for u-only 2D).
    * ``n_svars``: state-variable slots per element (0 ⇒ stateless).
    * ``mcrd``: spatial coordinate dimension (2 or 3).
    * ``n_elem``: number of elements (allocates the persistent ``svars`` buffer).

    ``element_rk_batch`` is the fast assembly path (one f2py call for the whole
    group); ``commit_group`` recomputes-and-commits the group's state at a
    converged ``U``.  The backend (``native`` or ``abaqus_uel``) is inferred from
    the module's entry points.
    """

    def __init__(self, module, props, dof_per_node, n_svars=0, mcrd=2,
                 n_elem=None, dt=1.0, backend=None, state_schema=None):
        self.m = module
        self.props = np.asarray(props, dtype=float)
        self.dof_per_node = int(dof_per_node)
        self.mcrd = int(mcrd)
        self.has_state = int(n_svars) > 0
        self.svars_size = max(int(n_svars), 1)
        self.dt = float(dt)
        # Declared state schema: OrderedDict name -> {init, shape, size,
        # quantity, offset}.  Lets the runtime know the named layout without
        # parsing Fortran.
        self.state_schema = state_schema
        if self.has_state and n_elem is not None:
            n_gp = 1
            if state_schema:
                per_gp = sum(entry['size'] for entry in state_schema.values())
                if per_gp > 0 and self.svars_size % per_gp == 0:
                    n_gp = self.svars_size // per_gp
            self.svars = _initial_svars_from_schema(
                n_elem, n_gp, self.svars_size, state_schema)
        else:
            self.svars = None
        # Trial state from the most recent residual/tangent evaluation.
        # For the native backend this is the out-of-place SVARS_OUT returned by
        # the kernel; commit() promotes it to the committed state.  For the UEL
        # backend it is the in-place-mutated copy returned by drive_uel.
        self.svars_trial = (self.svars.copy() if self.svars is not None
                            else None)
        self._jprops = np.array([0], dtype=np.int32)
        self._lflags = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        self._params = np.zeros(3)
        self._time = np.array([dt, dt])

        if backend is None:
            if hasattr(self.m, "drive_native"):
                backend = "native"
            elif hasattr(self.m, "drive_uel"):
                backend = "abaqus_uel"
            else:
                raise ValueError(
                    "Cannot infer backend from module: no drive_native or "
                    "drive_uel entry point found.")
        self.backend = backend

        if self.backend == "native":
            self._drive = self.m.drive_native
            self._drive_batch = self.m.drive_native_batch
        else:
            self._drive = self.m.drive_uel
            self._drive_batch = self.m.drive_uel_batch

    # ------------------------------------------------------------------ #
    # Single element (used for finite-difference / per-element checks).
    # ------------------------------------------------------------------ #
    def _call(self, ei, coords, U_e, DU_e, sv):
        if self.backend == "native":
            return self._drive(
                sv, np.asarray(coords, dtype=float).T[:self.mcrd],
                np.asarray(U_e, dtype=float), np.asarray(DU_e, dtype=float),
                self.props, self._jprops, self._time, self.dt, 1.0,
                self._lflags, self._params, 1, 1, 1, ei + 1, 1.0)
        # Abaqus UEL path
        return self._drive(
            sv, np.asarray(coords, dtype=float).T[:self.mcrd],
            np.asarray(U_e, dtype=float), np.asarray(DU_e, dtype=float),
            self.props, self._jprops, self._time, self.dt, 1.0,
            self._lflags, self._params, 1, 1, 1, ei + 1, 1.0)

    def element_rk(self, ei, coords, U_e, DU_e):
        """One element's ``(R_e, K_e)`` with the standard sign convention."""
        sv = (self.svars[ei].copy() if self.svars is not None
              else np.zeros(self.svars_size))
        if self.backend == "native":
            R, K, sv_new, _pn = self._call(ei, coords, U_e, DU_e, sv)
            if self.svars_trial is not None:
                self.svars_trial[ei] = np.asarray(sv_new, dtype=float).ravel()
            return (np.ascontiguousarray(R, dtype=float),
                    np.ascontiguousarray(K, dtype=float))
        rhs, amatrx, sv_new, _pn = self._call(ei, coords, U_e, DU_e, sv)
        if self.svars_trial is not None:
            self.svars_trial[ei] = np.asarray(sv_new, dtype=float).ravel()
        return (-np.ascontiguousarray(rhs, dtype=float),
                np.ascontiguousarray(amatrx, dtype=float))

    # ------------------------------------------------------------------ #
    # Whole-group batched call (the fast path).
    # ------------------------------------------------------------------ #
    def _call_batch(self, coords_all, U_all, DU_all, sv_all):
        crd = np.asfortranarray(
            np.transpose(np.asarray(coords_all, float), (2, 1, 0))[:self.mcrd])
        if self.backend == "native":
            return self._drive_batch(
                np.asfortranarray(sv_all.T), crd,
                np.asfortranarray(np.asarray(U_all, float).T),
                np.asfortranarray(np.asarray(DU_all, float).T),
                self.props, self._jprops, self._time, self.dt, 1.0,
                self._lflags, self._params, 1, 1, 1, 1.0)
        return self._drive_batch(
            np.asfortranarray(sv_all.T), crd,
            np.asfortranarray(np.asarray(U_all, float).T),
            np.asfortranarray(np.asarray(DU_all, float).T),
            self.props, self._jprops, self._time, self.dt, 1.0,
            self._lflags, self._params, 1, 1, 1, 1.0)

    def element_rk_batch(self, coords_all, U_all, DU_all):
        """Batched ``(R_all, K_all)`` for the whole group (the fast assembly path).

        Returns ``R_all`` (nelem, ndofel) and ``K_all`` (nelem, ndofel, ndofel)
        with the standard sign convention.  The backend-specific sign conversion
        (if any) is applied here.

        Also stores the returned state as ``svars_trial`` for a subsequent
        ``commit()``.
        """
        nelem = len(U_all)
        sv = (self.svars.copy() if self.svars is not None
              else np.zeros((nelem, self.svars_size)))
        if self.backend == "native":
            R, K, sv_new, _pn = self._call_batch(coords_all, U_all, DU_all, sv)
            if self.svars_trial is not None:
                self.svars_trial = np.ascontiguousarray(
                    np.asarray(sv_new, float).T)
            return (np.asarray(R, float).T,
                    np.ascontiguousarray(np.transpose(np.asarray(K, float), (2, 0, 1))))
        rhs, amatrx, sv_new, _pn = self._call_batch(coords_all, U_all, DU_all, sv)
        if self.svars_trial is not None:
            self.svars_trial = np.ascontiguousarray(
                np.asarray(sv_new, float).T)
        return (-np.asarray(rhs, float).T,
                np.ascontiguousarray(np.transpose(np.asarray(amatrx, float), (2, 0, 1))))

    def commit(self):
        """Promote the trial state from the last evaluation to committed state.

        This is the out-of-place native commit: no kernel recompute, just a
        buffer copy.  For the UEL backend the trial state was already computed
        by the last residual/tangent call and returned by drive_uel, so copying
        it is equivalent to (and cheaper than) the old commit_group recompute.
        """
        if self.svars is None or self.svars_trial is None:
            return
        self.svars = self.svars_trial.copy()

    def commit_group(self, coords_all, U_g, DU_g):
        """Recompute and commit the group's state at converged ``U`` (one batched
        f2py call over pre-gathered element data).  No-op for a stateless element.

        Kept for backward compatibility; new code should call ``commit()`` after
        a residual/tangent evaluation.
        """
        if self.svars is None:
            return
        _r, _a, sv_new, _pn = self._call_batch(coords_all, U_g, DU_g, self.svars.copy())
        self.svars = np.ascontiguousarray(np.asarray(sv_new, float).T)
        if self.svars_trial is not None:
            self.svars_trial = self.svars.copy()
