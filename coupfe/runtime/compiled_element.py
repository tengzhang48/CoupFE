"""Compiled (f2py) element kernel — CoupFE's fast assembly path.

A single residual definition is emitted once as a self-contained Fortran ``.for``
(its consistent tangent derived *inside* the kernel by complex step — one residual,
derived tangent).  :func:`build_element_kernel` f2py-compiles that ``.for`` together
with a thin driver wrapper into an importable module; :class:`CompiledElement`
drives it: one batched call evaluates every element in the group and returns
per-element ``(R, K)`` plus the updated state. Current native sources also carry
a residual-only twin for callbacks that do not need ``K``; older sources and
Abaqus UELs fall back to the joint path. A state commit is a direct write after
the caller has refreshed trial state at the accepted iterate.

The runtime receives element shape, DOF, and state sizes **explicitly**; it does
not import the build-time weak-form layer. The kernel is compiled Fortran behind
a documented ABI.

Sign convention: kernels may be emitted as Abaqus UELs or as CoupFE native kernels.
The Abaqus UEL convention is ``RHS = -R`` and ``AMATRX = -dRHS/dU``; the native
kernel already returns the weak-form ``R`` and ``K = dR/dU``.  This adapter keeps the
rest of CoupFE on the standard convention ``R`` with ``K = dR/dU``.
"""

from __future__ import annotations

import importlib
import os
import re
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

# f2py wrappers that adapt each backend ABI to a flat callable. They are shipped
# with CoupFE so the runtime has no external source-tree dependency.
_DRIVE_UEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "drive_uel.f90"))
_DRIVE_NATIVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "drive_native.f90"))
_DRIVE_NATIVE_R = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "drive_native_r.f90")
)
_NATIVE_R_ENTRY_RE = re.compile(
    r"^\s*SUBROUTINE\s+COUPFE_ELEMENT_R\s*\(",
    re.IGNORECASE | re.MULTILINE,
)

# The UEL backend below is a narrow compatibility adapter, not an Abaqus
# procedure simulator. It makes one normal nonlinear-static joint request for
# focused parity checks and selected research examples. Abaqus itself supplies
# LFLAGS when it runs the exported UEL.
_UEL_STATIC_JOINT_LFLAGS = (1, 1, 1, 0, 0, 0)


def build_element_kernel(for_path, module_name, workdir=None, backend=None):
    """f2py-compile an element ``.for`` + driver wrapper into a module.

    ``for_path`` is a self-contained Fortran kernel (material + helpers).  The
    backend is auto-detected from the source if not supplied:
    ``SUBROUTINE coupfe_element_rk`` → native, ``SUBROUTINE UEL`` → Abaqus UEL.
    The returned module exposes ``drive_native`` / ``drive_native_batch`` or
    ``drive_uel`` / ``drive_uel_batch`` accordingly.  A native source that
    also contains ``coupfe_element_r`` gains ``drive_native_r`` and
    ``drive_native_batch_r`` without changing the generated
    ``coupfe_element_rk`` entry. The f2py ``drive_native*`` wrappers use the
    compact seven-input native ABI shipped with this release, so cached modules
    built with an older wrapper must be rebuilt from source.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="coupfe_uel_")
    os.makedirs(workdir, exist_ok=True)
    forname = os.path.basename(for_path)

    with open(for_path, encoding="utf-8", errors="replace") as f:
        src = f.read()

    if backend is None:
        src_head = src[:4096]
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
        drivers = [_DRIVE_NATIVE]
        entry_points = ["drive_native", "drive_native_batch"]
        if _NATIVE_R_ENTRY_RE.search(src):
            drivers.append(_DRIVE_NATIVE_R)
            entry_points.extend(("drive_native_r", "drive_native_batch_r"))
    elif backend == "abaqus_uel":
        drivers = [_DRIVE_UEL]
        entry_points = ["drive_uel", "drive_uel_batch"]
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'native' or 'abaqus_uel'.")

    for driver in drivers:
        shutil.copy(driver, workdir)
    # Sanitize the generated .for to pure ASCII: its comments can contain a non-ASCII
    # char (e.g. a UTF-8 em-dash), which f2py's crackfortran fails to decode under a
    # C/ascii locale (as inside mpiexec).  Non-ASCII appears only in comments, so
    # replacing it is harmless and makes the build locale-independent.
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
         *[os.path.basename(driver) for driver in drivers], forname,
         "-m", module_name] + only,
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

    ``element_rk_batch`` is the joint assembly path (one f2py call for the whole
    group); ``element_r_batch`` uses the optional residual-only native entry.
    ``commit_group`` recomputes-and-commits the group's state at a converged
    ``U``. The backend (``native`` or ``abaqus_uel``) is inferred from the
    module's entry points. The native ABI contains no Abaqus ``LFLAGS`` or
    procedure data. The UEL backend is a narrow normal-static joint-call
    compatibility adapter; it is not a replacement for an Abaqus analysis or
    a general Abaqus procedure host.
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
            self._drive_r = getattr(self.m, "drive_native_r", None)
            self._drive_batch_r = getattr(self.m, "drive_native_batch_r", None)
        elif self.backend == "abaqus_uel":
            self._drive = self.m.drive_uel
            self._drive_batch = self.m.drive_uel_batch
            self._drive_r = None
            self._drive_batch_r = None
            self._uel_jprops = np.array([0], dtype=np.int32)
            self._uel_lflags = np.array(
                _UEL_STATIC_JOINT_LFLAGS, dtype=np.int32
            )
            self._uel_params = np.zeros(3)
        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. Use 'native' or "
                "'abaqus_uel'."
            )

    @property
    def has_residual_only(self):
        """Whether both single and batched R-only ABI entries are available."""
        return self.has_element_r and self.has_element_r_batch

    @property
    def has_element_r(self):
        """Whether the single-element R-only ABI entry is available."""
        return self._drive_r is not None

    @property
    def has_element_r_batch(self):
        """Whether the batched R-only ABI entry is available."""
        return self._drive_batch_r is not None

    # ------------------------------------------------------------------ #
    # Single element (used for finite-difference / per-element checks).
    # ------------------------------------------------------------------ #
    def _call(self, ei, coords, U_e, DU_e, sv):
        if self.backend == "native":
            return self._drive(
                sv, np.asarray(coords, dtype=float).T[:self.mcrd],
                np.asarray(U_e, dtype=float), np.asarray(DU_e, dtype=float),
                self.props, self._time, self.dt)
        # Narrow UEL compatibility path: normal static joint request only.
        return self._drive(
            sv, np.asarray(coords, dtype=float).T[:self.mcrd],
            np.asarray(U_e, dtype=float), np.asarray(DU_e, dtype=float),
            self.props, self._uel_jprops, self._time, self.dt, 1.0,
            self._uel_lflags, self._uel_params, 1, 1, 1, ei + 1, 1.0)

    def element_rk(self, ei, coords, U_e, DU_e):
        """One element's ``(R_e, K_e)`` with the standard sign convention."""
        sv = (self.svars[ei].copy() if self.svars is not None
              else np.zeros(self.svars_size))
        if self.backend == "native":
            R, K, sv_new = self._call(ei, coords, U_e, DU_e, sv)
            if self.svars_trial is not None:
                self.svars_trial[ei] = np.asarray(sv_new, dtype=float).ravel()
            return (np.ascontiguousarray(R, dtype=float),
                    np.ascontiguousarray(K, dtype=float))
        rhs, amatrx, sv_new, _pn = self._call(ei, coords, U_e, DU_e, sv)
        if self.svars_trial is not None:
            self.svars_trial[ei] = np.asarray(sv_new, dtype=float).ravel()
        return (-np.ascontiguousarray(rhs, dtype=float),
                np.ascontiguousarray(amatrx, dtype=float))

    def element_r(self, ei, coords, U_e, DU_e):
        """One element residual, skipping tangent work when the kernel permits.

        Native kernels generated before the residual-only ABI, and Abaqus UEL
        kernels, fall back to :meth:`element_rk` for compatibility.
        """
        if not self.has_element_r:
            return self.element_rk(ei, coords, U_e, DU_e)[0]
        sv = (self.svars[ei].copy() if self.svars is not None
              else np.zeros(self.svars_size))
        R, sv_new = self._drive_r(
            sv, np.asarray(coords, dtype=float).T[:self.mcrd],
            np.asarray(U_e, dtype=float), np.asarray(DU_e, dtype=float),
            self.props, self._time, self.dt)
        if self.svars_trial is not None:
            self.svars_trial[ei] = np.asarray(sv_new, dtype=float).ravel()
        return np.ascontiguousarray(R, dtype=float)

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
                self.props, self._time, self.dt)
        return self._drive_batch(
            np.asfortranarray(sv_all.T), crd,
            np.asfortranarray(np.asarray(U_all, float).T),
            np.asfortranarray(np.asarray(DU_all, float).T),
            self.props, self._uel_jprops, self._time, self.dt, 1.0,
            self._uel_lflags, self._uel_params, 1, 1, 1, 1.0)

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
            R, K, sv_new = self._call_batch(coords_all, U_all, DU_all, sv)
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

    def element_r_batch(self, coords_all, U_all, DU_all):
        """Batched residuals, using ``coupfe_element_r`` when available.

        The returned array has shape ``(nelem, ndofel)``.  Trial state is
        refreshed exactly as in :meth:`element_rk_batch`, so a residual-only
        accepted-state check can safely precede :meth:`commit`.  Older native
        kernels and Abaqus UELs retain their joint-R/K fallback.
        """
        if not self.has_element_r_batch:
            return self.element_rk_batch(coords_all, U_all, DU_all)[0]
        nelem = len(U_all)
        sv = (self.svars.copy() if self.svars is not None
              else np.zeros((nelem, self.svars_size)))
        crd = np.asfortranarray(
            np.transpose(np.asarray(coords_all, float), (2, 1, 0))[:self.mcrd])
        R, sv_new = self._drive_batch_r(
            np.asfortranarray(sv.T), crd,
            np.asfortranarray(np.asarray(U_all, float).T),
            np.asfortranarray(np.asarray(DU_all, float).T),
            self.props, self._time, self.dt)
        if self.svars_trial is not None:
            self.svars_trial = np.ascontiguousarray(
                np.asarray(sv_new, float).T)
        return np.asarray(R, float).T

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
        self.element_r_batch(coords_all, U_g, DU_g)
        self.commit()
