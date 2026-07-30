"""Geometry backends — the authoritative curved boundary, independent of the mesh.

A coarse mesh supplies *topology*; curvature comes from here. The contract is small:
``project(x)`` returns the closest point on the geometry, ``normal(x)`` the unit
normal there. During refinement a new boundary node is **projected** onto its
classified geometry — refining a faceted boundary *without* this just makes smaller
facets on the wrong surface (plan §26). Feature classification (which node sits on
which curve/surface) lives in the mesh view; this module is the geometry itself.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GeometryBackend(Protocol):
    def project(self, x: np.ndarray) -> np.ndarray:
        """Closest point on the geometry to ``x``."""
        ...

    def normal(self, x: np.ndarray) -> np.ndarray:
        """Unit normal of the geometry at ``project(x)``."""
        ...


class Circle:
    """A 2D circle/arc of radius ``R`` about ``center``. Projection is radial."""

    def __init__(self, R: float, center=(0.0, 0.0)):
        self.R = float(R)
        self.c = np.asarray(center, dtype=float)

    def project(self, x):
        d = np.asarray(x, dtype=float) - self.c
        r = float(np.linalg.norm(d))
        if r < 1e-300:
            return self.c + np.array([self.R, 0.0])
        return self.c + (self.R / r) * d

    def normal(self, x):
        d = np.asarray(x, dtype=float) - self.c
        r = float(np.linalg.norm(d))
        return d / r if r > 1e-300 else np.array([1.0, 0.0])


class Sphere:
    """A 3D sphere of radius ``R`` about ``center``. Projection is radial."""

    def __init__(self, R: float, center=(0.0, 0.0, 0.0)):
        self.R = float(R)
        self.c = np.asarray(center, dtype=float)

    def project(self, x):
        d = np.asarray(x, dtype=float) - self.c
        r = float(np.linalg.norm(d))
        if r < 1e-300:
            return self.c + np.array([self.R, 0.0, 0.0])
        return self.c + (self.R / r) * d

    def normal(self, x):
        d = np.asarray(x, dtype=float) - self.c
        r = float(np.linalg.norm(d))
        return d / r if r > 1e-300 else np.array([1.0, 0.0, 0.0])


class Plane:
    """A flat geometry through ``point`` with unit ``normal_`` (a line in 2D).

    Projection drops the component along the normal; a refined node already on the
    plane stays put (a flat boundary needs no re-embedding — included for
    completeness and as a classification target).
    """

    def __init__(self, point, normal_):
        self.p = np.asarray(point, dtype=float)
        n = np.asarray(normal_, dtype=float)
        self.n = n / np.linalg.norm(n)

    def project(self, x):
        x = np.asarray(x, dtype=float)
        return x - np.dot(x - self.p, self.n) * self.n

    def normal(self, x):
        return self.n
