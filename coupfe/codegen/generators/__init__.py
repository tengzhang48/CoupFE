"""Code generators: Python Material -> Fortran UMAT/UEL and input scaffolds."""

from .inp_scaffold import (
    UELModelConfig,
    ScaffoldReport,
    generate as generate_inp_scaffold,
    write_job_inp,
)
from .umat_gen import generate_small_strain_umat


def generate_uel(*args, **kwargs):
    """Generate an Abaqus UEL; imported lazily to avoid circular imports."""
    from .uel_gen import generate_uel as _generate_uel
    return _generate_uel(*args, **kwargs)


def generate_umat(*args, **kwargs):
    """Generate an Abaqus UMAT; imported lazily to avoid circular imports.

    NOTE: ``umat_gen`` is kept because it carries the shared Python->Fortran translation
    engine (``FortranTranslator``) that ``uel_gen`` reuses; this UEL-focused port validates
    the UEL path. (UINTER was dropped — ``uel_gen`` does not depend on it.)
    """
    from .umat_gen import generate_umat as _generate_umat
    return _generate_umat(*args, **kwargs)


__all__ = [
    "UELModelConfig",
    "ScaffoldReport",
    "generate_inp_scaffold",
    "write_job_inp",
    "generate_small_strain_umat",
    "generate_uel",
    "generate_umat",
]
