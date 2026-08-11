"""Validate the tracked source tree and built CoupFE release artifacts."""

from __future__ import annotations

import argparse
import re
import stat
import subprocess
import tarfile
import zipfile
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath


RUNTIME_ASSETS = {
    "coupfe/runtime/drive_native.f90",
    "coupfe/runtime/drive_native_r.f90",
    "coupfe/runtime/drive_uel.f90",
    "coupfe/runtime/elements/neo_hookean_hex8_fbar.for",
    "coupfe/runtime/elements/neo_hookean_q4.for",
    "coupfe/runtime/elements/neo_hookean_q4_fbar_native.for",
    "coupfe/runtime/elements/neo_hookean_q4_native.for",
}

TEMPLATE_ASSETS = {
    f"coupfe/codegen/generators/templates/{name}"
    for name in {
        "cs_linalg.for",
        "cs_tangent_engine.for",
        "edge_quad8.for",
        "face_hex.for",
        "gauss_hex.for",
        "gauss_rules.for",
        "isoparametric.for",
        "shape_hex20.for",
        "shape_hex8.for",
        "shape_quad4.for",
        "shape_quad8.for",
        "shape_tet4.for",
        "tangent_identities.for",
        "tensor_ops.for",
    }
}

PACKAGE_ASSETS = RUNTIME_ASSETS | TEMPLATE_ASSETS
PACKAGE_ASSETS |= {
    "coupfe/constraints/__init__.py",
    "coupfe/constraints/affine.py",
}

# Keep the installable package surface explicit.  Minimum-file checks catch
# missing runtime assets, while this exact inventory also rejects omitted
# Python modules and accidentally published internal modules.
PUBLIC_PACKAGE_FILES = PACKAGE_ASSETS | {
    "coupfe/__init__.py",
    "coupfe/materials.py",
    "coupfe/model.py",
} | {
    f"coupfe/assembly/{name}"
    for name in {
        "__init__.py",
        "assemble.py",
        "distributed.py",
        "factored.py",
    }
} | {
    "coupfe/codegen/__init__.py",
} | {
    f"coupfe/codegen/core/{name}"
    for name in {
        "__init__.py",
        "_cs_state.py",
        "defs.py",
        "fields.py",
        "fortran_helper.py",
        "material.py",
        "reference_assembly.py",
        "small_strain_plasticity.py",
        "soil.py",
        "symbolic_tangent.py",
        "tagent.py",
        "tensor.py",
        "verify.py",
        "weakform.py",
    }
} | {
    f"coupfe/codegen/generators/{name}"
    for name in {
        "__init__.py",
        "_fortran_format.py",
        "element_config.py",
        "inp_scaffold.py",
        "uel_fbar_coupled.py",
        "uel_gen.py",
        "uel_local_pressure.py",
        "uel_magneto.py",
        "umat_gen.py",
    }
} | {
    f"coupfe/codegen/testing/{name}"
    for name in {
        "__init__.py",
        "_util.py",
        "element_convergence.py",
        "finite_strain.py",
        "invariants.py",
        "manifest.py",
        "objectivity.py",
        "operators.py",
        "paths.py",
    }
} | {
    f"coupfe/mesh/{name}"
    for name in {
        "__init__.py",
        "distribute.py",
        "geometry.py",
        "refine.py",
        "view.py",
    }
} | {
    f"coupfe/operators/{name}"
    for name in {
        "__init__.py",
        "base.py",
        "bvh_numba.py",
        "contact.py",
        "contact3d.py",
        "contact3d_numba.py",
        "contact_search.py",
        "contact_semismooth.py",
        "element_group.py",
        "inertia.py",
    }
} | {
    "coupfe/runtime/__init__.py",
    "coupfe/runtime/compiled_element.py",
}

RIGHTS_BLOCKED_PATHS = {
    "examples/ring_compress/abaqus_ring_compress_reactions.csv",
}

# Keep application mesh semantics out of Core.  These are the historical paths
# that mixed periodic-box/node-matching policy with generic affine algebra; EDA
# now owns their replacement adapter.
CORE_BOUNDARY_BLOCKED_PATHS = {
    "coupfe/constraints/periodic.py",
    "coupfe/mesh/periodic.py",
}

PUBLIC_BASE_TEST_FILES = {
    f"tests/{name}"
    for name in {
        "test_affine_constraints.py",
        "test_all_primitive_barrier_2d.py",
        "test_contact_3d_persistent_friction.py",
        "test_contact_finite_sliding_repairing.py",
        "test_contact_persistent_friction.py",
        "test_contact_return_map_friction.py",
        "test_contact_search.py",
        "test_compiled_element_boundary.py",
        "test_distribute.py",
        "test_distributed_residual_split.py",
        "test_dynamics.py",
        "test_element_group.py",
        "test_mesh.py",
        "test_multibody_contact.py",
        "test_operator_contract.py",
        "test_tire_mesh.py",
    }
}
PUBLIC_OPTIONAL_TEST_FILES = {
    f"tests/{name}"
    for name in {
        "test_abaqus_ufl_hyperelastic_umats.py",
        "test_abaqus_ufl_inelastic_umats.py",
        "test_abaqus_ufl_paper_examples.py",
        "test_chester_anand_upmu_quad8.py",
        "test_codegen_native_residual_entry.py",
        "test_curved_convergence.py",
        "test_contact_semismooth_friction.py",
        "test_contact_vs_ppf.py",
        "test_exact_stick_friction.py",
        "test_examples_contact_3d.py",
        "test_friction_relay_finite_sliding.py",
        "test_hertz_contact.py",
        "test_neo_hookean_material.py",
        "test_model.py",
        "test_morphing_hex8.py",
        "test_morphing_hex8_inp.py",
        "test_pipeline.py",
        "test_phasefield_corrosion_cui.py",
        "test_stabilized_tet4.py",
    }
}
PUBLIC_TEST_FILES = PUBLIC_BASE_TEST_FILES | PUBLIC_OPTIONAL_TEST_FILES
PUBLIC_TEST_SUPPORT_FILES = {
    "tests/_fd_tangent.py",
}

PAPER_EXAMPLE_FILES = {
    "examples/gel_chester_anand/README.md",
    "examples/gel_chester_anand/paper_example_record.json",
    "examples/gel_chester_anand/u_p_mu_quad8/build.py",
    "examples/gel_chester_anand/u_p_mu_quad8/chester_anand_upmu_quad8_uel.for",
    "examples/morphing_hex8/README.md",
    "examples/morphing_hex8/abaqus_mesh.py",
    "examples/morphing_hex8/build.py",
    "examples/morphing_hex8/pressuregel_local_pressure_hex8.for",
    "examples/morphing_hex8/evidence_record.json",
    "examples/phasefield_corrosion_cui/README.md",
    "examples/phasefield_corrosion_cui/build.py",
    "examples/phasefield_corrosion_cui/paper_example_record.json",
    "examples/phasefield_corrosion_cui/phasefield_corrosion_cui_full_uel.for",
    "examples/stabilized_tet4/README.md",
    "examples/stabilized_tet4/build.py",
    "examples/stabilized_tet4/evidence_record.json",
    "examples/stabilized_tet4/scovazzi_block_tet4.for",
}

UMAT_EXAMPLE_FILES = {
    f"examples/{example}/{name}"
    for example, names in {
        "neo_hookean_umat": {
            "README.md",
            "build.py",
            "neo_hookean_umat.for",
        },
        "ogden_umat": {
            "README.md",
            "build.py",
            "ogden_umat.for",
        },
        "small_strain_j2_umat": {
            "README.md",
            "build.py",
            "small_strain_j2.for",
        },
        "small_strain_viscoelastic_umat": {
            "README.md",
            "build.py",
            "small_strain_viscoelastic.for",
        },
    }.items()
    for name in names
}

PUBLIC_EXAMPLE_FILES = {
    "examples/README.md",
    "examples/REFERENCES.md",
    "examples/compression_cylinders/README.md",
    "examples/compression_cylinders/build_model.py",
    "examples/compression_cylinders/dirichlet_lid.py",
    "examples/compression_cylinders/mooney_rivlin.py",
    "examples/compression_cylinders/mr_fbar_q4.for",
    "examples/compression_cylinders/neo_fbar_q4.for",
    "examples/compression_cylinders/neo_std_q4.for",
    "examples/compression_cylinders/neo_up_native_q4.for",
    "examples/compression_cylinders/parse_inp.py",
    "examples/compression_cylinders/run.py",
    "examples/contact_3d_blocks/render.py",
    "examples/contact_3d_blocks/run.py",
    "examples/contact_3d_friction/run.py",
    "examples/contact_vs_ppf/README.md",
    "examples/contact_vs_ppf/coupfe_box_on_floor.py",
    "examples/contact_vs_ppf/friction_threshold_sweep.py",
    "examples/contact_vs_ppf/ppf_reference.py",
    "examples/curved_annulus/annulus.py",
    "examples/curved_annulus/run.py",
    "examples/exact_stick_friction/run.py",
    "examples/finite_sliding_capstan/run.py",
    "examples/finite_sliding_friction/run.py",
    "examples/friction_identifiability/run.py",
    "examples/gel_chester_anand/README.md",
    "examples/gel_chester_anand/paper_example_record.json",
    "examples/gel_chester_anand/u_p_mu_quad8/build.py",
    "examples/gel_chester_anand/u_p_mu_quad8/chester_anand_upmu_quad8_uel.for",
    "examples/hertz_contact/README.md",
    "examples/hertz_contact/render.py",
    "examples/hertz_contact/run.py",
    "examples/j2_plasticity_uel/build.py",
    "examples/linear_bar/bar.py",
    "examples/linear_bar/render.py",
    "examples/linear_bar/run.py",
    "examples/model_pipeline/run.py",
    "examples/morphing_hex8/README.md",
    "examples/morphing_hex8/abaqus_mesh.py",
    "examples/morphing_hex8/build.py",
    "examples/morphing_hex8/evidence_record.json",
    "examples/morphing_hex8/pressuregel_local_pressure_hex8.for",
    "examples/mpi_smoke/distributed_3d_primitives.py",
    "examples/mpi_smoke/distributed_broadphase.py",
    "examples/mpi_smoke/distributed_cylinders.py",
    "examples/mpi_smoke/distributed_deformable_barrier.py",
    "examples/mpi_smoke/distributed_deformable_residual.py",
    "examples/mpi_smoke/distributed_deformable_solve.py",
    "examples/mpi_smoke/distributed_dual_multiplier.py",
    "examples/mpi_smoke/distributed_dynamics_3d_blocks.py",
    "examples/mpi_smoke/distributed_dynamics_3d_friction.py",
    "examples/mpi_smoke/distributed_dynamics_barrier.py",
    "examples/mpi_smoke/distributed_dynamics_friction.py",
    "examples/mpi_smoke/distributed_friction.py",
    "examples/mpi_smoke/distributed_lid_walls.py",
    "examples/mpi_smoke/distributed_neohookean.py",
    "examples/mpi_smoke/distributed_residual.py",
    "examples/mpi_smoke/distributed_robin_pressure.py",
    "examples/mpi_smoke/distributed_solve.py",
    "examples/neo_hookean_block/block.py",
    "examples/neo_hookean_block/render.py",
    "examples/neo_hookean_block/run.py",
    "examples/neo_hookean_inelastic_local_pressure_quad4/build.py",
    "examples/neo_hookean_local_pressure_hex8/build.py",
    "examples/neo_hookean_local_pressure_hex8/neohookean_up_hex8_uel.for",
    "examples/neo_hookean_local_pressure_quad4/build.py",
    "examples/neo_hookean_local_pressure_quad4/neohookean_up_q4_uel.for",
    "examples/neo_hookean_mixed/build.py",
    "examples/neo_hookean_mixed/neo_hookean_mixed_uel.for",
    "examples/neo_hookean_umat/README.md",
    "examples/neo_hookean_umat/build.py",
    "examples/neo_hookean_umat/neo_hookean_umat.for",
    "examples/ogden_umat/README.md",
    "examples/ogden_umat/build.py",
    "examples/ogden_umat/ogden_umat.for",
    "examples/phasefield_corrosion_cui/README.md",
    "examples/phasefield_corrosion_cui/build.py",
    "examples/phasefield_corrosion_cui/paper_example_record.json",
    "examples/phasefield_corrosion_cui/phasefield_corrosion_cui_full_uel.for",
    "examples/phasefield_fracture_uel/build.py",
    "examples/phasefield_fracture_uel/phasefield_fracture_uel.for",
    "examples/ring_compress/README.md",
    "examples/ring_compress/reproduce.py",
    "examples/ring_compress/reproduce_dynamics.py",
    "examples/scalar_diffusion_uel/build.py",
    "examples/semismooth_friction/run.py",
    "examples/simple_gel_quad4/build.py",
    "examples/simple_gel_quad4/simple_gel_quad4_uel.for",
    "examples/small_strain_j2_umat/README.md",
    "examples/small_strain_j2_umat/build.py",
    "examples/small_strain_j2_umat/small_strain_j2.for",
    "examples/small_strain_viscoelastic_umat/README.md",
    "examples/small_strain_viscoelastic_umat/build.py",
    "examples/small_strain_viscoelastic_umat/small_strain_viscoelastic.for",
    "examples/stabilized_tet4/README.md",
    "examples/stabilized_tet4/build.py",
    "examples/stabilized_tet4/evidence_record.json",
    "examples/stabilized_tet4/scovazzi_block_tet4.for",
    "examples/thermo_mechanics_quad8/build.py",
    "examples/thermo_mechanics_quad8/thermo_mechanics_quad8_uel.for",
    "examples/tire_contact/README.md",
    "examples/tire_contact/analyze.py",
    "examples/tire_contact/mesh.py",
    "examples/tire_contact/run.py",
    "examples/tire_contact/sensitivity.py",
    "examples/tire_contact/vonmises.py",
    "examples/uel_scaffold_quad4/build.py",
}

PUBLIC_DOC_FILES = {
    "docs/assets/hertz-contact-benchmark.svg",
    "docs/assets/linear-bar-snapshot.svg",
    "docs/assets/neo-hookean-block-snapshot.svg",
    "docs/assets/contact-3d-blocks-snapshot.svg",
    "docs/DESIGN.md",
    "docs/api.md",
    "docs/capabilities.md",
    "docs/install.md",
    "docs/lessons_learned.md",
    "docs/porting.md",
    "docs/roadmap.md",
    "docs/standalone_gpu_plan.md",
    "docs/status.md",
    "docs/theory/contact_dynamics.md",
    "docs/theory/framework.md",
}
PUBLIC_SKILL_FILES = {
    f"skills/{name}"
    for name in {
        "SKILL.md",
        "contact.md",
        "distributed.md",
        "model_development.md",
        "performance.md",
        "pipeline.md",
        "pitfalls.md",
        "preflight.md",
        "testing.md",
    }
}
PUBLIC_SITE_FILES = {
    "site/evidence.json",
    "site/hertz-contact-benchmark.svg",
    "site/linear-bar-snapshot.svg",
    "site/neo-hookean-block-snapshot.svg",
    "site/contact-3d-blocks-snapshot.svg",
    "site/index.html",
    "site/styles.css",
}
PUBLIC_VALIDATION_FILES = {
    "validation/README.md",
    "validation/example_review_2026-08-08.md",
}

REQUIRED_SDIST_FILES = (
    PUBLIC_PACKAGE_FILES
    | PUBLIC_TEST_FILES
    | PUBLIC_TEST_SUPPORT_FILES
    | PAPER_EXAMPLE_FILES
    | UMAT_EXAMPLE_FILES
    | PUBLIC_EXAMPLE_FILES
    | PUBLIC_DOC_FILES
    | PUBLIC_SKILL_FILES
    | PUBLIC_SITE_FILES
    | PUBLIC_VALIDATION_FILES
    | {
        ".github/scripts/check_release_artifacts.py",
        ".github/scripts/check_site.py",
        "LICENSE",
        "LICENSE-DOCS.md",
        "LICENSE-ABAQUS-UFL-EXAMPLES",
        "NOTICE",
        "README.md",
        "CITATION.cff",
        "CREDITS.md",
        "MANIFEST.in",
        "pyproject.toml",
        "examples/README.md",
        "examples/REFERENCES.md",
        "examples/ring_compress/README.md",
        "examples/tire_contact/README.md",
    }
)

RESEARCH_EXAMPLE_DIRS = {
    "compression_cylinders",
    "contact_3d_blocks",
    "contact_3d_friction",
    "contact_vs_ppf",
    "gel_chester_anand",
    "morphing_hex8",
    "neo_hookean_inelastic_local_pressure_quad4",
    "neo_hookean_local_pressure_hex8",
    "neo_hookean_local_pressure_quad4",
    "phasefield_fracture_uel",
    "phasefield_corrosion_cui",
    "ring_compress",
    "stabilized_tet4",
    "tire_contact",
}
READY_EXAMPLE_DIRS = {
    "curved_annulus",
    "exact_stick_friction",
    "finite_sliding_capstan",
    "finite_sliding_friction",
    "friction_identifiability",
    "hertz_contact",
    "j2_plasticity_uel",
    "linear_bar",
    "model_pipeline",
    "mpi_smoke",
    "neo_hookean_block",
    "neo_hookean_umat",
    "neo_hookean_mixed",
    "ogden_umat",
    "scalar_diffusion_uel",
    "semismooth_friction",
    "simple_gel_quad4",
    "small_strain_j2_umat",
    "small_strain_viscoelastic_umat",
    "thermo_mechanics_quad8",
    "uel_scaffold_quad4",
}
PUBLIC_EXAMPLE_DIRS = READY_EXAMPLE_DIRS | RESEARCH_EXAMPLE_DIRS

FORBIDDEN_PARTS = {
    ".git",
    ".mypy_cache",
    ".numba_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
FORBIDDEN_NAME_PATTERNS = {
    "EXTERNAL_OBSERVER_FINDINGS*.md",
    "HANDOFF*.md",
    "HISTORY_*benchmark*.md",
    "NOTE_*.md",
    "NOTE_TO_*.md",
    "PORT_PROMPT*.md",
    "RELEASE_READINESS*.md",
    "REVIEW_NOTES*.md",
    "SUMMARY_for_*.md",
    "agent_working_agreement.md",
    "codegen_*_next.md",
    "codegen_*_start.md",
    "*_for_claude*.md",
    "*_for_gpt*.md",
    "*.tar.bz2",
    "*.tar.gz",
    "*.tar.xz",
    "*.tgz",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".a",
    ".bin",
    ".bz2",
    ".cab",
    ".class",
    ".dll",
    ".dmg",
    ".doc",
    ".docx",
    ".dylib",
    ".exe",
    ".gz",
    ".iso",
    ".jar",
    ".lib",
    ".lz",
    ".lz4",
    ".mod",
    ".nbc",
    ".nbi",
    ".npz",
    ".o",
    ".obj",
    ".pdf",
    ".ppt",
    ".pptx",
    ".pyc",
    ".pyd",
    ".pyo",
    ".rar",
    ".so",
    ".tar",
    ".tgz",
    ".whl",
    ".xls",
    ".xlsx",
    ".xz",
    ".zip",
    ".zst",
    ".zstd",
}
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
DOCUMENTATION_IMAGE_DIRS = {"_static", "assets", "figures", "images"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".csv",
    ".css",
    ".f90",
    ".for",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# This address is deliberately published in package and citation metadata. The
# exact allowlist keeps the private-material scan useful for every other
# personal address.
PUBLIC_CONTACT_EMAILS = {"tzhang48@syr.edu"}


def _validate_names(names: list[str], artifact: Path) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise SystemExit(f"{artifact.name} contains duplicate entries: {duplicates}")

    unsafe = []
    for name in names:
        path = PurePosixPath(name)
        if "\\" in name or path.is_absolute() or ".." in path.parts:
            unsafe.append(name)
    if unsafe:
        raise SystemExit(f"{artifact.name} contains unsafe paths: {sorted(unsafe)}")


def _is_documentation_image(path: PurePosixPath) -> bool:
    """Allow curated README/doc figures without allowing images anywhere."""

    parts = tuple(part.casefold() for part in path.parts)
    if path.suffix.casefold() not in IMAGE_SUFFIXES or not parts:
        return False
    if path.as_posix() in PUBLIC_SITE_FILES:
        return True
    if parts[0] not in {"assets", "docs", "examples"}:
        return False
    return any(part in DOCUMENTATION_IMAGE_DIRS for part in parts[:-1])


def _is_forbidden_path(name: str) -> bool:
    path = PurePosixPath(name)
    parts = tuple(part.casefold() for part in path.parts)
    basename = path.name.casefold()
    suffix = path.suffix.casefold()
    return bool(
        path.as_posix() in RIGHTS_BLOCKED_PATHS
        or path.as_posix() in CORE_BOUNDARY_BLOCKED_PATHS
        or
        set(parts).intersection(part.casefold() for part in FORBIDDEN_PARTS)
        or any(
            fnmatchcase(basename, pattern.casefold())
            for pattern in FORBIDDEN_NAME_PATTERNS
        )
        # Keep the retired experimental PQP path out of public artifacts even
        # when a filename varies.
        or any("pqp" in part for part in parts)
        or suffix in FORBIDDEN_SUFFIXES
        or (suffix in IMAGE_SUFFIXES and not _is_documentation_image(path))
    )


def _reject_forbidden_files(names: set[str], artifact: Path) -> None:
    rejected = sorted(
        name for name in names if _is_forbidden_path(name)
    )
    if rejected:
        raise SystemExit(f"{artifact.name} contains forbidden entries: {rejected}")


def _validate_artifact_example_dirs(
    names: set[str], artifact: Path, *, require_complete: bool
) -> None:
    present = {
        path.parts[1]
        for name in names
        if len((path := PurePosixPath(name)).parts) >= 3
        and path.parts[0] == "examples"
    }
    unexpected = sorted(present - PUBLIC_EXAMPLE_DIRS)
    missing = sorted(PUBLIC_EXAMPLE_DIRS - present) if require_complete else []
    if missing or unexpected:
        raise SystemExit(
            f"{artifact.name} public example directory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_exact_subtree(
    names: set[str], artifact: Path, prefix: str, expected: set[str]
) -> None:
    """Require one public subtree to match its reviewed inventory exactly."""

    present = {
        name
        for name in names
        if PurePosixPath(name).parts
        and PurePosixPath(name).parts[0] == prefix
    }
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing or unexpected:
        raise SystemExit(
            f"{artifact.name} public {prefix} inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_public_subtrees(names: set[str], artifact: Path) -> None:
    """Require all reviewed public source subtrees exactly."""

    inventories = {
        "coupfe": PUBLIC_PACKAGE_FILES,
        "docs": PUBLIC_DOC_FILES,
        "skills": PUBLIC_SKILL_FILES,
        "examples": PUBLIC_EXAMPLE_FILES,
        "site": PUBLIC_SITE_FILES,
        "validation": PUBLIC_VALIDATION_FILES,
    }
    for prefix, expected in inventories.items():
        _validate_exact_subtree(names, artifact, prefix, expected)


def _reject_private_harness(names: set[str], artifact: Path) -> None:
    """Allow only the explicitly passing public test partition."""

    rejected = sorted(
        name
        for name in names
        if PurePosixPath(name).parts
        and (
            (
                PurePosixPath(name).parts[0] == "tests"
                and name not in PUBLIC_TEST_FILES | PUBLIC_TEST_SUPPORT_FILES
            )
            or (
                PurePosixPath(name).parts[0] == "validation"
                and name not in PUBLIC_VALIDATION_FILES
            )
        )
    )
    if rejected:
        raise SystemExit(
            f"{artifact.name} contains non-public validation harness files: "
            f"{rejected}"
        )


def _require_files(
    available: set[str], required: set[str], artifact: Path
) -> None:
    missing = sorted(required - available)
    if missing:
        raise SystemExit(f"{artifact.name} is missing required files: {missing}")


def _sensitive_fragments() -> tuple[str, ...]:
    # Concatenation keeps the checker from rejecting its own source.
    return (
        "/" + "home/",
        "/" + "media/",
        "git" + "@jetstream",
        "gh" + "p_",
        "BEGIN " + "PRIVATE KEY",
    )


def _validate_text(name: str, payload: bytes, artifact: Path) -> None:
    path = PurePosixPath(name)
    if path.suffix.casefold() not in TEXT_SUFFIXES:
        return
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{artifact.name}:{name} is not valid UTF-8 text") from exc

    hits = [fragment for fragment in _sensitive_fragments() if fragment in text]
    personal_emails = {
        email.casefold()
        for email in re.findall(
            r"[\w.+-]+@(?:gmail\.com|syr\.edu)\b",
            text,
            flags=re.IGNORECASE,
        )
    }
    if personal_emails - PUBLIC_CONTACT_EMAILS:
        hits.append("personal-email-address")
    if hits:
        raise SystemExit(
            f"{artifact.name}:{name} contains private-path/credential material: "
            f"{sorted(hits)}"
        )


def _validate_citation_records(
    citation: str, project: str, credits: str, artifact: Path
) -> None:
    """Keep public citation metadata and role-specific references synchronized."""

    project_version = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    cff_version = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation, re.MULTILINE)
    project_author = re.search(
        r'^authors\s*=\s*\[\{\s*name\s*=\s*"([^"]+)"'
        r'(?:,\s*email\s*=\s*"([^"]+)")?',
        project,
        re.MULTILINE,
    )
    given_name = re.search(r"^\s+given-names:\s*([^\n]+)", citation, re.MULTILINE)
    family_name = re.search(
        r"^\s+-?\s*family-names:\s*([^\n]+)", citation, re.MULTILINE
    )
    cff_email = re.search(r"^\s+email:\s*([^\s]+)", citation, re.MULTILINE)

    required_matches = {
        "project version": project_version,
        "CFF version": cff_version,
        "project author": project_author,
        "CFF given name": given_name,
        "CFF family name": family_name,
        "CFF email": cff_email,
    }
    missing = sorted(name for name, match in required_matches.items() if match is None)
    if missing:
        raise SystemExit(f"{artifact.name} citation metadata is incomplete: {missing}")

    assert project_version is not None and cff_version is not None
    assert project_author is not None and given_name is not None and family_name is not None
    assert cff_email is not None
    cff_author = " ".join(
        value.group(1).strip().strip("'\"") for value in (given_name, family_name)
    )
    project_email = project_author.group(2)
    cff_email_value = cff_email.group(1).strip().strip("'\"")
    if project_version.group(1) != cff_version.group(1):
        raise SystemExit(f"{artifact.name} CFF and package versions disagree")
    if project_author.group(1) != cff_author:
        raise SystemExit(f"{artifact.name} CFF and package authors disagree")
    if project_email != cff_email_value:
        raise SystemExit(f"{artifact.name} CFF and package author emails disagree")

    required_credits = {
        "Making coupled-field Abaqus user elements simple",
        "manuscript submitted for publication",
        "https://github.com/tengzhang48/abaqus_ufl",
        "10.1145/2566630",
        "10.5281/zenodo.10447666",
        "https://github.com/st-tech/ppf-contact-solver",
        "8b7740b032131aeeb46f51d882c96e09b171acc8",
        "10.1145/3687908",
        "10.1145/3386569.3392425",
    }
    normalized_credits = " ".join(credits.split())
    absent = sorted(
        reference for reference in required_credits if reference not in normalized_credits
    )
    if absent:
        raise SystemExit(
            f"{artifact.name} role-specific citation record is incomplete: {absent}"
        )


def _validate_example_policy(source_root: Path) -> None:
    """Require the source tree and ledger to match the public example policy."""

    example_root = source_root / "examples"
    directories = {
        path.name
        for path in example_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    ledger = (example_root / "REFERENCES.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `([^`]+)` \| \*\*(READY|RESEARCH)",
        ledger,
        flags=re.MULTILINE,
    )
    row_names = [name for name, _status in rows]
    duplicates = sorted(
        name for name, count in Counter(row_names).items() if count > 1
    )
    ledger_names = set(row_names)
    missing_directories = sorted(PUBLIC_EXAMPLE_DIRS - directories)
    unexpected_directories = sorted(directories - PUBLIC_EXAMPLE_DIRS)
    if missing_directories or unexpected_directories:
        raise SystemExit(
            "source public example directory mismatch: "
            f"missing={missing_directories}, "
            f"unexpected={unexpected_directories}"
        )
    if duplicates or ledger_names != PUBLIC_EXAMPLE_DIRS:
        raise SystemExit(
            "example reference inventory mismatch: "
            f"duplicates={duplicates}, "
            f"ledger_only={sorted(ledger_names - PUBLIC_EXAMPLE_DIRS)}, "
            f"guard_only={sorted(PUBLIC_EXAMPLE_DIRS - ledger_names)}"
        )
    ledger_research = {name for name, status in rows if status == "RESEARCH"}
    if ledger_research != RESEARCH_EXAMPLE_DIRS:
        raise SystemExit(
            "research-example policy mismatch: "
            f"ledger_only={sorted(ledger_research - RESEARCH_EXAMPLE_DIRS)}, "
            f"guard_only={sorted(RESEARCH_EXAMPLE_DIRS - ledger_research)}"
        )
    ledger_ready = {name for name, status in rows if status == "READY"}
    if ledger_ready != READY_EXAMPLE_DIRS:
        raise SystemExit(
            "ready-example policy mismatch: "
            f"ledger_only={sorted(ledger_ready - READY_EXAMPLE_DIRS)}, "
            f"guard_only={sorted(READY_EXAMPLE_DIRS - ledger_ready)}"
        )


def _validate_source_tree(
    source_root: Path,
    *,
    allow_untracked_required: bool = False,
    allow_dirty_source: bool = False,
    allow_indexed_rights_blocked_deletion: bool = False,
) -> int:
    """Check tracked plus non-ignored untracked release inputs.

    Ignored local build products are intentionally left to the artifact checks.
    Including non-ignored untracked files makes this useful before a new
    documentation file has been added to the index.
    """

    status_result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    )
    if status_result.stdout:
        message = (
            f"{source_root} is not a clean committed source tree; release "
            "validation must run from the exact committed export"
        )
        if not allow_dirty_source:
            raise SystemExit(message)
        print(f"WARNING: {message} (audit override enabled)")

    tracked_result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--cached",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    untracked_result = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    tracked = {
        name.decode("utf-8", errors="strict")
        for name in tracked_result.stdout.split(b"\0")
        if name
    }
    untracked = {
        name.decode("utf-8", errors="strict")
        for name in untracked_result.stdout.split(b"\0")
        if name
    }
    names = sorted(tracked | untracked)
    _validate_names(names, source_root)

    blocked_present = sorted(
        name
        for name in names
        if name in RIGHTS_BLOCKED_PATHS
        and (
            (source_root / PurePosixPath(name)).exists()
            or (source_root / PurePosixPath(name)).is_symlink()
        )
    )
    if blocked_present:
        raise SystemExit(
            f"{source_root} contains rights-blocked release inputs: "
            f"{blocked_present}"
        )
    blocked_indexed = sorted(RIGHTS_BLOCKED_PATHS.intersection(tracked))
    if blocked_indexed:
        message = (
            f"{source_root} still indexes rights-blocked release inputs: "
            f"{blocked_indexed}"
        )
        if not allow_indexed_rights_blocked_deletion:
            raise SystemExit(message)
        print(f"WARNING: {message} (audit override enabled)")
    # A review worktree may carry an unstaged tracked deletion. Exclude only
    # the exact blocked path from the index-derived inventory after proving it
    # is absent; a committed release root will naturally omit it.
    names = [name for name in names if name not in RIGHTS_BLOCKED_PATHS]

    missing = sorted(
        name
        for name in names
        if not (source_root / PurePosixPath(name)).exists()
        and not (source_root / PurePosixPath(name)).is_symlink()
    )
    if missing:
        raise SystemExit(
            f"{source_root} has tracked paths missing from the working tree: {missing}"
        )

    symlinks = sorted(
        name
        for name in names
        if (source_root / PurePosixPath(name)).is_symlink()
    )
    if symlinks:
        raise SystemExit(f"{source_root} contains symbolic links: {symlinks}")

    files = set(names)
    _require_files(files, REQUIRED_SDIST_FILES, source_root)
    untracked_required = sorted(REQUIRED_SDIST_FILES - tracked)
    if untracked_required:
        message = (
            f"{source_root} has required release inputs that are not tracked: "
            f"{untracked_required}"
        )
        if not allow_untracked_required:
            raise SystemExit(message)
        print(f"WARNING: {message} (audit override enabled)")
    _reject_forbidden_files(files, source_root)
    _validate_public_subtrees(files, source_root)
    for name in sorted(files):
        path = source_root / PurePosixPath(name)
        if path.is_file():
            _validate_text(name, path.read_bytes(), source_root)
    _validate_citation_records(
        (source_root / "CITATION.cff").read_text(encoding="utf-8"),
        (source_root / "pyproject.toml").read_text(encoding="utf-8"),
        (source_root / "CREDITS.md").read_text(encoding="utf-8"),
        source_root,
    )
    _validate_example_policy(source_root)
    return len(files)


def _validate_wheel(wheel: Path) -> int:
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()

    names = [member.filename for member in members]
    _validate_names(names, wheel)

    symlinks = sorted(
        member.filename
        for member in members
        if stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF)
    )
    if symlinks:
        raise SystemExit(f"{wheel.name} contains symbolic links: {symlinks}")

    files = {member.filename for member in members if not member.is_dir()}
    metadata = [
        PurePosixPath(name).parts[0]
        for name in files
        if len(PurePosixPath(name).parts) == 2
        and PurePosixPath(name).parts[0].endswith(".dist-info")
        and PurePosixPath(name).parts[1] == "METADATA"
    ]
    if len(metadata) != 1:
        raise SystemExit(
            f"{wheel.name} must contain exactly one .dist-info/METADATA file"
        )

    dist_info = metadata[0]
    required = PUBLIC_PACKAGE_FILES | {
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/LICENSE-DOCS.md",
        f"{dist_info}/licenses/LICENSE-ABAQUS-UFL-EXAMPLES",
        f"{dist_info}/licenses/NOTICE",
    }
    _require_files(files, required, wheel)
    _reject_forbidden_files(files, wheel)
    _validate_exact_subtree(files, wheel, "coupfe", PUBLIC_PACKAGE_FILES)
    _validate_artifact_example_dirs(files, wheel, require_complete=False)
    _reject_private_harness(files, wheel)
    with zipfile.ZipFile(wheel) as archive:
        for member in members:
            if not member.is_dir():
                _validate_text(member.filename, archive.read(member), wheel)
    return len(files)


def _validate_sdist(sdist: Path) -> int:
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()

    names = [member.name for member in members]
    _validate_names(names, sdist)

    unsupported = sorted(
        member.name for member in members if not (member.isfile() or member.isdir())
    )
    if unsupported:
        raise SystemExit(
            f"{sdist.name} contains links or special-file entries: {unsupported}"
        )

    roots = {
        PurePosixPath(member.name).parts[0]
        for member in members
        if PurePosixPath(member.name).parts
    }
    if len(roots) != 1:
        raise SystemExit(f"{sdist.name} must contain exactly one top-level directory")
    root = roots.pop()

    files = {
        PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
        for member in members
        if member.isfile()
        and len(PurePosixPath(member.name).parts) > 1
        and PurePosixPath(member.name).parts[0] == root
    }
    _require_files(files, REQUIRED_SDIST_FILES, sdist)
    _reject_forbidden_files(files, sdist)
    _validate_public_subtrees(files, sdist)
    _reject_private_harness(files, sdist)
    _validate_artifact_example_dirs(files, sdist, require_complete=True)
    with tarfile.open(sdist, mode="r:gz") as archive:
        citation_payloads: dict[str, str] = {}
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise SystemExit(f"{sdist.name}:{member.name} could not be read")
                relative = PurePosixPath(
                    *PurePosixPath(member.name).parts[1:]
                ).as_posix()
                payload = stream.read()
                _validate_text(relative, payload, sdist)
                if relative in {"CITATION.cff", "CREDITS.md", "pyproject.toml"}:
                    citation_payloads[relative] = payload.decode("utf-8")
    _validate_citation_records(
        citation_payloads["CITATION.cff"],
        citation_payloads["pyproject.toml"],
        citation_payloads["CREDITS.md"],
        sdist,
    )
    return len(files)


def validate(
    dist_dir: Path,
    source_root: Path | None = None,
    *,
    allow_untracked_required: bool = False,
    allow_dirty_source: bool = False,
    allow_indexed_rights_blocked_deletion: bool = False,
) -> None:
    source_count = (
        _validate_source_tree(
            source_root.resolve(),
            allow_untracked_required=allow_untracked_required,
            allow_dirty_source=allow_dirty_source,
            allow_indexed_rights_blocked_deletion=(
                allow_indexed_rights_blocked_deletion
            ),
        )
        if source_root is not None
        else None
    )
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"expected one wheel and one sdist in {dist_dir}, "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )

    wheel_count = _validate_wheel(wheels[0])
    sdist_count = _validate_sdist(sdists[0])
    source_summary = (
        f"source tree ({source_count} files), " if source_count is not None else ""
    )
    print(
        f"validated {source_summary}{wheels[0].name} ({wheel_count} files) and "
        f"{sdists[0].name} ({sdist_count} files)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing exactly one wheel and one .tar.gz sdist",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help=(
            "git worktree whose tracked and non-ignored untracked files are "
            "checked before the artifacts (default: current directory)"
        ),
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="skip the source-tree check and inspect only the built archives",
    )
    parser.add_argument(
        "--allow-untracked-required-for-audit",
        action="store_true",
        help=(
            "inspect a review worktree whose required new release files are not "
            "yet tracked; the source result is not publishable"
        ),
    )
    parser.add_argument(
        "--allow-dirty-source-for-audit",
        action="store_true",
        help=(
            "inspect a dirty review worktree; the source result is not "
            "publishable and must be repeated from the exact committed export"
        ),
    )
    parser.add_argument(
        "--allow-indexed-rights-blocked-deletion-for-audit",
        action="store_true",
        help=(
            "inspect a review worktree where the rights-blocked ring table is "
            "absent from disk but its deletion has not been staged; the source "
            "result is not publishable"
        ),
    )
    args = parser.parse_args()
    validate(
        args.dist_dir,
        source_root=None if args.artifacts_only else args.source_root,
        allow_untracked_required=args.allow_untracked_required_for_audit,
        allow_dirty_source=args.allow_dirty_source_for_audit,
        allow_indexed_rights_blocked_deletion=(
            args.allow_indexed_rights_blocked_deletion_for_audit
        ),
    )


if __name__ == "__main__":
    main()
