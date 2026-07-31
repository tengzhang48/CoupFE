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
    "coupfe/runtime/drive_uel.f90",
    "coupfe/runtime/elements/neo_hookean_hex8_fbar.for",
    "coupfe/runtime/elements/neo_hookean_q4.for",
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
        "test_distribute.py",
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
        "test_curved_convergence.py",
        "test_contact_semismooth_friction.py",
        "test_contact_vs_ppf.py",
        "test_exact_stick_friction.py",
        "test_examples_contact_3d.py",
        "test_friction_relay_finite_sliding.py",
        "test_hertz_contact.py",
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
    "examples/morphing_hex8/reference_result.json",
    "examples/phasefield_corrosion_cui/README.md",
    "examples/phasefield_corrosion_cui/build.py",
    "examples/phasefield_corrosion_cui/paper_example_record.json",
    "examples/phasefield_corrosion_cui/phasefield_corrosion_cui_full_uel.for",
    "examples/stabilized_tet4/README.md",
    "examples/stabilized_tet4/build.py",
    "examples/stabilized_tet4/reference_result.json",
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

REQUIRED_SDIST_FILES = (
    PACKAGE_ASSETS
    | PUBLIC_TEST_FILES
    | PUBLIC_TEST_SUPPORT_FILES
    | PAPER_EXAMPLE_FILES
    | UMAT_EXAMPLE_FILES
    | {
    ".github/scripts/check_release_artifacts.py",
    "LICENSE",
    "LICENSE-DOCS.md",
    "LICENSE-ABAQUS-UFL-EXAMPLES",
    "NOTICE",
    "README.md",
    "MANIFEST.in",
    "pyproject.toml",
    "docs/capabilities.md",
    "examples/README.md",
    "examples/REFERENCES.md",
    "examples/ring_compress/README.md",
    "examples/tire_contact/README.md",
    "skills/SKILL.md",
    "validation/README.md",
    }
)

WITHHELD_EXAMPLE_DIRS = {
    "Fbar_uel",
    "cattaneo_3d",
    "gel_axisymmetric_quad8",
    "gel_chester_anand_local_pressure_quad4",
    "gel_three_field_hex20",
    "hussein_2026_ductile_pff",
    "hussein_2026_mediavilla_pff",
    "j2_fefp_uel",
    "lce_quad4",
    "li_2026_battery",
    "strain_gradient_plasticity_msg",
    "self_contact_friction",
}
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

FORBIDDEN_PARTS = {
    ".git",
    ".mypy_cache",
    ".numba_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "EXTERNAL_OBSERVER_FINDINGS.md",
    "HISTORY_pqp_cattaneo_benchmark.md",
    "NOTE_penalty_petsc_result_for_claude.md",
    "RELEASE_READINESS.md",
    "REVIEW_NOTES_for_gpt.md",
    "SUMMARY_for_gpt_petsc_solve.md",
    "contact_pqp.py",
    "pqp_hertz_cattaneo",
    "test_contact_pqp_friction.py",
    "test_pqp_hertz_cattaneo.py",
}
FORBIDDEN_NAME_PATTERNS = {
    "HANDOFF*.md",
    "NOTE_TO_*.md",
    "PORT_PROMPT*.md",
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
    ".cfg",
    ".csv",
    ".f90",
    ".for",
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
        # Catch contact_pqp_utils.py, pqp-notes/, and other near-name variants,
        # not just the exact historical filenames above.
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


def _reject_withheld_examples(names: set[str], artifact: Path) -> None:
    """Keep provenance-blocked examples out of release artifacts."""

    rejected = sorted(
        name
        for name in names
        if len(PurePosixPath(name).parts) >= 2
        and PurePosixPath(name).parts[0] == "examples"
        and PurePosixPath(name).parts[1] in WITHHELD_EXAMPLE_DIRS
    )
    if rejected:
        raise SystemExit(
            f"{artifact.name} contains first-release-withheld examples: {rejected}"
        )


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
                and name != "validation/README.md"
            )
        )
    )
    if rejected:
        raise SystemExit(
            f"{artifact.name} contains non-public validation harness files: "
            f"{rejected}"
        )


def _require_shipped_examples(names: set[str], artifact: Path) -> None:
    expected = READY_EXAMPLE_DIRS | RESEARCH_EXAMPLE_DIRS
    present = {
        path.parts[1]
        for name in names
        if len((path := PurePosixPath(name)).parts) >= 3
        and path.parts[0] == "examples"
        and path.parts[1] in expected
    }
    missing = sorted(expected - present)
    if missing:
        raise SystemExit(
            f"{artifact.name} is missing shipped example directories: {missing}"
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
    if re.search(r"[\w.+-]+@(?:gmail\.com|syr\.edu)\b", text, flags=re.IGNORECASE):
        hits.append("personal-email-address")
    if hits:
        raise SystemExit(
            f"{artifact.name}:{name} contains private-path/credential material: "
            f"{sorted(hits)}"
        )


def _validate_example_policy(source_root: Path) -> set[str]:
    """Keep the inventory aligned while allowing a filtered public example tree."""

    example_root = source_root / "examples"
    directories = {
        path.name
        for path in example_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    ledger = (example_root / "REFERENCES.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `([^`]+)` \| \*\*(READY|RESEARCH|WITHHELD)",
        ledger,
        flags=re.MULTILINE,
    )
    row_names = [name for name, _status in rows]
    duplicates = sorted(
        name for name, count in Counter(row_names).items() if count > 1
    )
    ledger_names = set(row_names)
    if duplicates or directories - ledger_names:
        raise SystemExit(
            "example reference inventory mismatch: "
            f"duplicates={duplicates}, "
            f"unlisted_directories={sorted(directories - ledger_names)}"
        )
    ledger_withheld = {name for name, status in rows if status == "WITHHELD"}
    ledger_research = {name for name, status in rows if status == "RESEARCH"}
    if ledger_withheld != WITHHELD_EXAMPLE_DIRS:
        raise SystemExit(
            "withheld-example policy mismatch: "
            f"ledger_only={sorted(ledger_withheld - WITHHELD_EXAMPLE_DIRS)}, "
            f"guard_only={sorted(WITHHELD_EXAMPLE_DIRS - ledger_withheld)}"
        )
    if ledger_research != RESEARCH_EXAMPLE_DIRS:
        raise SystemExit(
            "research-example policy mismatch: "
            f"ledger_only={sorted(ledger_research - RESEARCH_EXAMPLE_DIRS)}, "
            f"guard_only={sorted(RESEARCH_EXAMPLE_DIRS - ledger_research)}"
        )
    ledger_ready = ledger_names - ledger_withheld - ledger_research
    if ledger_ready != READY_EXAMPLE_DIRS:
        raise SystemExit(
            "ready-example policy mismatch: "
            f"ledger_only={sorted(ledger_ready - READY_EXAMPLE_DIRS)}, "
            f"guard_only={sorted(READY_EXAMPLE_DIRS - ledger_ready)}"
        )
    missing_ready = sorted(ledger_ready - directories)
    if missing_ready:
        raise SystemExit(
            f"example reference inventory is missing READY directories: {missing_ready}"
        )
    return directories.intersection(ledger_withheld)


def _validate_source_tree(
    source_root: Path,
    *,
    allow_withheld_source: bool = False,
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
    for name in sorted(files):
        path = source_root / PurePosixPath(name)
        if path.is_file():
            _validate_text(name, path.read_bytes(), source_root)
    present_withheld = _validate_example_policy(source_root)
    if present_withheld:
        message = (
            f"{source_root} still contains first-release-withheld example "
            f"directories: {sorted(present_withheld)}"
        )
        if not allow_withheld_source:
            raise SystemExit(message)
        print(f"WARNING: {message} (audit override enabled)")
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
    required = PACKAGE_ASSETS | {
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/LICENSE-DOCS.md",
        f"{dist_info}/licenses/LICENSE-ABAQUS-UFL-EXAMPLES",
        f"{dist_info}/licenses/NOTICE",
    }
    _require_files(files, required, wheel)
    _reject_forbidden_files(files, wheel)
    _reject_withheld_examples(files, wheel)
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
    _reject_withheld_examples(files, sdist)
    _reject_private_harness(files, sdist)
    _require_shipped_examples(files, sdist)
    with tarfile.open(sdist, mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise SystemExit(f"{sdist.name}:{member.name} could not be read")
                relative = PurePosixPath(
                    *PurePosixPath(member.name).parts[1:]
                ).as_posix()
                _validate_text(relative, stream.read(), sdist)
    return len(files)


def validate(
    dist_dir: Path,
    source_root: Path | None = None,
    *,
    allow_withheld_source: bool = False,
    allow_untracked_required: bool = False,
    allow_dirty_source: bool = False,
    allow_indexed_rights_blocked_deletion: bool = False,
) -> None:
    source_count = (
        _validate_source_tree(
            source_root.resolve(),
            allow_withheld_source=allow_withheld_source,
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
        "--allow-withheld-source-for-audit",
        action="store_true",
        help=(
            "inspect artifacts while private staging still contains WITHHELD "
            "examples; the source result is not publishable"
        ),
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
        allow_withheld_source=args.allow_withheld_source_for_audit,
        allow_untracked_required=args.allow_untracked_required_for_audit,
        allow_dirty_source=args.allow_dirty_source_for_audit,
        allow_indexed_rights_blocked_deletion=(
            args.allow_indexed_rights_blocked_deletion_for_audit
        ),
    )


if __name__ == "__main__":
    main()
