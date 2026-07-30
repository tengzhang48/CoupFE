"""Tests for the example-local pasta Abaqus input mesh reader."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_READER_PY = _ROOT / "examples" / "morphing_hex8" / "abaqus_mesh.py"
_FULL_DECK_ENV = "COUPFE_MORPHING_ABAQUS_INP"


def _load_reader():
    spec = importlib.util.spec_from_file_location(
        "morphing_hex8_abaqus_mesh", _READER_PY
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_deck(tmp_path: Path) -> Path:
    companion = tmp_path / "companion_connectivity.inp"
    companion.write_text(
        "1000100, 10, 20, 30, 40, 50, 60, 70, 80\n"
    )
    main = tmp_path / "pasta.inp"
    main.write_text(
        """\
*Heading
tiny label-preserving pasta mesh
*Node
10, 0.0, 0.0, 0.0
20, 1.0, 0.0, 0.0
30, 1.0, 1.0, 0.0
40, 0.0, 1.0, 0.0
50, 0.0, 0.0, 1.0
60, 1.0, 0.0, 1.0
70, 1.0, 1.0, 1.0
80, 0.0, 1.0, 1.0
*User Element, Nodes=8, Type=U3
*Element, type=U3, elset=GelFromHeader
100, 10, 20, 30, 40, 50, 60, 70, 80
*Nset, nset=Corners
10, 30, 50, 70
*Nset, nset=AllPhysical, generate
10, 80, 10
*Elset, elset=Gel, generate
100, 100, 1
*Element, type=C3D8
*Include, input="companion_connectivity.inp"
*Elset, elset=elDummy
1000100
"""
    )
    return main


def test_synthetic_mesh_preserves_labels_sets_and_include(tmp_path):
    reader = _load_reader()
    main = _write_synthetic_deck(tmp_path)
    mesh = reader.parse_pasta_mesh(main)

    assert tuple(mesh.nodes) == (10, 20, 30, 40, 50, 60, 70, 80)
    expected_connectivity = (10, 20, 30, 40, 50, 60, 70, 80)
    assert mesh.uel_elements[100] == expected_connectivity
    assert mesh.companion_elements[1000100] == expected_connectivity
    assert mesh.node_set("corners") == (10, 30, 50, 70)
    assert mesh.node_set("ALLPHYSICAL") == tuple(range(10, 81, 10))
    assert mesh.element_set("gel") == (100,)
    assert mesh.element_set("gelfromheader") == (100,)
    assert mesh.element_set("ELDUMMY") == (1000100,)
    assert mesh.included_mesh_files == (
        (tmp_path / "companion_connectivity.inp").resolve(),
    )


def test_unknown_connectivity_label_is_a_consistency_error(tmp_path):
    reader = _load_reader()
    main = _write_synthetic_deck(tmp_path)
    text = main.read_text().replace(
        "100, 10, 20, 30, 40, 50, 60, 70, 80",
        "100, 10, 20, 30, 40, 50, 60, 70, 999",
    )
    main.write_text(text)

    with pytest.raises(
        reader.AbaqusMeshConsistencyError,
        match=r"unknown node label\(s\): 999",
    ):
        reader.parse_pasta_mesh(main)


def test_scoping_and_coordinate_transforms_fail_explicitly(tmp_path):
    reader = _load_reader()
    deck = tmp_path / "scoped.inp"
    deck.write_text("*Part, name=Pasta\n*End Part\n")

    with pytest.raises(
        reader.UnsupportedAbaqusMeshError,
        match="flat pasta-mesh layout",
    ):
        reader.parse_pasta_mesh(deck)

    deck.write_text(
        "*Node, system=C\n"
        "1, 1.0, 45.0, 0.0\n"
    )
    with pytest.raises(
        reader.UnsupportedAbaqusMeshError,
        match="coordinate transforms are unsupported",
    ):
        reader.parse_pasta_mesh(deck)


if os.environ.get(_FULL_DECK_ENV):

    def test_opt_in_public_pasta_deck_inventory():
        """Read-only inventory gate for a supplied full public deck."""

        reader = _load_reader()
        mesh = reader.parse_pasta_mesh(os.environ[_FULL_DECK_ENV])

        assert len(mesh.nodes) == 61472  # 61,464 physical + 8 control nodes
        assert len(mesh.uel_elements) == 54000
        assert len(mesh.companion_elements) == 54000
        assert len(mesh.elements_by_type["C3D8T"]) == 1
        assert len(mesh.node_set("gel")) == 61464
        assert len(mesh.element_set("gel")) == 54000
        assert len(mesh.element_set("elDummy")) == 54000
        assert mesh.uel_elements[1] == mesh.companion_elements[1000001]
        assert mesh.uel_elements[54000] == mesh.companion_elements[1054000]
        assert all(
            mesh.companion_elements[label + 1000000] == connectivity
            for label, connectivity in mesh.uel_elements.items()
        )
