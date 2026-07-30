"""Narrow Abaqus-input mesh reader for the pasta-morphing paper example.

This is deliberately example-local, not an installed CoupFE input API.  It
supports the flat mesh layout used by the public paper deck:

* three-coordinate ``*NODE`` records;
* eight-node ``U3``, ``C3D8``, and ``C3D8T`` element records;
* numeric ``*NSET``/``*ELSET`` records, including ``GENERATE``; and
* a relative ``*INCLUDE`` that continues one of those mesh-data blocks.

It does not interpret parts, instances, assemblies, coordinate systems,
parameterized mesh values, set-name composition, materials, steps, contact, or
solver controls. Unsupported mesh constructs fail explicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Set, Tuple


Coordinates = Tuple[float, float, float]
Connectivity = Tuple[int, ...]

_ELEMENT_NODE_COUNTS = {
    "U3": 8,
    "C3D8": 8,
    "C3D8T": 8,
}
_UNSUPPORTED_SCOPE_KEYWORDS = {
    "*assembly",
    "*end assembly",
    "*instance",
    "*end instance",
    "*part",
    "*end part",
    "*system",
}
_MESH_DATA_MODES = {"node", "element", "nset", "elset"}


class AbaqusMeshError(ValueError):
    """Base class for explicit mesh-reader failures."""


class UnsupportedAbaqusMeshError(AbaqusMeshError):
    """The deck uses an Abaqus construct outside this reader's scope."""


class AbaqusMeshConsistencyError(AbaqusMeshError):
    """Parsed labels or connectivity are internally inconsistent."""


@dataclass(frozen=True)
class PastaMesh:
    """Label-preserving mesh data extracted from one flat Abaqus deck."""

    source: Path
    nodes: Mapping[int, Coordinates]
    elements_by_type: Mapping[str, Mapping[int, Connectivity]]
    node_sets: Mapping[str, Tuple[int, ...]]
    element_sets: Mapping[str, Tuple[int, ...]]
    included_mesh_files: Tuple[Path, ...]

    @property
    def uel_elements(self) -> Mapping[int, Connectivity]:
        """The pressure-gel U3 analysis elements, keyed by Abaqus label."""

        return self.elements_by_type.get("U3", MappingProxyType({}))

    @property
    def companion_elements(self) -> Mapping[int, Connectivity]:
        """The node-sharing C3D8 companion elements, kept separate from U3."""

        return self.elements_by_type.get("C3D8", MappingProxyType({}))

    def node_set(self, name: str) -> Tuple[int, ...]:
        """Return a named node set using Abaqus-style case-insensitive lookup."""

        return _case_insensitive_lookup(self.node_sets, name, "node set")

    def element_set(self, name: str) -> Tuple[int, ...]:
        """Return an element set using Abaqus-style case-insensitive lookup."""

        return _case_insensitive_lookup(
            self.element_sets, name, "element set"
        )


@dataclass
class _ParseState:
    nodes: Dict[int, Coordinates]
    elements_by_type: Dict[str, Dict[int, Connectivity]]
    node_sets: Dict[str, List[int]]
    element_sets: Dict[str, List[int]]
    set_names: Dict[Tuple[str, str], str]
    included_mesh_files: List[Path]
    include_stack: List[Path]
    mode: Optional[str] = None
    element_type: Optional[str] = None
    element_set_from_header: Optional[str] = None
    active_set_name: Optional[str] = None
    active_set_generate: bool = False


def _case_insensitive_lookup(mapping, name: str, kind: str):
    folded = name.casefold()
    for existing, value in mapping.items():
        if existing.casefold() == folded:
            return value
    raise KeyError(f"unknown {kind} {name!r}")


def _location(path: Path, line_number: int) -> str:
    return f"{path}:{line_number}"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _keyword_parts(line: str):
    tokens = [token.strip() for token in line.split(",")]
    keyword = tokens[0].casefold()
    options = {}
    flags = set()
    for token in tokens[1:]:
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            options[key.strip().casefold()] = _unquote(value)
        else:
            flags.add(token.casefold())
    return keyword, options, flags


def _parse_ints(text: str, path: Path, line_number: int) -> List[int]:
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise UnsupportedAbaqusMeshError(
            f"{_location(path, line_number)}: expected numeric Abaqus labels; "
            "set-name composition and parameterized mesh data are unsupported"
        ) from exc


def _set_bucket(
    state: _ParseState, kind: str, name: str
) -> Tuple[str, List[int]]:
    key = (kind, name.casefold())
    canonical = state.set_names.get(key)
    target = state.node_sets if kind == "nset" else state.element_sets
    if canonical is None:
        canonical = name
        state.set_names[key] = canonical
        target[canonical] = []
    return canonical, target[canonical]


def _append_set_values(
    state: _ParseState,
    values: List[int],
    path: Path,
    line_number: int,
) -> None:
    if state.active_set_name is None or state.mode not in {"nset", "elset"}:
        raise AssertionError("set parser called without an active set")
    _, bucket = _set_bucket(
        state, state.mode, state.active_set_name
    )
    if not state.active_set_generate:
        bucket.extend(values)
        return

    if not values or len(values) % 3:
        raise AbaqusMeshError(
            f"{_location(path, line_number)}: GENERATE data must contain "
            "start, stop, increment triplets"
        )
    for index in range(0, len(values), 3):
        start, stop, increment = values[index:index + 3]
        if increment == 0:
            raise AbaqusMeshError(
                f"{_location(path, line_number)}: GENERATE increment is zero"
            )
        if (stop - start) * increment < 0:
            raise AbaqusMeshError(
                f"{_location(path, line_number)}: GENERATE increment points "
                "away from the stop label"
            )
        if (stop - start) % increment:
            raise AbaqusMeshError(
                f"{_location(path, line_number)}: GENERATE stop label is not "
                "reached by the increment"
            )
        bucket.extend(range(start, stop + (1 if increment > 0 else -1),
                            increment))


def _start_keyword(
    state: _ParseState,
    keyword: str,
    options: Mapping[str, str],
    flags: Set[str],
    path: Path,
    line_number: int,
) -> None:
    state.mode = None
    state.element_type = None
    state.element_set_from_header = None
    state.active_set_name = None
    state.active_set_generate = False

    if keyword in _UNSUPPORTED_SCOPE_KEYWORDS:
        raise UnsupportedAbaqusMeshError(
            f"{_location(path, line_number)}: {keyword.upper()} scoping or "
            "coordinate transforms are unsupported; this reader accepts only "
            "the flat pasta-mesh layout"
        )

    if keyword == "*node":
        if "nset" in options:
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: *NODE,NSET is unsupported; "
                "declare a separate numeric *NSET"
            )
        if "system" in options:
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: *NODE,SYSTEM coordinate "
                "transforms are unsupported"
            )
        state.mode = "node"
        return

    if keyword == "*element":
        element_type = options.get("type", "").upper()
        if not element_type:
            raise AbaqusMeshError(
                f"{_location(path, line_number)}: *ELEMENT requires TYPE"
            )
        if element_type not in _ELEMENT_NODE_COUNTS:
            supported = ", ".join(_ELEMENT_NODE_COUNTS)
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: element type "
                f"{element_type!r} is outside this reader's scope "
                f"(supported: {supported})"
            )
        state.mode = "element"
        state.element_type = element_type
        state.elements_by_type.setdefault(element_type, {})
        if "elset" in options:
            canonical, _ = _set_bucket(
                state, "elset", options["elset"]
            )
            state.element_set_from_header = canonical
        return

    if keyword in {"*nset", "*elset"}:
        kind = keyword[1:]
        name = options.get(kind)
        if not name:
            raise AbaqusMeshError(
                f"{_location(path, line_number)}: {keyword.upper()} requires "
                f"{kind.upper()}=name"
            )
        if "instance" in options:
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: instance-scoped sets are "
                "unsupported"
            )
        canonical, _ = _set_bucket(state, kind, name)
        state.mode = kind
        state.active_set_name = canonical
        state.active_set_generate = "generate" in flags


def _parse_data_line(
    state: _ParseState, text: str, path: Path, line_number: int
) -> None:
    if state.mode is None:
        return

    if state.mode == "node":
        tokens = [token.strip() for token in text.split(",")
                  if token.strip()]
        if len(tokens) != 4:
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: pasta nodes require exactly "
                "one label and three coordinates"
            )
        try:
            label = int(tokens[0])
            coordinates = tuple(float(value) for value in tokens[1:])
        except ValueError as exc:
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: parameterized or nonnumeric "
                "node data are unsupported"
            ) from exc
        if label in state.nodes:
            raise AbaqusMeshConsistencyError(
                f"{_location(path, line_number)}: duplicate node label {label}"
            )
        state.nodes[label] = coordinates
        return

    values = _parse_ints(text, path, line_number)
    if state.mode in {"nset", "elset"}:
        _append_set_values(state, values, path, line_number)
        return

    if state.mode == "element":
        if state.element_type is None:
            raise AssertionError("element parser called without an active type")
        expected = _ELEMENT_NODE_COUNTS[state.element_type]
        if len(values) != expected + 1:
            raise UnsupportedAbaqusMeshError(
                f"{_location(path, line_number)}: {state.element_type} "
                f"requires one label and {expected} node labels; element "
                "continuation records are unsupported"
            )
        label = values[0]
        if any(
            label in block for block in state.elements_by_type.values()
        ):
            raise AbaqusMeshConsistencyError(
                f"{_location(path, line_number)}: duplicate element label "
                f"{label}"
            )
        state.elements_by_type[state.element_type][label] = tuple(values[1:])
        if state.element_set_from_header is not None:
            state.element_sets[state.element_set_from_header].append(label)


def _parse_file(path: Path, state: _ParseState) -> None:
    resolved = path.expanduser().resolve()
    if resolved in state.include_stack:
        chain = " -> ".join(
            str(item) for item in state.include_stack + [resolved]
        )
        raise AbaqusMeshError(f"cyclic *INCLUDE chain: {chain}")
    if not resolved.is_file():
        raise AbaqusMeshError(f"Abaqus input file not found: {resolved}")

    state.include_stack.append(resolved)
    try:
        with resolved.open(encoding="utf-8", errors="strict") as stream:
            for line_number, raw in enumerate(stream, start=1):
                text = raw.strip()
                if not text or text.startswith("**"):
                    continue
                if not text.startswith("*"):
                    _parse_data_line(
                        state, text, resolved, line_number
                    )
                    continue

                keyword, options, flags = _keyword_parts(text)
                if keyword == "*include":
                    include_name = options.get("input")
                    if not include_name:
                        raise AbaqusMeshError(
                            f"{_location(resolved, line_number)}: *INCLUDE "
                            "requires INPUT=path"
                        )
                    # The paper deck needs only the bare C3D8 connectivity
                    # included within an active *ELEMENT block. Material
                    # property includes are intentionally outside this reader.
                    if state.mode in _MESH_DATA_MODES:
                        include_path = (resolved.parent / include_name).resolve()
                        state.included_mesh_files.append(include_path)
                        _parse_file(include_path, state)
                    continue

                _start_keyword(
                    state,
                    keyword,
                    options,
                    flags,
                    resolved,
                    line_number,
                )
    finally:
        state.include_stack.pop()


def _deduplicate(labels: List[int]) -> Tuple[int, ...]:
    return tuple(dict.fromkeys(labels))


def _validate(state: _ParseState, source: Path) -> None:
    if not state.nodes:
        raise AbaqusMeshConsistencyError(f"{source}: no nodes were parsed")
    if not state.elements_by_type.get("U3"):
        raise AbaqusMeshConsistencyError(f"{source}: no U3 elements were parsed")

    all_element_labels = {
        label
        for block in state.elements_by_type.values()
        for label in block
    }
    missing_connectivity = sorted({
        node
        for block in state.elements_by_type.values()
        for connectivity in block.values()
        for node in connectivity
        if node not in state.nodes
    })
    if missing_connectivity:
        preview = ", ".join(map(str, missing_connectivity[:8]))
        raise AbaqusMeshConsistencyError(
            f"{source}: element connectivity references unknown node "
            f"label(s): {preview}"
        )

    for name, labels in state.node_sets.items():
        missing = sorted(set(labels).difference(state.nodes))
        if missing:
            preview = ", ".join(map(str, missing[:8]))
            raise AbaqusMeshConsistencyError(
                f"{source}: node set {name!r} references unknown node "
                f"label(s): {preview}"
            )
    for name, labels in state.element_sets.items():
        missing = sorted(set(labels).difference(all_element_labels))
        if missing:
            preview = ", ".join(map(str, missing[:8]))
            raise AbaqusMeshConsistencyError(
                f"{source}: element set {name!r} references unknown element "
                f"label(s): {preview}"
            )


def parse_pasta_mesh(path) -> PastaMesh:
    """Parse and validate the flat pasta-morphing Abaqus mesh.

    Numeric Abaqus node and element labels are preserved as dictionary keys.
    The returned ``uel_elements`` and ``companion_elements`` mappings expose
    U3 and C3D8 blocks separately; no conversion to CoupFE indexing is made.
    """

    source = Path(path).expanduser().resolve()
    state = _ParseState(
        nodes={},
        elements_by_type={},
        node_sets={},
        element_sets={},
        set_names={},
        included_mesh_files=[],
        include_stack=[],
    )
    _parse_file(source, state)
    _validate(state, source)

    frozen_blocks = {
        element_type: MappingProxyType(dict(elements))
        for element_type, elements in state.elements_by_type.items()
    }
    return PastaMesh(
        source=source,
        nodes=MappingProxyType(dict(state.nodes)),
        elements_by_type=MappingProxyType(frozen_blocks),
        node_sets=MappingProxyType({
            name: _deduplicate(labels)
            for name, labels in state.node_sets.items()
        }),
        element_sets=MappingProxyType({
            name: _deduplicate(labels)
            for name, labels in state.element_sets.items()
        }),
        included_mesh_files=tuple(state.included_mesh_files),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the flat Abaqus mesh for the morphing Hex8 example"
    )
    parser.add_argument("input", type=Path, help="main Abaqus .inp deck")
    args = parser.parse_args()
    mesh = parse_pasta_mesh(args.input)
    other_count = sum(
        len(elements)
        for element_type, elements in mesh.elements_by_type.items()
        if element_type not in {"U3", "C3D8"}
    )
    print(f"nodes: {len(mesh.nodes)}")
    print(f"U3 analysis elements: {len(mesh.uel_elements)}")
    print(f"C3D8 companion elements: {len(mesh.companion_elements)}")
    print(f"other supported elements: {other_count}")
    print(f"node sets: {', '.join(mesh.node_sets)}")
    print(f"element sets: {', '.join(mesh.element_sets)}")


if __name__ == "__main__":
    _main()
