"""Render the retained 3-D block collision as a deterministic technical SVG.

The visible geometry is the actual accepted minimum-signed-gap displacement
returned by :func:`run_collision_snapshot`.  Face color represents nodal
displacement magnitude only; no stress or continuous contact field is inferred.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.contact_3d_blocks.run import run_collision_snapshot

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _ROOT / "docs" / "assets" / "contact-3d-blocks-snapshot.svg"

_INK = "#17222B"
_MUTED = "#5C6B76"
_GRID = "#D8E0E5"
_TEAL = "#155E75"
_TEAL_DARK = "#0B3E52"
_COPPER = "#CF6945"
_COPPER_DARK = "#7B3F2A"
_FIELD_COLORS = (
    "#F2F6F7",
    "#DCEAED",
    "#BFDADF",
    "#91C2CC",
    "#5EA5B5",
    "#2F8194",
    "#155E75",
)
_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def _num(value):
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _points(values):
    return " ".join(f"{_num(x)},{_num(y)}" for x, y in values)


def _field_color(value, maximum):
    fraction = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
    index = min(int(fraction * len(_FIELD_COLORS)), len(_FIELD_COLORS) - 1)
    return _FIELD_COLORS[index]


def _boundary_faces(elements):
    """Return exterior Hex8 faces in a stable element/local-face order."""

    elements = np.asarray(elements, dtype=int)
    occurrences = {}
    ordered = []
    for element_index, element in enumerate(elements):
        for local_index, local_face in enumerate(_HEX_FACES):
            face = tuple(int(element[index]) for index in local_face)
            key = tuple(sorted(face))
            occurrences[key] = occurrences.get(key, 0) + 1
            ordered.append((element_index, local_index, key, face))
    return [
        face
        for _element_index, _local_index, key, face in ordered
        if occurrences[key] == 1
    ]


def _projection(points, box):
    """Return projected coordinates, view depth, and a reusable mapping."""

    points = np.asarray(points, dtype=float)
    raw = np.column_stack(
        (
            points[:, 0] - points[:, 1],
            0.30 * (points[:, 0] + points[:, 1]) - 0.64 * points[:, 2],
        )
    )
    raw_min = raw.min(axis=0)
    raw_max = raw.max(axis=0)
    span = np.maximum(raw_max - raw_min, 1.0e-12)
    x, y, width, height = (float(value) for value in box)
    scale = min(width / span[0], height / span[1])
    offset = np.array(
        (
            x + 0.5 * (width - scale * span[0]),
            y + 0.5 * (height - scale * span[1]),
        )
    ) - scale * raw_min

    def project(values):
        values = np.asarray(values, dtype=float)
        plane = np.column_stack(
            (
                values[:, 0] - values[:, 1],
                0.30 * (values[:, 0] + values[:, 1]) - 0.64 * values[:, 2],
            )
        )
        return offset + scale * plane

    depth = points @ np.array([1.0, 1.0, 0.94])
    return project(points), depth, project


def render_collision_svg(evidence, output_path=DEFAULT_OUTPUT):
    """Write a 720×480 SVG from one solved collision evidence record."""

    config = evidence["configuration"]
    snapshot = evidence["snapshot"]
    results = evidence["results"]
    nodes = np.asarray(snapshot["nodes_deformed"], dtype=float)
    displacement = np.asarray(snapshot["displacement"], dtype=float)
    elements = np.asarray(snapshot["elements"], dtype=int)
    secondary = np.asarray(snapshot["secondary_nodes"], dtype=int)
    magnitude = np.linalg.norm(displacement, axis=1)
    maximum = float(np.max(magnitude))
    projected, depth, _project = _projection(nodes, (42.0, 105.0, 405.0, 270.0))

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" '
        'viewBox="0 0 720 480" role="img" aria-labelledby="title description">',
        '<title id="title">Deformable Hex8 block collision at the accepted minimum-gap frame</title>',
        '<desc id="description">Two deformable Hex8 blocks at the accepted trajectory frame '
        'with the smallest positive signed gap. Exterior faces are colored by actual '
        'displacement magnitude and interface nodes are marked in copper.</desc>',
        '<rect width="720" height="480" fill="#FFFFFF"/>',
        f'<text x="34" y="38" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="24" font-weight="650">Deformable block collision</text>',
        f'<text x="34" y="65" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="16">Accepted step {snapshot["accepted_step"]} of {config["accepted_steps"]} '
        f'· t = {snapshot["time"]:.2f} · deformation ×1</text>',
        f'<line x1="34" y1="82" x2="686" y2="82" stroke="{_GRID}"/>',
        '<rect x="30" y="94" width="430" height="296" rx="4" fill="#F8FAFA" '
        f'stroke="{_GRID}"/>',
    ]

    faces = _boundary_faces(elements)
    face_order = sorted(
        range(len(faces)),
        key=lambda index: (
            round(float(np.mean(depth[np.asarray(faces[index], dtype=int)])), 12),
            tuple(sorted(faces[index])),
        ),
    )
    for index in face_order:
        face_nodes = np.asarray(faces[index], dtype=int)
        value = float(np.mean(magnitude[face_nodes]))
        svg.append(
            f'<polygon points="{_points(projected[face_nodes])}" '
            f'fill="{_field_color(value, maximum)}" stroke="#71858C" '
            'stroke-width="0.75" stroke-linejoin="round"/>'
        )

    marker_order = sorted(
        range(len(secondary)),
        key=lambda index: (
            round(float(projected[secondary[index], 1]), 2),
            round(float(projected[secondary[index], 0]), 2),
            int(secondary[index]),
        ),
    )
    for index in marker_order:
        x, y = projected[secondary[index]]
        svg.append(
            f'<circle cx="{_num(x)}" cy="{_num(y)}" r="2.8" '
            f'fill="{_COPPER}" stroke="{_COPPER_DARK}" stroke-width="0.8"/>'
        )

    svg.extend(
        [
            f'<text x="482" y="122" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14" font-weight="700" letter-spacing="1">TRAJECTORY CHECK</text>',
            f'<text x="482" y="157" fill="{_INK}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" '
            f'font-size="24" font-weight="700">{results["minimum_signed_gap"]:.3e}</text>',
            f'<text x="482" y="180" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="15">minimum signed gap</text>',
            f'<line x1="482" y1="202" x2="682" y2="202" stroke="{_GRID}"/>',
            f'<text x="482" y="230" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="16" font-weight="650">d̂ = {config["activation_distance"]:.3f}</text>',
            f'<text x="482" y="252" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">barrier activation distance</text>',
            f'<text x="482" y="294" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="16" font-weight="650">{config["elements"]} Hex8</text>',
            f'<text x="482" y="316" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">two deformable blocks</text>',
            f'<circle cx="489" cy="353" r="4" fill="{_COPPER}" stroke="{_COPPER_DARK}"/>',
            f'<text x="502" y="358" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">secondary interface nodes</text>',
        ]
    )

    legend_x, legend_y, swatch_width = 42.0, 415.0, 30.0
    for index, color in enumerate(_FIELD_COLORS):
        svg.append(
            f'<rect x="{_num(legend_x + index * swatch_width)}" y="{legend_y}" '
            f'width="{swatch_width}" height="11" fill="{color}" stroke="#FFFFFF" stroke-width="0.5"/>'
        )
    svg.extend(
        [
            f'<text x="42" y="405" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="14">face color: displacement magnitude |u| · 0 to {maximum:.4f}</text>',
            f'<line x1="34" y1="444" x2="686" y2="444" stroke="{_GRID}"/>',
            f'<text x="34" y="468" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="13">Checked nodal displacement field · positive-gap accepted frame · no smoothed interface field</text>',
            '</svg>',
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def main():
    evidence = run_collision_snapshot()
    output = render_collision_svg(evidence)
    print(f"wrote {output.relative_to(_ROOT)}")
    return output


if __name__ == "__main__":
    main()
