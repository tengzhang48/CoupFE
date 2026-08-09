"""Render the checked Hertz solve as a dependency-free technical SVG.

The field panel uses the actual final Hex8 displacement and discrete nodal
contact reactions returned by :func:`examples.hertz_contact.run.run_hertz`.
It does not reconstruct or imply a continuous contact-pressure field.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from examples.hertz_contact.run import hertz_force, run_hertz

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _ROOT / "docs" / "assets" / "hertz-contact-benchmark.svg"

_INK = "#17222B"
_MUTED = "#5C6B76"
_GRID = "#D8E0E5"
_BLUE = "#155E75"
_BLUE_DARK = "#0B3E52"
_GOLD = "#D5A62E"
_GOLD_DARK = "#6A5315"
_FIELD_COLORS = (
    "#F2F6F7",
    "#DCEAED",
    "#BFDADF",
    "#91C2CC",
    "#5EA5B5",
    "#2F8194",
    "#155E75",
)


def _num(value):
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _points(values):
    return " ".join(f"{_num(x)},{_num(y)}" for x, y in values)


def _marker_order(projected, node_ids):
    """Return a deterministic painter's order at the SVG output precision."""

    projected = np.asarray(projected, dtype=float)
    node_ids = np.asarray(node_ids, dtype=int)
    if projected.size == 0:
        return np.empty(0, dtype=int)
    if projected.shape != (len(node_ids), 2):
        raise ValueError("projected markers and node IDs must have matching rows")

    # ``_num`` emits two decimal places.  Sorting the same rounded coordinates
    # prevents sub-rendering-precision floating-point noise from swapping
    # symmetric markers and changing the retained SVG hash.  Node ID is the
    # final tie-breaker when both rendered coordinates coincide.
    rendered = np.round(projected, decimals=2)
    return np.lexsort((node_ids, rendered[:, 0], rendered[:, 1]))


def _project(values, center_xy=(1.25, 1.25)):
    """Orthographic engineering projection used by the field panel."""

    values = np.asarray(values, dtype=float)
    centered = values - np.array([center_xy[0], center_xy[1], 0.0])
    projected_x = 365.0 + 95.0 * (0.82 * centered[:, 0] - 0.82 * centered[:, 1])
    projected_y = 455.0 + 80.0 * (
        0.34 * centered[:, 0] + 0.34 * centered[:, 1] - centered[:, 2]
    )
    return np.column_stack((projected_x, projected_y))


def _field_color(value, maximum):
    fraction = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
    index = min(int(fraction * len(_FIELD_COLORS)), len(_FIELD_COLORS) - 1)
    return _FIELD_COLORS[index]


def _node_id(i, j, k, nx, ny):
    return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i


def _visible_faces(nx, ny, nz):
    """Return the two visible side grids followed by the loaded top grid."""

    faces = []
    for k in range(nz):
        for i in range(nx):
            faces.append(
                (
                    _node_id(i, 0, k, nx, ny),
                    _node_id(i + 1, 0, k, nx, ny),
                    _node_id(i + 1, 0, k + 1, nx, ny),
                    _node_id(i, 0, k + 1, nx, ny),
                )
            )
    for k in range(nz):
        for j in range(ny):
            faces.append(
                (
                    _node_id(nx, j, k, nx, ny),
                    _node_id(nx, j + 1, k, nx, ny),
                    _node_id(nx, j + 1, k + 1, nx, ny),
                    _node_id(nx, j, k + 1, nx, ny),
                )
            )
    for j in range(ny):
        for i in range(nx):
            faces.append(
                (
                    _node_id(i, j, nz, nx, ny),
                    _node_id(i + 1, j, nz, nx, ny),
                    _node_id(i + 1, j + 1, nz, nx, ny),
                    _node_id(i, j + 1, nz, nx, ny),
                )
            )
    return faces


def _sphere_cap_paths(center, radius, center_xy):
    """Project a local analytical sphere-cap wireframe around the contact patch."""

    paths = []
    for radial in (0.22, 0.45, 0.68, 0.90):
        theta = np.linspace(0.0, 2.0 * np.pi, 97)
        z = center[2] - np.sqrt(radius * radius - radial * radial)
        points = np.column_stack(
            (
                center[0] + radial * np.cos(theta),
                center[1] + radial * np.sin(theta),
                np.full(theta.shape, z),
            )
        )
        paths.append(_project(points, center_xy))
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        radial = np.linspace(0.0, 0.9, 28)
        z = center[2] - np.sqrt(radius * radius - radial * radial)
        points = np.column_stack(
            (
                center[0] + radial * np.cos(angle),
                center[1] + radial * np.sin(angle),
                z,
            )
        )
        paths.append(_project(points, center_xy))
    return paths


def _chart_geometry(evidence):
    deltas = np.asarray(evidence["deltas"], dtype=float)
    force_fe = np.asarray(evidence["force_fe"], dtype=float)
    curve_delta = np.geomspace(0.018, 0.088, 96)
    curve_force = hertz_force(curve_delta)
    x_min, x_max = 0.017, 0.092
    y_min = 0.045
    y_max = 0.58
    x0, y0, width, height = 790.0, 170.0, 410.0, 280.0

    def map_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        px = x0 + width * (np.log(x) - np.log(x_min)) / (np.log(x_max) - np.log(x_min))
        py = y0 + height - height * (np.log(y) - np.log(y_min)) / (
            np.log(y_max) - np.log(y_min)
        )
        return np.column_stack((px, py))

    return {
        "origin": (x0, y0),
        "width": width,
        "height": height,
        "map": map_xy,
        "fe": map_xy(deltas, force_fe),
        "hertz": map_xy(curve_delta, curve_force),
    }


def render_hertz_svg(evidence, output_path=DEFAULT_OUTPUT):
    """Write a solver-backed SVG and return its path."""

    config = evidence["configuration"]
    snapshot = evidence["snapshot"]
    nx, ny, nz = (int(value) for value in config["mesh_shape"])
    deformed = np.asarray(snapshot["nodes_deformed"], dtype=float)
    displacement = np.asarray(snapshot["displacement"], dtype=float)
    top_nodes = np.asarray(snapshot["top_nodes"], dtype=int)
    active = np.asarray(snapshot["active_contact"], dtype=bool)
    reaction = np.asarray(snapshot["contact_vertical_reaction"], dtype=float)
    downward = np.maximum(-displacement[:, 2], 0.0)
    max_downward = float(np.max(downward))
    block_size = np.asarray(config["block_size"], dtype=float)
    center_xy = (float(block_size[0] / 2.0), float(block_size[1] / 2.0))
    projected = _project(deformed, center_xy)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="620" '
        'viewBox="0 0 1280 620" role="img" aria-labelledby="title description">',
        '<title id="title">Rigid-sphere Hertz contact on a finite Hex8 block</title>',
        '<desc id="description">True-scale deformed finite-element mesh colored by '
        'downward displacement, active nodal reactions, and a log-log comparison of '
        'CoupFE force with the analytic Hertz law.</desc>',
        '<rect width="1280" height="620" fill="#FFFFFF"/>',
        f'<text x="48" y="48" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="28" font-weight="650">Rigid-sphere Hertz contact on a finite Hex8 block</text>',
        f'<text x="48" y="76" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="14">Corrected E–ν material mapping · frictionless nodal penalty contact · '
        'finite-block approximation to an elastic half-space</text>',
        f'<line x1="48" y1="96" x2="1232" y2="96" stroke="{_GRID}"/>',
        f'<text x="56" y="128" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="17" font-weight="600">Finite-element field at δ = {snapshot["delta"]:.3f}</text>',
        f'<text x="56" y="150" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="12">True-scale deformation · exterior faces colored by actual downward displacement −u<tspan baseline-shift="sub" font-size="9">z</tspan></text>',
    ]

    for face in _visible_faces(nx, ny, nz):
        face_nodes = np.asarray(face, dtype=int)
        value = float(np.mean(downward[face_nodes]))
        svg.append(
            f'<polygon points="{_points(projected[face_nodes])}" '
            f'fill="{_field_color(value, max_downward)}" stroke="#83939D" '
            'stroke-width="0.45" stroke-linejoin="round"/>'
        )

    sphere_center = np.asarray(snapshot["sphere_center"], dtype=float)
    for path in _sphere_cap_paths(
        sphere_center, float(config["sphere_radius"]), center_xy
    ):
        svg.append(
            f'<polyline points="{_points(path)}" fill="none" stroke="{_GOLD_DARK}" '
            'stroke-width="0.9" stroke-opacity="0.58"/>'
        )

    active_positions = deformed[top_nodes][active]
    active_node_ids = top_nodes[active]
    active_reaction = reaction[active]
    active_projected = _project(active_positions, center_xy)
    maximum_reaction = float(np.max(active_reaction)) if active_reaction.size else 1.0
    order = _marker_order(active_projected, active_node_ids)
    for index in order:
        marker_radius = 2.4 + 5.0 * np.sqrt(active_reaction[index] / maximum_reaction)
        svg.append(
            f'<circle cx="{_num(active_projected[index, 0])}" '
            f'cy="{_num(active_projected[index, 1])}" r="{_num(marker_radius)}" '
            f'fill="{_GOLD}" fill-opacity="0.88" stroke="{_GOLD_DARK}" stroke-width="1"/>'
        )

    legend_x, legend_y = 62.0, 548.0
    swatch_width = 25.0
    for index, color in enumerate(_FIELD_COLORS):
        svg.append(
            f'<rect x="{_num(legend_x + index * swatch_width)}" y="{legend_y}" '
            f'width="{swatch_width}" height="10" fill="{color}" stroke="#FFFFFF" stroke-width="0.5"/>'
        )
    svg.extend(
        [
            f'<text x="{legend_x}" y="538" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="11">−u<tspan baseline-shift="sub" font-size="8">z</tspan>: 0 to {max_downward:.4f}</text>',
            f'<circle cx="282" cy="553" r="5" fill="{_GOLD}" stroke="{_GOLD_DARK}"/>',
            f'<text x="294" y="557" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="11">active nodal reaction (marker size follows magnitude)</text>',
            f'<text x="760" y="128" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="17" font-weight="600">Normal force–indentation relation</text>',
            f'<text x="760" y="150" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="12">Logarithmic axes · five retained load states</text>',
        ]
    )

    chart = _chart_geometry(evidence)
    x0, y0 = chart["origin"]
    width, height = chart["width"], chart["height"]
    mapper = chart["map"]
    for tick in (0.02, 0.04, 0.08):
        x = float(mapper([tick], [0.05])[0, 0])
        svg.extend(
            [
                f'<line x1="{_num(x)}" y1="{y0}" x2="{_num(x)}" y2="{_num(y0 + height)}" '
                f'stroke="{_GRID}" stroke-width="0.7"/>',
                f'<text x="{_num(x)}" y="468" text-anchor="middle" fill="{_MUTED}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
                f'{tick:.2f}</text>',
            ]
        )
    for tick in (0.05, 0.1, 0.2, 0.5):
        y = float(mapper([0.02], [tick])[0, 1])
        svg.extend(
            [
                f'<line x1="{x0}" y1="{_num(y)}" x2="{_num(x0 + width)}" y2="{_num(y)}" '
                f'stroke="{_GRID}" stroke-width="0.7"/>',
                f'<text x="778" y="{_num(y + 4)}" text-anchor="end" fill="{_MUTED}" '
                'font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">'
                f'{tick:g}</text>',
            ]
        )
    svg.extend(
        [
            f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" '
            f'stroke="{_INK}" stroke-width="1.1"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" '
            f'stroke="{_INK}" stroke-width="1.1"/>',
            f'<text x="{x0 + width / 2}" y="492" text-anchor="middle" fill="{_INK}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12">prescribed approach δ</text>',
            f'<text x="742" y="{y0 + height / 2}" text-anchor="middle" fill="{_INK}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="12" '
            f'transform="rotate(-90 742 {y0 + height / 2})">normal force</text>',
            f'<polyline points="{_points(chart["hertz"])}" fill="none" stroke="{_INK}" '
            'stroke-width="2" stroke-dasharray="6 5"/>',
            f'<polyline points="{_points(chart["fe"])}" fill="none" stroke="{_BLUE}" '
            'stroke-width="2.4"/>',
        ]
    )
    for x, y in chart["fe"]:
        svg.append(
            f'<circle cx="{_num(x)}" cy="{_num(y)}" r="4.8" fill="#FFFFFF" '
            f'stroke="{_BLUE_DARK}" stroke-width="2.2"/>'
        )

    ratios = np.asarray(evidence["force_ratios"], dtype=float)
    errors = 100.0 * (ratios - 1.0)
    max_residual = max(case["free_residual_norm"] for case in evidence["cases"])
    svg.extend(
        [
            f'<text x="1130" y="202" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="12">Hertz · m = 1.500</text>',
            f'<text x="1060" y="278" fill="{_BLUE_DARK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="12" font-weight="600">CoupFE · m = {evidence["fit_slope"]:.3f}</text>',
            '<rect x="790" y="510" width="410" height="52" rx="3" fill="#F5F8F9" '
            f'stroke="{_GRID}"/>',
            f'<text x="806" y="531" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="12" font-weight="600">Force error {errors.min():+.1f}% to {errors.max():+.1f}%</text>',
            f'<text x="806" y="550" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="11">{config["nodes"]} nodes · {config["elements"]} Hex8 · '
            f'{int(np.count_nonzero(active))} active nodes · max free residual {max_residual:.1e}</text>',
            f'<line x1="48" y1="580" x2="1232" y2="580" stroke="{_GRID}"/>',
            f'<text x="48" y="602" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="11">Checked solver output · finite block and nodal contact are not a mesh/domain-converged pressure solution · no smoothing</text>',
            '</svg>',
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def main():
    evidence = run_hertz()
    output = render_hertz_svg(evidence)
    print(f"wrote {escape(str(output.relative_to(_ROOT)))}")
    return output


if __name__ == "__main__":
    main()
