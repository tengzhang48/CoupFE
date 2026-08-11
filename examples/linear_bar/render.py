"""Render the retained nonlinear-bar solve as a deterministic technical SVG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.linear_bar.run import solve_bar


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _ROOT / "docs" / "assets" / "linear-bar-snapshot.svg"

_INK = "#17252C"
_MUTED = "#607078"
_GRID = "#D9E1E4"
_REFERENCE = "#9AA8AE"
_TEAL = "#176B74"
_TEAL_DARK = "#0E4C55"
_COPPER = "#A86432"
_FIELD = ("#E7EFF0", "#C9DFE1", "#9EC8CB", "#6AACB2", "#3A8B94", "#176B74")


def _num(value):
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _points(values):
    return " ".join(f"{_num(x)},{_num(y)}" for x, y in values)


def _field_color(value, maximum):
    fraction = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
    index = min(int(fraction * len(_FIELD)), len(_FIELD) - 1)
    return _FIELD[index]


def render_linear_bar_svg(evidence, output_path=DEFAULT_OUTPUT):
    """Write a fixed-size SVG from one already-computed bar solution."""

    config = evidence["configuration"]
    reference = np.asarray(evidence["nodes_reference"], dtype=float)
    deformed = np.asarray(evidence["nodes_deformed"], dtype=float)
    displacement = np.asarray(evidence["displacement"], dtype=float)
    elements = np.asarray(evidence["elements"], dtype=int)

    x0, geometry_width = 76.0, 530.0
    geometry_limit = max(float(deformed[-1]) + 0.18, float(reference[-1]) * 1.08)

    def map_geometry(values):
        return x0 + geometry_width * np.asarray(values, dtype=float) / geometry_limit

    geometry_x = map_geometry(deformed)
    reference_tip_x = float(map_geometry([reference[-1]])[0])
    y_bar = 179.0

    plot_x, plot_y, plot_w, plot_h = 82.0, 304.0, 544.0, 104.0
    y_max = 1.10 * max(float(evidence["linear_tip_reference"]), float(displacement[-1]))

    def map_plot(x_values, u_values):
        px = plot_x + plot_w * np.asarray(x_values, dtype=float) / float(reference[-1])
        py = plot_y + plot_h - plot_h * np.asarray(u_values, dtype=float) / y_max
        return np.column_stack((px, py))

    solved_curve = map_plot(reference, displacement)
    linear_displacement = evidence["linear_tip_reference"] * reference / float(reference[-1])
    linear_curve = map_plot(reference, linear_displacement)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" '
        'viewBox="0 0 720 480" role="img" aria-labelledby="title description">',
        '<title id="title">Nonlinear axial bar displacement</title>',
        '<desc id="description">The actual solved ten-element bar at true axial scale, '
        'with nodal displacement plotted against the reference coordinate.</desc>',
        '<rect width="720" height="480" fill="#FFFFFF"/>',
        f'<text x="42" y="45" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="24" font-weight="650">Nonlinear bar · solved displacement</text>',
        f'<text x="42" y="72" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="15">{config["n_elements"]} elements · fixed left end · tip force {config["tip_force"]:g}</text>',
        f'<line x1="42" y1="91" x2="678" y2="91" stroke="{_GRID}"/>',
        f'<text x="42" y="121" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="17" font-weight="600">True-scale axial geometry</text>',
        f'<line x1="{_num(map_geometry([reference[0]])[0])}" y1="{y_bar}" '
        f'x2="{_num(reference_tip_x)}" y2="{y_bar}" stroke="{_REFERENCE}" '
        'stroke-width="3" stroke-dasharray="7 6"/>',
        f'<line x1="{_num(reference_tip_x)}" y1="145" x2="{_num(reference_tip_x)}" y2="210" '
        f'stroke="{_REFERENCE}" stroke-width="1.2" stroke-dasharray="4 4"/>',
        f'<text x="{_num(reference_tip_x)}" y="137" text-anchor="middle" fill="{_MUTED}" '
        'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13">reference tip</text>',
    ]

    for n0, n1 in elements:
        mean_u = 0.5 * (displacement[n0] + displacement[n1])
        svg.append(
            f'<line x1="{_num(geometry_x[n0])}" y1="{y_bar}" '
            f'x2="{_num(geometry_x[n1])}" y2="{y_bar}" '
            f'stroke="{_field_color(mean_u, displacement[-1])}" stroke-width="10" '
            'stroke-linecap="round"/>'
        )
    for index, x in enumerate(geometry_x):
        svg.append(
            f'<circle cx="{_num(x)}" cy="{y_bar}" r="4.2" fill="#FFFFFF" '
            f'stroke="{_field_color(displacement[index], displacement[-1])}" stroke-width="2"/>'
        )

    clamp_x = float(geometry_x[0])
    svg.extend(
        [
            f'<line x1="{_num(clamp_x)}" y1="151" x2="{_num(clamp_x)}" y2="207" '
            f'stroke="{_INK}" stroke-width="3"/>',
            *[
                f'<line x1="{_num(clamp_x - 12)}" y1="{y}" x2="{_num(clamp_x)}" y2="{y - 8}" '
                f'stroke="{_INK}" stroke-width="1.4"/>'
                for y in range(158, 215, 11)
            ],
            f'<line x1="{_num(geometry_x[-1] + 9)}" y1="{y_bar}" '
            f'x2="{_num(geometry_x[-1] + 51)}" y2="{y_bar}" stroke="{_COPPER}" stroke-width="2.5"/>',
            f'<path d="M {_num(geometry_x[-1] + 51)} {y_bar} l -12 -7 v 14 z" fill="{_COPPER}"/>',
            f'<text x="{_num(geometry_x[-1] + 30)}" y="164" text-anchor="middle" fill="{_COPPER}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14" font-weight="650">F</text>',
            f'<text x="42" y="255" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="17" font-weight="600">Nodal displacement u(X)</text>',
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" '
            f'stroke="{_INK}" stroke-width="1.2"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" '
            f'stroke="{_INK}" stroke-width="1.2"/>',
            f'<polyline points="{_points(linear_curve)}" fill="none" stroke="{_REFERENCE}" '
            'stroke-width="2" stroke-dasharray="7 6"/>',
            f'<polyline points="{_points(solved_curve)}" fill="none" stroke="{_TEAL}" stroke-width="3"/>',
        ]
    )
    for x, y in solved_curve:
        svg.append(
            f'<circle cx="{_num(x)}" cy="{_num(y)}" r="3.4" fill="#FFFFFF" '
            f'stroke="{_TEAL_DARK}" stroke-width="1.8"/>'
        )
    svg.extend(
        [
            f'<text x="{plot_x}" y="431" fill="{_MUTED}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13">X = 0</text>',
            f'<text x="{plot_x + plot_w}" y="431" text-anchor="end" fill="{_MUTED}" '
            f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13">X = {reference[-1]:g}</text>',
            f'<text x="{plot_x + 18}" y="288" fill="{_REFERENCE}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13">-- linear H = 0</text>',
            f'<text x="{plot_x + 168}" y="288" fill="{_TEAL_DARK}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13">— nonlinear solve</text>',
            f'<text x="42" y="462" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="14">u_tip = {evidence["tip_displacement"]:.6f} · '
            f'{evidence["newton_iterations"]} Newton iterations · nodal displacement only</text>',
            '</svg>',
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def main():
    output = render_linear_bar_svg(solve_bar())
    print(f"wrote {output.relative_to(_ROOT)}")
    return output


if __name__ == "__main__":
    main()
