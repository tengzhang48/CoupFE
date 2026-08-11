"""Render the retained finite-strain block solve as a deterministic SVG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.neo_hookean_block.run import solve_block


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _ROOT / "docs" / "assets" / "neo-hookean-block-snapshot.svg"

_INK = "#17252C"
_MUTED = "#607078"
_GRID = "#D9E1E4"
_REFERENCE = "#9AA8AE"
_TEAL = "#176B74"
_TEAL_DARK = "#0E4C55"
_COPPER = "#A86432"
_FIELD = ("#EDF3F3", "#D5E7E8", "#B5D6D8", "#88BEC2", "#55A0A7", "#287E87", "#176B74")


def _num(value):
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _points(values):
    return " ".join(f"{_num(x)},{_num(y)}" for x, y in values)


def _field_color(value, maximum):
    fraction = 0.0 if maximum <= 0.0 else float(np.clip(value / maximum, 0.0, 1.0))
    index = min(int(fraction * len(_FIELD)), len(_FIELD) - 1)
    return _FIELD[index]


def render_neo_hookean_block_svg(evidence, output_path=DEFAULT_OUTPUT):
    """Write a true-scale deformed Quad4 snapshot from solved nodal arrays."""

    config = evidence["configuration"]
    reference = np.asarray(evidence["nodes_reference"], dtype=float)
    deformed = np.asarray(evidence["nodes_deformed"], dtype=float)
    displacement = np.asarray(evidence["displacement"], dtype=float)
    elements = np.asarray(evidence["elements"], dtype=int)
    downward = np.maximum(-displacement[:, 1], 0.0)
    max_downward = float(np.max(downward))

    origin_x, baseline_y, scale = 72.0, 401.0, 290.0

    def project(values):
        values = np.asarray(values, dtype=float)
        return np.column_stack(
            (origin_x + scale * values[:, 0], baseline_y - scale * values[:, 1])
        )

    projected = project(deformed)
    lx, ly = config["block_size"]
    reference_outline = project(
        np.array([[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly], [0.0, 0.0]])
    )
    lam = config["axial_stretch"]
    lam_t = evidence["lateral_stretch_analytic"]
    analytic_outline = project(
        np.array(
            [
                [0.0, 0.0],
                [lam * lx, 0.0],
                [lam * lx, lam_t * ly],
                [0.0, lam_t * ly],
                [0.0, 0.0],
            ]
        )
    )

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480" '
        'viewBox="0 0 720 480" role="img" aria-labelledby="title description">',
        '<title id="title">Finite-strain uniaxial block solution</title>',
        '<desc id="description">The actual true-scale deformed six-by-six Quad4 mesh, '
        'colored by solved lateral displacement and compared with its traction-free analytic outline.</desc>',
        '<rect width="720" height="480" fill="#FFFFFF"/>',
        f'<text x="42" y="45" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="24" font-weight="650">Finite-strain block · lateral contraction</text>',
        f'<text x="42" y="72" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="15">{config["mesh_shape"][0]} × {config["mesh_shape"][1]} Quad4 · '
        f'axial stretch λₓ = {lam:.2f} · true deformation scale</text>',
        f'<line x1="42" y1="91" x2="678" y2="91" stroke="{_GRID}"/>',
        f'<text x="57" y="121" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        'font-size="17" font-weight="600">Solved mesh · color is actual uᵧ</text>',
        f'<polyline points="{_points(reference_outline)}" fill="none" stroke="{_REFERENCE}" '
        'stroke-width="2" stroke-dasharray="7 6"/>',
    ]

    for element in elements:
        value = float(np.mean(downward[element]))
        svg.append(
            f'<polygon points="{_points(projected[element])}" '
            f'fill="{_field_color(value, max_downward)}" stroke="#6E858B" '
            'stroke-width="0.8" stroke-linejoin="round"/>'
        )

    svg.extend(
        [
            f'<polyline points="{_points(analytic_outline)}" fill="none" stroke="{_COPPER}" '
            'stroke-width="2.5" stroke-dasharray="8 6"/>',
            f'<line x1="{origin_x}" y1="{baseline_y + 18}" x2="{origin_x + scale * lam * lx}" '
            f'y2="{baseline_y + 18}" stroke="{_TEAL_DARK}" stroke-width="1.8"/>',
            f'<path d="M {origin_x + scale * lam * lx} {baseline_y + 18} l -10 -6 v 12 z" '
            f'fill="{_TEAL_DARK}"/>',
            f'<text x="{origin_x + 0.5 * scale * lam * lx}" y="{baseline_y + 42}" '
            f'text-anchor="middle" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">20% prescribed extension</text>',
            f'<text x="472" y="132" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">LATERAL STRETCH</text>',
            f'<text x="472" y="171" fill="{_INK}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="28" font-weight="700">{evidence["lateral_stretch_fe"]:.6f}</text>',
            f'<text x="472" y="195" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="15">finite-element mean</text>',
            f'<text x="472" y="235" fill="{_COPPER}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            f'font-size="23" font-weight="650">{evidence["lateral_stretch_analytic"]:.6f}</text>',
            f'<text x="472" y="258" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="15">traction-free oracle</text>',
            f'<line x1="472" y1="282" x2="661" y2="282" stroke="{_GRID}"/>',
            f'<line x1="472" y1="307" x2="503" y2="307" stroke="{_REFERENCE}" '
            'stroke-width="2" stroke-dasharray="7 6"/>',
            f'<text x="513" y="313" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">reference block</text>',
            f'<line x1="472" y1="343" x2="503" y2="343" stroke="{_COPPER}" '
            'stroke-width="2.5" stroke-dasharray="8 6"/>',
            f'<text x="513" y="349" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">analytic outline</text>',
            f'<text x="472" y="384" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">FE outline coincides with</text>',
            f'<text x="472" y="404" fill="{_MUTED}" font-family="Inter,Segoe UI,Arial,sans-serif" '
            'font-size="14">the analytic outline.</text>',
        ]
    )

    legend_x, legend_y, swatch_width = 472.0, 431.0, 25.0
    for index, color in enumerate(_FIELD):
        svg.append(
            f'<rect x="{_num(legend_x + index * swatch_width)}" y="{legend_y}" '
            f'width="{swatch_width}" height="10" fill="{color}" stroke="#FFFFFF" stroke-width="0.5"/>'
        )
    svg.extend(
        [
            f'<text x="{legend_x}" y="464" fill="{_MUTED}" '
            'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13">uᵧ: '
            f'{-max_downward:.4f} to 0 · no stress reconstruction</text>',
            '</svg>',
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def main():
    output = render_neo_hookean_block_svg(solve_block())
    print(f"wrote {output.relative_to(_ROOT)}")
    return output


if __name__ == "__main__":
    main()
