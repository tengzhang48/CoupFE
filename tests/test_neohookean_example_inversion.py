"""Exercise shipped neo-Hookean sources, including coupled example snapshots."""
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from test_neohookean_inversion import ROOT, Q4, HEX8, TET4, load_example, build, outputs

Q8 = np.vstack([Q4, [[.5, 0], [1, .5], [.5, 1], [0, .5]]])
# source relative to ROOT, material example, weak form, geometry, nodal widths.
SNAPSHOTS = [
    ('examples/morphing_hex8/pressuregel_local_pressure_hex8.for',
     'morphing_hex8', 'PressureGelLocalPressureHex8', HEX8, [4]*8),
    ('examples/compression_cylinders/mr_fbar_q4.for',
     'compression_cylinders/mooney_rivlin.py', 'MRQuad4', Q4, [2]*4),
    ('coupfe/runtime/elements/neo_hookean_q4_native.for', None, None, Q4, [2]*4),
    ('coupfe/runtime/elements/neo_hookean_q4.for', None, None, Q4, [2]*4),
    ('examples/neo_hookean_local_pressure_quad4/neohookean_up_q4_uel.for',
     'neo_hookean_local_pressure_quad4', 'NeoHookeanUPQuad4', Q4, [2]*4),
    ('examples/neo_hookean_local_pressure_hex8/neohookean_up_hex8_uel.for',
     'neo_hookean_local_pressure_hex8', 'NeoHookeanUPHex8', HEX8, [3]*8),
    ('examples/phasefield_fracture_uel/phasefield_fracture_uel.for',
     'phasefield_fracture_uel', 'PhaseFieldFractureQuad4', Q4, [3]*4),
    ('examples/thermo_mechanics_quad8/thermo_mechanics_quad8_uel.for',
     'thermo_mechanics_quad8', 'ThermoMechanicalQuad8', Q8, [3]*4+[2]*4),
    ('examples/gel_chester_anand/u_p_mu_quad8/chester_anand_upmu_quad8_uel.for',
     'gel_chester_anand/u_p_mu_quad8', 'ChesterAnandUPMuQuad8', Q8, [4]*4+[3]*4),
    ('coupfe/runtime/elements/neo_hookean_hex8_fbar.for', None, None, HEX8, [3]*8),
    ('coupfe/runtime/elements/neo_hookean_q4_fbar_native.for', None, None, Q4, [2]*4),
    ('examples/stabilized_tet4/scovazzi_block_tet4.for',
     'stabilized_tet4', 'ScovazziBlockTet4', TET4, [4]*4),
]


def snapshot_data(entry):
    relative, example, cls, nodes, widths = entry
    if example:
        ex = load_example(example)
        problem = getattr(ex, cls)()
        props = tuple(problem._mat.props.values())
    else:
        props = (0.5, 50.0)
    ns = 1
    return relative, nodes, widths, props, ns


def nodal_state(nodes, widths, amount):
    u = np.zeros(sum(widths))
    offset = 0
    for point, width in zip(nodes, widths):
        u[offset:offset+nodes.shape[1]] = amount * point
        offset += width
    return u


@pytest.mark.parametrize('entry', SNAPSHOTS, ids=[Path(s[0]).stem for s in SNAPSHOTS])
def test_shipped_neo_hookean_inversion(entry, tmp_path):
    if any(shutil.which(t) is None for t in ('gfortran', 'meson', 'ninja')):
        pytest.skip('gfortran/f2py toolchain unavailable')
    relative, nodes, widths, props, ns = snapshot_data(entry)
    path = tmp_path / 'kernel.for'
    path.write_text((ROOT / relative).read_text())
    module, element = build(path, 'snapshot_' + Path(relative).stem,
                            tmp_path / 'new', props, max(widths), ns, nodes)
    control = tmp_path / 'control.for'
    source = re.sub(r'^C     BEGIN positive-J guard\n.*?^C     END positive-J guard\n',
                    '', path.read_text(), flags=re.M | re.S)
    control.write_text(source)
    old_module, old = (build(control, 'control_' + Path(relative).stem,
                             tmp_path / 'old', props, max(widths), ns, nodes)
                       if source != path.read_text() else (module, element))
    for amount in (-.03, .04, .15):
        u = nodal_state(nodes, widths, amount)
        for new_value, old_value in zip(outputs(module, element, nodes, u),
                                        outputs(old_module, old, nodes, u)):
            assert np.isfinite(new_value).all()
            assert new_value.tobytes() == old_value.tobytes()
    u = np.zeros(sum(widths))
    # One corner crosses the opposite face; diagonal push also catches F-bar.
    # Tet4: the apex (node 3) moves through the base, det(F) = -2.
    index = 2 if nodes.shape[1] == 2 else (3 if len(nodes) == 4 else 6)
    offset = sum(widths[:index])
    u[offset:offset+nodes.shape[1]] = -3.
    for value in outputs(module, element, nodes, u):
        assert np.isnan(value).all(), relative


@pytest.fixture(scope='module')
def hex8_fbar_cpu(tmp_path_factory):
    if any(shutil.which(t) is None for t in ('gfortran', 'meson', 'ninja')):
        pytest.skip('gfortran/f2py toolchain unavailable')
    work = tmp_path_factory.mktemp('translated_hex8')
    path = work / 'kernel.for'
    path.write_text((ROOT / 'coupfe/runtime/elements/neo_hookean_hex8_fbar.for').read_text())
    return build(path, 'translated_hex8', work / 'build', (1., 2.), 3, 1, HEX8)


@pytest.mark.parametrize('entry', ['rk', 'r'])
def test_translated_collapsed_hex8_fbar_cpu_returns_nan(hex8_fbar_cpu, entry):
    """Lab's exact-zero Jbar state, through each actual public CPU entry."""
    module, element = hex8_fbar_cpu
    u = np.zeros(24)
    u[2::3] = [-2.9375] * 4 + [-3.9375] * 4
    np.testing.assert_array_equal(HEX8[:, 2] + u[2::3], -2.9375)
    values = outputs(module, element, HEX8, u)
    mechanical = values[:2] if entry == 'rk' else values[2:]
    assert len(mechanical) == (2 if entry == 'rk' else 1)
    for value in mechanical:
        assert np.isnan(value).all(), f'{entry}: translated collapse must return all NaNs'
