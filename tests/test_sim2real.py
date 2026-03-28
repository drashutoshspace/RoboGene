import json

import numpy as np
import pandas as pd
import pytest
import yaml

from robogene import sim2real
from robogene.config import DEFAULT_CONFIG, save_config
from robogene.phenotype import ActuatorInfo, Phenotype

MAPPING = {
    "hardware": {"wheel_module": "moonbot_wheel_module", "limb": "moonbot_limb"},
    "wheels": [
        {"hardware": "wheel module 1 / wheel 1", "node": 2, "sign": 1.0},
        {"hardware": "wheel module 1 / wheel 2", "node": 4, "sign": -1.0},
        {"hardware": "wheel module 2 / wheel 1", "node": 1, "sign": 1.0},
        {"hardware": "wheel module 2 / wheel 2", "node": 3, "sign": 1.0},
    ],
    "limb": {"center_arm_node": 5, "joints": 7},
    "rollout": {"steps": 10, "goal": [10.0, 0.0]},
}
V_MAX = 0.654


def _act(node, category, lo, hi):
    mode = "velocity" if category == "wheel" else "position"
    return ActuatorInfo(f"a{node}", f"j{node}", node, "m", category, mode, (lo, hi), 5.0)


@pytest.fixture
def ph() -> Phenotype:
    nodes = [{"id": i, "module": "m", "params": {}} for i in range(6)]
    acts = [_act(1, "wheel", -5, 5), _act(2, "wheel", -5, 5), _act(5, "arm", -1.0, 0.5),
            _act(3, "wheel", -5, 5), _act(4, "wheel", -5, 5)]
    return Phenotype({"root": 0, "nodes": nodes, "edges": []}, "3Arm+4Wheel", "", "", acts, None, 1.0, 0.1)


def _actions(T=8):
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, (T, 5))


def test_wheel_scaling_and_sign(ph):
    a = _actions()
    df, scaling = sim2real.hardware_commands(a, ph, MAPPING, control_dt=0.02)
    # actuator order is nodes 1, 2, 5, 3, 4
    np.testing.assert_allclose(df["wheel_1"], a[:, 1] * V_MAX)
    np.testing.assert_allclose(df["wheel_2"], -a[:, 4] * V_MAX)
    np.testing.assert_allclose(df["wheel_3"], a[:, 0] * V_MAX)
    np.testing.assert_allclose(df["wheel_4"], a[:, 3] * V_MAX)
    np.testing.assert_allclose(df["t"], np.arange(8) * 0.02)
    assert [w["module"] for w in scaling["wheels"]] == [1, 1, 2, 2]


def test_limb_constant(ph):
    a = _actions()
    df, scaling = sim2real.hardware_commands(a, ph, MAPPING, control_dt=0.02)
    expected = np.mean(-1.0 + (a[:, 2] + 1.0) * 0.5 * 1.5)
    for j in range(1, 8):
        np.testing.assert_allclose(df[f"limb_{j}"], expected)
    assert scaling["limb"]["a_c_rad"] == pytest.approx(expected)
    assert list(df.columns) == ["t"] + [f"wheel_{k}" for k in range(1, 5)] + [f"limb_{j}" for j in range(1, 8)]


def test_limb_without_joint_is_zero(ph):
    m = dict(MAPPING, limb={"center_arm_node": 0, "joints": 7})
    df, _ = sim2real.hardware_commands(_actions(), ph, m, control_dt=0.02)
    assert (df["limb_1"] == 0.0).all()


@pytest.mark.parametrize("wheel, message", [
    ({"node": 5, "sign": 1.0}, "not an actuated wheel"),
    ({"node": 9, "sign": 1.0}, "not in the genotype"),
    ({"node": 2, "sign": 2.0}, "sign"),
])
def test_validation(ph, wheel, message):
    m = dict(MAPPING, wheels=[wheel] + MAPPING["wheels"][1:])
    with pytest.raises(ValueError, match=message):
        sim2real.resolve_wheels(m, ph)


def test_run_sim2real_writes_files(tmp_path, ph):
    run = tmp_path / "run"
    folder = run / "individuals" / "g001_i02"
    folder.mkdir(parents=True)
    save_config(DEFAULT_CONFIG, run / "config.yaml")
    entry = folder / "robogene.json"
    entry.write_text(json.dumps({"id": "g001_i02", "genotype": ph.genotype, "policy": {"checkpoint": "policy.zip"}}))
    mapping = tmp_path / "map.yaml"
    mapping.write_text(yaml.safe_dump(MAPPING))
    calls = {}

    def fake_rollout(model, phen, cfg, steps, goal):
        calls.update(steps=steps, goal=goal)
        return np.full((steps, 5), 0.5)

    out = tmp_path / "commands.csv"
    sim2real.run_sim2real(entry, mapping, out, rollout_fn=fake_rollout, phenotype=ph)
    assert calls == {"steps": 10, "goal": [10.0, 0.0]}
    df = pd.read_csv(out)
    assert len(df) == 10
    np.testing.assert_allclose(df["wheel_2"], -0.5 * V_MAX)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["sources"]["entry_id"] == "g001_i02"
    assert meta["control_hz"] == pytest.approx(50.0)
    assert "open loop" in meta["note"]
