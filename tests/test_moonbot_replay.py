import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "moonbot"))

from replay_commands import load_channels, load_commands, messages, replay, stop_messages  # noqa: E402


def write_commands(path: Path) -> None:
    lines = ["t,wheel_1,wheel_2,limb_1", "0.0,0.1,-0.1,0.5", "0.02,0.2,-0.2,0.5"]
    path.write_text("\n".join(lines) + "\n")


def write_mapping(path: Path, joint: str = "wheel11_left_joint") -> None:
    channels = [
        {"column": "wheel_1", "leg": 11, "joint": joint, "field": "velocity"},
        {"column": "wheel_2", "leg": 11, "joint": "wheel11_right_joint", "field": "velocity"},
        {"column": "limb_1", "leg": 4, "joint": "limb_joint_1", "field": "position"},
    ]
    path.write_text(yaml.safe_dump({"channels": channels}))


def test_messages_group_joints_per_leg(tmp_path):
    write_commands(tmp_path / "c.csv")
    write_mapping(tmp_path / "m.yaml")
    t, cols = load_commands(tmp_path / "c.csv")
    steps = list(messages(t, cols, load_channels(tmp_path / "m.yaml")))
    assert len(steps) == 2
    _, per_leg = steps[1]
    assert per_leg[11] == (["wheel11_left_joint", "wheel11_right_joint"], [], [0.2, -0.2])
    assert per_leg[4] == (["limb_joint_1"], [0.5], [])
    stop = stop_messages(load_channels(tmp_path / "m.yaml"))
    assert stop == {11: (["wheel11_left_joint", "wheel11_right_joint"], [], [0.0, 0.0])}


def test_placeholder_joint_names_are_rejected(tmp_path):
    write_mapping(tmp_path / "m.yaml", joint="SET_wheel")
    with pytest.raises(ValueError, match="joint name"):
        load_channels(tmp_path / "m.yaml")


def test_dry_run_prints_messages(tmp_path, capsys):
    write_commands(tmp_path / "c.csv")
    write_mapping(tmp_path / "m.yaml")
    replay(str(tmp_path / "c.csv"), str(tmp_path / "m.yaml"), dry_run=True)
    out = capsys.readouterr().out
    assert "/leg11/joint_set" in out and "2 steps" in out


def test_default_vehicle_mapping_is_well_formed():
    cfg = yaml.safe_load((ROOT / "moonbot/replay_vehicle.yaml").read_text())
    columns = [c["column"] for c in cfg["channels"]]
    assert columns == [f"wheel_{i}" for i in range(1, 5)] + [f"limb_{j}" for j in range(1, 8)]


def test_build_phenotype_example(tmp_path):
    r = subprocess.run(
        [sys.executable, "examples/build_phenotype.py", "--out", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "class=3Arm+4Wheel" in r.stdout and "checks=True" in r.stdout
    assert (tmp_path / "model.xml").exists() and (tmp_path / "genotype.json").exists()
