import mujoco
import numpy as np
import pytest

from robogene.config import load_config
from robogene.genotype import Genotype, Node, forward_kinematics, validate_genotype
from robogene.phenotype import build_model, check_phenotype, expand, model_xml
from robogene.schema import default_library

LIB = default_library()


def _settle(m: mujoco.MjModel, ph, seconds: float = 1.0) -> mujoco.MjData:
    d = mujoco.MjData(m)
    d.ctrl[:] = [0.5 * sum(a.ctrl_range) if a.control_mode == "position" else 0.0 for a in ph.actuators]
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def wheeled_arm_robot(arm_links: int = 3) -> Genotype:
    g = Genotype([Node(0, "body_small")], [])
    for port, wheel in [("+y_front", "wheel_left"), ("+y_rear", "wheel_left"),
                        ("-y_front", "wheel_right"), ("-y_rear", "wheel_right")]:
        g.add_node(wheel, 0, port, "hub")
    parent, port = 0, "+z"
    for k in range(arm_links):
        parent = g.add_node("arm_link", parent, port, "end_a", roll_deg=90.0 * (k % 2))
        port = "end_b"
    return g


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(overrides={"checks": {"stability": True, "reachability": True}})


@pytest.fixture(scope="module")
def robot():
    return expand(wheeled_arm_robot())


def test_wheeled_arm_robot_compiles(robot, cfg):
    m = build_model(robot, cfg)
    assert robot.morphology_class == "3Arm+4Wheel"
    assert m.nu == 7
    assert [m.actuator(i).name for i in range(m.nu)] == [f"a{i}" for i in range(1, 8)]
    assert [a.name for a in robot.actuators] == [f"a{i}" for i in range(1, 8)]
    assert [a.control_mode for a in robot.actuators] == ["velocity"] * 4 + ["position"] * 3
    assert m.joint("root").type == mujoco.mjtJoint.mjJNT_FREE
    for name in ("base", "ee"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    assert robot.end_effector == 7
    assert robot.spawn_clearance == pytest.approx(0.1)
    assert robot.total_mass == pytest.approx(sum(LIB[n["module"]].mass() for n in robot.genotype["nodes"]))


def test_actuator_ranges(robot, cfg):
    m = build_model(robot, cfg)
    wheel = LIB["wheel_left"].actuation
    arm = LIB["arm_link"].actuation
    np.testing.assert_allclose(m.actuator("a1").ctrlrange, [-wheel["velocity_limit"], wheel["velocity_limit"]])
    np.testing.assert_allclose(m.actuator("a1").forcerange, [-wheel["torque_limit"], wheel["torque_limit"]])
    np.testing.assert_allclose(m.actuator("a5").ctrlrange, arm["range"])
    np.testing.assert_allclose(m.jnt_range[m.joint("j5").id], arm["range"])
    assert not m.jnt_limited[m.joint("j1").id]


def test_limit_scale_shrinks_joint_range(cfg):
    g = wheeled_arm_robot()
    g.node(5).params["limit_scale"] = 0.5
    ph = expand(g)
    m = build_model(ph, cfg)
    lo, hi = LIB["arm_link"].actuation["range"]
    np.testing.assert_allclose(m.jnt_range[m.joint("j5").id], [0.5 * lo, 0.5 * hi])
    assert ph.actuators[4].ctrl_range == pytest.approx((0.5 * lo, 0.5 * hi))


def test_body_poses_match_forward_kinematics(robot, cfg):
    m = build_model(robot, cfg)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    poses = forward_kinematics(robot.genotype_obj(), LIB)
    offset = d.xpos[m.body("n0").id]
    for nid, (R, p) in poses.items():
        b = m.body(f"n{nid}").id
        np.testing.assert_allclose(d.xpos[b] - offset, p, atol=1e-9)
        np.testing.assert_allclose(d.xmat[b].reshape(3, 3), R, atol=1e-9)
    np.testing.assert_allclose(d.site("ee").xpos - offset, [0.0, 0.0, 0.05 + 3 * 0.3], atol=1e-9)


def test_hinge_joint_sits_at_attach_port(robot, cfg):
    m = build_model(robot, cfg)
    np.testing.assert_allclose(m.jnt_pos[m.joint("j5").id], LIB["arm_link"].ports["end_a"].position)
    np.testing.assert_allclose(m.jnt_pos[m.joint("j1").id], 0.0)


def test_wheels_rest_on_ground(robot, cfg):
    m = build_model(robot, cfg)
    d = _settle(m, robot)
    radius = LIB["wheel_left"].size[0]
    for nid in range(1, 5):
        assert d.xpos[m.body(f"n{nid}").id][2] == pytest.approx(radius, abs=2e-3)
    assert np.all(np.isfinite(d.qpos))


def test_wheeled_arm_robot_passes_checks(robot, cfg):
    ok, reasons, metrics = check_phenotype(robot, cfg)
    assert ok, reasons
    assert metrics["max_mass_dev"] < 1e-6 and metrics["max_inertia_dev"] < 1e-6
    assert metrics["tilt_deg"] < 1.0
    assert metrics["n_ground_contacts"] >= 4
    assert metrics["max_penetration"] < 1e-3
    assert metrics["reach"] == pytest.approx(0.9)


def test_reachability(cfg):
    no_arm = expand(wheeled_arm_robot(arm_links=0))
    ok, reasons, _ = check_phenotype(no_arm, cfg)
    assert not ok and any("end effector" in r for r in reasons)
    short = expand(wheeled_arm_robot(arm_links=1))
    strict = load_config(overrides={"checks": {"reachability": True, "reach_min": 0.5}})
    ok, reasons, metrics = check_phenotype(short, strict)
    assert not ok and metrics["reach"] == pytest.approx(0.3)
    loose = load_config(overrides={"checks": {"reachability": True, "reach_min": 0.3}})
    assert check_phenotype(short, loose)[0]


def test_end_effector_picks_longest_chain():
    g = wheeled_arm_robot(arm_links=0)
    first = g.add_node("arm_link", 0, "+x", "end_a")
    long_tip = g.add_node("arm_link", g.add_node("arm_link", 0, "-x", "end_a"), "end_b", "end_a")
    ph = expand(g)
    assert ph.end_effector == long_tip != first


def test_toppling_arm_chain_fails_stability(cfg):
    g = Genotype([Node(0, "arm_link")], [])
    g.add_node("arm_link", g.add_node("arm_link", 0, "end_b", "end_a"), "end_b", "end_a")
    ok, reasons, _ = check_phenotype(expand(g), cfg)
    assert not ok
    assert any("ground contacts" in r or "tilt" in r or "margin" in r for r in reasons)


def test_one_sided_wheels_fail_stability(cfg):
    g = Genotype([Node(0, "body_small")], [])
    g.add_node("wheel_left", 0, "+y_front", "hub")
    g.add_node("wheel_left", 0, "+y_rear", "hub")
    ok, reasons, metrics = check_phenotype(expand(g), cfg)
    assert not ok and any("tilt" in r for r in reasons)
    assert metrics["tilt_deg"] > cfg["checks"]["max_tilt_deg"]


def test_overlapping_wheels_fail_self_penetration(cfg):
    g = Genotype([Node(0, "body_small")], [])
    g.add_node("wheel_left", 0, "+y_front", "hub")
    g.add_node("wheel_left", 0, "+y", "hub")
    assert validate_genotype(g, LIB, cfg["checks"], cfg["ga"])[0]
    ok, reasons, metrics = check_phenotype(expand(g), cfg)
    assert not ok and any("self-penetration" in r for r in reasons)
    assert metrics["max_penetration"] > 0.01


def test_model_xml_options(robot, cfg):
    xml = model_xml(robot, cfg, spawn_xy=(1.0, -2.0))
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.opt.timestep == pytest.approx(cfg["simulation"]["timestep"])
    np.testing.assert_allclose(m.opt.gravity, [0, 0, -cfg["simulation"]["gravity"]])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.qpos[:3], [1.0, -2.0, robot.spawn_clearance + 0.02])


def _random_genotype(rng: np.random.Generator, cfg: dict) -> Genotype:
    modules = list(LIB.modules)
    while True:
        g = Genotype([Node(0, str(rng.choice(modules)))], [])
        for _ in range(int(rng.integers(1, 9))):
            open_nodes = [n for n in g.ids if g.free_ports(n, LIB)]
            if not open_nodes:
                break
            parent = int(rng.choice(open_nodes))
            parent_port = str(rng.choice(g.free_ports(parent, LIB)))
            child = str(rng.choice(modules))
            child_port = str(rng.choice(list(LIB[child].ports)))
            roll = float(rng.choice(LIB[g.node(parent).module].ports[parent_port].roll_options_deg))
            params = {"limit_scale": float(rng.uniform(0.3, 1.0))} if rng.random() < 0.3 else None
            g.add_node(child, parent, parent_port, child_port, roll, params)
        if validate_genotype(g, LIB, cfg["checks"], cfg["ga"])[0]:
            return g


def test_random_valid_genotypes_compile(cfg):
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = _random_genotype(rng, cfg)
        ph = expand(g)
        m = build_model(ph, cfg)
        order = [nid for nid in g.subtree(g.root)[1:] if LIB[g.node(nid).module].actuated]
        assert [a.node for a in ph.actuators] == order
        assert m.nu == len(order)
        assert m.nbody == len(g) + 1
        assert ph.spawn_clearance > 0
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        lowest = min(d.geom_xpos[m.geom(f"g{nid}").id][2] for nid in g.ids)
        assert lowest > 0.0
