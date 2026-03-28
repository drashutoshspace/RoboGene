import mujoco
import numpy as np
import pytest

from robogene import phenotype
from robogene.config import load_config
from robogene.envs import RoboGeneEnv
from robogene.genotype import Genotype
from robogene.phenotype import ActuatorInfo, Phenotype

# body_small with two wheels at the front and a two-segment arm on top
GENOTYPE = {
    "root": 0,
    "nodes": [
        {"id": 0, "module": "body_small", "params": {}},
        {"id": 1, "module": "wheel_left", "params": {}},
        {"id": 2, "module": "wheel_right", "params": {}},
        {"id": 3, "module": "arm_link", "params": {}},
        {"id": 4, "module": "arm_link", "params": {}},
    ],
    "edges": [
        {"parent": 0, "parent_port": "+y_front", "child": 1, "child_port": "hub", "roll_deg": 0.0},
        {"parent": 0, "parent_port": "-y_front", "child": 2, "child_port": "hub", "roll_deg": 0.0},
        {"parent": 0, "parent_port": "+z", "child": 3, "child_port": "end_a", "roll_deg": 0.0},
        {"parent": 3, "parent_port": "end_b", "child": 4, "child_port": "end_a", "roll_deg": 0.0},
    ],
}

_WHEEL = """
    <body name="n{i}" pos="0.1 {y} 0">
      <joint name="j{i}" type="hinge" axis="0 {s} 0" damping="0.05" armature="0.005"/>
      <geom name="g{i}" type="cylinder" size="0.1 0.025" zaxis="0 {s} 0" density="1000"/>
    </body>"""

_ARM_JOINT = '<joint name="j{i}" type="hinge" pos="0 0 -0.15" axis="1 0 0" range="-1.57 1.57" damping="0.3" armature="0.01"/>'

_ROBOT = f"""
  <body name="n0" pos="0 0 0.12">
    <freejoint name="root"/>
    <site name="base" size="0.01"/>
    <geom name="g0" type="box" size="0.15 0.1 0.05" density="500"/>
    {_WHEEL.format(i=1, y=0.125, s=1)}
    {_WHEEL.format(i=2, y=-0.125, s=-1)}
    <body name="n3" pos="0 0 0.2">
      {_ARM_JOINT.format(i=3)}
      <geom name="g3" type="capsule" size="0.03 0.12" density="1000"/>
      <body name="n4" pos="0 0 0.3">
        {_ARM_JOINT.format(i=4)}
        <geom name="g4" type="capsule" size="0.03 0.12" density="1000"/>
        <site name="ee" pos="0 0 0.15" size="0.01"/>
        {{extra}}
      </body>
    </body>
  </body>"""

_ACTUATORS = """
  <velocity name="a1" joint="j1" kv="3" ctrlrange="-6 6" forcerange="-3 3"/>
  <velocity name="a2" joint="j2" kv="3" ctrlrange="-6 6" forcerange="-3 3"/>
  <position name="a3" joint="j3" kp="15" ctrlrange="-1.57 1.57" forcerange="-8 8"/>
  <position name="a4" joint="j4" kp="15" ctrlrange="-1.57 1.57" forcerange="-8 8"/>"""

# a sphere hanging from the second arm segment into the body, for the self-collision test
_OVERLAP = '<body name="n5" pos="0 0 -0.5"><geom name="g5" type="sphere" size="0.04" density="1000"/></body>'


def _fake_xml(ph: Phenotype, cfg: dict, spawn_xy=(0.0, 0.0), extra: str = "") -> str:
    sim = cfg["simulation"]
    friction = cfg["terrain"]["friction"]
    return f"""<mujoco model="fixture">
  <option timestep="{sim['timestep']}" gravity="0 0 {-sim['gravity']}" integrator="{sim['integrator']}"/>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1" friction="{friction} 0.005 0.0001"/>
    {ph.robot_xml.replace("{extra}", extra)}
  </worldbody>
  <actuator>{ph.actuator_xml}</actuator>
</mujoco>"""


def _fake_phenotype() -> Phenotype:
    acts = [
        ActuatorInfo("a1", "j1", 1, "wheel_left", "wheel", "velocity", (-6.0, 6.0), 6.0),
        ActuatorInfo("a2", "j2", 2, "wheel_right", "wheel", "velocity", (-6.0, 6.0), 6.0),
        ActuatorInfo("a3", "j3", 3, "arm_link", "arm", "position", (-1.57, 1.57), 3.0),
        ActuatorInfo("a4", "j4", 4, "arm_link", "arm", "position", (-1.57, 1.57), 3.0),
    ]
    return Phenotype(GENOTYPE, "2Arm+2Wheel", _ROBOT, _ACTUATORS, acts, 4, 3.0, 0.1)


def real_phenotype_available() -> bool:
    try:
        ph = phenotype.expand(Genotype.from_dict(GENOTYPE))
        phenotype.build_model(ph, load_config())
    except NotImplementedError:
        return False
    return True


def install_fake(monkeypatch, extra: str = "") -> Phenotype:
    """Replace the phenotype functions with the hand-written fixture model."""
    monkeypatch.setattr(phenotype, "expand", lambda g, library=None: _fake_phenotype())
    monkeypatch.setattr(phenotype, "model_xml", lambda ph, cfg, spawn_xy=(0.0, 0.0): _fake_xml(ph, cfg, spawn_xy, extra))
    monkeypatch.setattr(
        phenotype, "build_model",
        lambda ph, cfg, spawn_xy=(0.0, 0.0): mujoco.MjModel.from_xml_string(_fake_xml(ph, cfg, spawn_xy, extra)),
    )
    return _fake_phenotype()


@pytest.fixture
def robot(monkeypatch) -> Phenotype:
    if real_phenotype_available():
        return phenotype.expand(Genotype.from_dict(GENOTYPE))
    return install_fake(monkeypatch)


def small_cfg(task: str = "locomotion", **sections) -> dict:
    overrides = {"task": task, "episode": {"length": 50, "eval_episodes": 2}}
    overrides.update(sections)
    return load_config(overrides=overrides)


def shift_base(env: RoboGeneEnv, dx: float) -> None:
    env.data.qpos[env.root_qpos_adr] += dx
    mujoco.mj_forward(env.model, env.data)


def test_spaces_and_shapes(robot):
    n_act = robot.n_actuators
    n_pos = sum(a.control_mode == "position" for a in robot.actuators)
    n_flags = sum(a.category == "wheel" for a in robot.actuators) + (robot.end_effector is not None)
    for task, extra in (("locomotion", 0), ("loco_manip", 3)):
        env = RoboGeneEnv(robot, small_cfg(task))
        assert env.action_space.shape == (n_act,)
        assert env.observation_space.shape == (n_pos + n_act + 9 + n_flags + extra,)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape and np.isfinite(reward)
        assert set(info) >= {"progress", "arm_progress", "fall", "collision", "base_pos", "ee_pos"}


def test_action_maps_to_ctrlrange(robot):
    env = RoboGeneEnv(robot, small_cfg())
    lo, hi = env.model.actuator_ctrlrange[env.act_ids].T
    assert np.allclose(env.action_to_ctrl(-np.ones(env.n_act)), lo)
    assert np.allclose(env.action_to_ctrl(np.ones(env.n_act)), hi)
    assert np.allclose(env.action_to_ctrl(np.zeros(env.n_act)), 0.5 * (lo + hi))
    assert np.allclose(env.action_to_ctrl(5 * np.ones(env.n_act)), hi)
    for a, (rlo, rhi) in zip(robot.actuators, zip(lo, hi)):
        assert np.allclose(a.ctrl_range, (rlo, rhi))
    env.reset(seed=0)
    action = np.linspace(-1, 1, env.n_act)
    env.step(action)
    assert np.allclose(env.data.ctrl[env.act_ids], env.action_to_ctrl(action))


def test_settle_brings_base_to_rest(robot):
    env = RoboGeneEnv(robot, small_cfg())
    obs, info = env.reset(seed=0)
    root_vel = env.data.qvel[env.root_qvel_adr:env.root_qvel_adr + 6]
    assert np.linalg.norm(root_vel[:3]) < 0.05
    assert info["base_pos"][2] > 0.0
    assert np.linalg.norm(info["base_pos"][:2]) < 0.1
    n_wheels = sum(a.category == "wheel" for a in robot.actuators)
    flags, _ = env.ground_contacts()
    assert np.all(flags[:n_wheels] == 1.0)


def test_goal_sampling(robot):
    cfg = small_cfg()
    env = RoboGeneEnv(robot, cfg)
    goals = []
    for seed in range(20):
        _, info = env.reset(seed=seed)
        g = info["goal"]
        assert cfg["goal"]["x_range"][0] <= g[0] <= cfg["goal"]["x_range"][1]
        assert cfg["goal"]["y_range"][0] <= g[1] <= cfg["goal"]["y_range"][1]
        assert np.linalg.norm(g) >= cfg["goal"]["min_distance"]
        goals.append(g)
    _, info = env.reset(seed=3)
    assert np.allclose(info["goal"], goals[3])
    _, info = env.reset(options={"goal": (7.0, -2.0)})
    assert np.allclose(info["goal"], (7.0, -2.0))


def test_progress_sign(robot):
    env = RoboGeneEnv(robot, small_cfg())
    env.reset(seed=0, options={"goal": (10.0, 0.0)})
    zero = np.zeros(env.n_act)
    shift_base(env, 0.5)
    _, _, _, _, info = env.step(zero)
    assert info["progress"] == pytest.approx(0.5, abs=0.05)
    shift_base(env, -1.0)
    _, _, _, _, info = env.step(zero)
    assert info["progress"] == pytest.approx(-1.0, abs=0.05)


def test_progress_telescopes(robot):
    env = RoboGeneEnv(robot, small_cfg())
    env.action_space.seed(0)
    _, info = env.reset(seed=1)
    goal = info["goal"]
    start = np.linalg.norm(info["base_pos"][:2] - goal)
    total, steps = 0.0, 0
    while True:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        total += info["progress"]
        steps += 1
        if terminated or truncated:
            break
    assert steps == 50 and truncated
    end = np.linalg.norm(info["base_pos"][:2] - goal)
    assert total == pytest.approx(start - end, abs=1e-9)


def test_arm_progress(robot):
    env = RoboGeneEnv(robot, small_cfg("loco_manip"))
    _, info = env.reset(seed=0, options={"goal": (5.0, 0.0), "target_height": 0.4})
    assert np.allclose(info["target"], (5.0, 0.0, 0.4))
    start = np.linalg.norm(info["ee_pos"] - info["target"])
    shift_base(env, 0.5)
    _, reward, _, _, info = env.step(np.zeros(env.n_act))
    assert info["arm_progress"] == pytest.approx(0.5, abs=0.1)
    assert info["progress"] == pytest.approx(0.5, abs=0.05)
    assert reward > 0.0
    total = info["arm_progress"]
    for _ in range(10):
        _, _, _, _, info = env.step(env.action_space.sample())
        total += info["arm_progress"]
    end = np.linalg.norm(info["ee_pos"] - info["target"])
    assert total == pytest.approx(start - end, abs=1e-9)


def test_fall_flag_and_termination(robot):
    cfg = small_cfg()
    env = RoboGeneEnv(robot, cfg)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.n_act))
    assert not info["fall"]
    env.data.qpos[env.root_qpos_adr + 2] += 1.0
    env.data.qpos[env.root_qpos_adr + 3:env.root_qpos_adr + 7] = (0.0, 1.0, 0.0, 0.0)  # upside down
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.zeros(env.n_act))
    assert info["fall"] and not terminated

    env = RoboGeneEnv(robot, small_cfg(reward={"terminate_on_fall": True}))
    env.reset(seed=0)
    env.data.qpos[env.root_qpos_adr + 2] += 1.0
    env.data.qpos[env.root_qpos_adr + 3:env.root_qpos_adr + 7] = (0.0, 1.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.zeros(env.n_act))
    assert info["fall"] and terminated


def test_collision_flag(monkeypatch):
    clean = install_fake(monkeypatch)
    env = RoboGeneEnv(clean, small_cfg())
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(env.n_act))
    assert not info["collision"]

    overlapping = install_fake(monkeypatch, extra=_OVERLAP)
    cfg = small_cfg(reward={"zeta": 0.5, "beta": 0.0})
    env = RoboGeneEnv(overlapping, cfg)
    _, info = env.reset(seed=0, options={"goal": (10.0, 0.0)})
    assert info["collision"]
    _, reward, _, _, info = env.step(np.zeros(env.n_act))
    assert info["collision"]
    assert reward == pytest.approx(info["progress"] - 0.5)


def test_blowup_terminates(robot):
    env = RoboGeneEnv(robot, small_cfg())
    env.reset(seed=0)
    env.data.qvel[:] = 1.0e6
    _, reward, terminated, _, info = env.step(np.zeros(env.n_act))
    assert terminated and reward == 0.0 and info["progress"] == 0.0
