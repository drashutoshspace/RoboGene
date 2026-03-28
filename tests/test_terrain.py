import mujoco
import numpy as np
import pytest

from robogene import terrain
from robogene.config import load_config
from robogene.genotype import Genotype, Node
from robogene.phenotype import build_model, expand


@pytest.fixture(scope="module")
def hilly_cfg() -> dict:
    return load_config(overrides={"terrain": {"type": "hilly"}})


def _ground_only_model(terrain_cfg: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml = (
        f"<mujoco><asset>{terrain.terrain_assets_xml(terrain_cfg)}</asset>"
        f"<worldbody>{terrain.terrain_geom_xml(terrain_cfg)}</worldbody></mujoco>"
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _ray_height(m: mujoco.MjModel, d: mujoco.MjData, x: float, y: float) -> float:
    top = 10.0
    geomid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    assert dist >= 0
    return top - dist


def _wheeled_robot() -> Genotype:
    g = Genotype([Node(0, "body_small")], [])
    for port, wheel in [("+y_front", "wheel_left"), ("+y_rear", "wheel_left"),
                        ("-y_front", "wheel_right"), ("-y_rear", "wheel_right")]:
        g.add_node(wheel, 0, port, "hub")
    return g


def test_flat_ground_is_a_plane():
    cfg = load_config()
    m, _ = _ground_only_model(cfg["terrain"])
    assert m.geom("ground").type == mujoco.mjtGeom.mjGEOM_PLANE
    assert terrain.ground_height(cfg["terrain"], 3.0, -2.0) == 0.0


def test_hilly_heights_match_reference(hilly_cfg):
    h = hilly_cfg["terrain"]["hilly"]
    ref = np.clip(np.random.RandomState(h["seed"]).normal(h["mean"], h["std"], size=(h["nx"], h["ny"])), 0, 1)
    heights = terrain.hilly_heights(hilly_cfg["terrain"])
    assert heights.shape == (h["ny"], h["nx"])
    np.testing.assert_array_equal(heights, ref.T)


def test_hfield_size_and_placement(hilly_cfg):
    tc = hilly_cfg["terrain"]
    h = tc["hilly"]
    m, _ = _ground_only_model(tc)
    assert (m.hfield_nrow[0], m.hfield_ncol[0]) == (h["ny"], h["nx"])
    np.testing.assert_allclose(m.hfield_size[0][:2], [h["size_x"] / 2, h["size_y"] / 2])
    assert m.geom("ground").pos[0] == pytest.approx(h["center_x"])
    heights = terrain.hilly_heights(tc)
    surface = m.geom("ground").pos[2] + m.hfield_data.reshape(heights.shape) * m.hfield_size[0][2]
    np.testing.assert_allclose(surface, (heights - h["mean"]) * h["max_height"], atol=1e-6)


def test_ground_height_agrees_with_ray(hilly_cfg):
    tc = hilly_cfg["terrain"]
    m, d = _ground_only_model(tc)
    h = tc["hilly"]
    rng = np.random.default_rng(0)
    x0, x1 = h["center_x"] - h["size_x"] / 2, h["center_x"] + h["size_x"] / 2
    points = [(rng.uniform(x0, x1), rng.uniform(-h["size_y"] / 2, h["size_y"] / 2)) for _ in range(300)]
    points += [(x0 - 3.0, 0.0), (x1 + 3.0, 2.0), (10.0, h["size_y"] / 2 + 2.0)]
    for x, y in points:
        assert _ray_height(m, d, x, y) == pytest.approx(terrain.ground_height(tc, x, y), abs=1e-3)


def test_max_ground_height_bounds_samples(hilly_cfg):
    tc = hilly_cfg["terrain"]
    h = tc["hilly"]
    grid_x = h["center_x"] - h["size_x"] / 2 + np.arange(h["nx"]) * h["size_x"] / (h["nx"] - 1)
    grid_y = -h["size_y"] / 2 + np.arange(h["ny"]) * h["size_y"] / (h["ny"] - 1)
    rng = np.random.default_rng(1)
    for _ in range(30):
        x, y, w = rng.uniform(-10, 48), rng.uniform(-10, 8), rng.uniform(0.2, 2.0)
        top = terrain.max_ground_height(tc, (x, x + w), (y, y + w))
        xs = np.concatenate([np.linspace(x, x + w, 81), grid_x[(grid_x >= x) & (grid_x <= x + w)]])
        ys = np.concatenate([np.linspace(y, y + w, 81), grid_y[(grid_y >= y) & (grid_y <= y + w)]])
        samples = [terrain.ground_height(tc, a, b) for a in xs for b in ys]
        assert max(samples) - 1e-9 <= top <= max(samples) + 2e-3


def test_robot_spawns_above_hills(hilly_cfg):
    ph = expand(_wheeled_robot())
    ground_names = ("ground", "ground_base")
    for xy in [(0.0, 0.0), (12.3, -4.1), (33.0, 6.5), (49.5, 0.0)]:
        m = build_model(ph, hilly_cfg, spawn_xy=xy)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ground = {m.geom(n).id for n in ground_names}
        assert not any((c.geom1 in ground) != (c.geom2 in ground) for c in d.contact[: d.ncon])
        for _ in range(1500):
            mujoco.mj_step(m, d)
        assert np.all(np.isfinite(d.qpos))
        assert d.qpos[2] > terrain.ground_height(hilly_cfg["terrain"], *d.qpos[:2])


@pytest.mark.parametrize("friction", [0.2, 0.8])
def test_ground_friction_has_priority(friction):
    cfg = load_config(overrides={"terrain": {"friction": friction}})
    m = build_model(expand(_wheeled_robot()), cfg)
    assert m.geom("g1").friction[0] == pytest.approx(1.0)
    d = mujoco.MjData(m)
    for _ in range(1000):
        mujoco.mj_step(m, d)
    ground = m.geom("ground").id
    frictions = [c.friction[0] for c in d.contact[: d.ncon] if ground in (c.geom1, c.geom2)]
    assert frictions
    np.testing.assert_allclose(frictions, friction)
