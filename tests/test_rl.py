import pickle

import numpy as np
import pytest

from robogene import rl
from robogene.envs import RoboGeneEnv

from test_envs import GENOTYPE, robot, small_cfg  # noqa: F401  (robot is a fixture)


def tiny_cfg(task: str = "locomotion") -> dict:
    return small_cfg(task, rl={"budget": 512, "n_steps": 128, "batch_size": 64, "n_epochs": 2})


def test_make_env_is_picklable(robot):
    factory = rl.make_env(robot, tiny_cfg(), seed=3)
    env = pickle.loads(pickle.dumps(factory))()
    obs, _ = env.reset(seed=0)
    assert isinstance(env.unwrapped, RoboGeneEnv)
    assert obs.shape == env.observation_space.shape


def test_policy_kwargs():
    import torch

    kw = rl._policy_kwargs(small_cfg()["rl"])
    assert kw["net_arch"] == {"pi": [64, 64], "vf": [64, 64]}
    assert kw["activation_fn"] is torch.nn.Tanh and kw["ortho_init"]
    with pytest.raises(ValueError):
        rl._policy_kwargs({"net_arch": [8], "activation": "sigmoid"})


def test_train_save_load_and_deterministic_eval(robot, tmp_path):
    cfg = tiny_cfg()
    model = rl.train_policy(robot, cfg, seed=0, log_dir=tmp_path / "log")
    assert model.num_timesteps >= 512
    assert (tmp_path / "log" / "progress.csv").exists()
    model.save(str(tmp_path / "policy.zip"))
    loaded = rl.load_policy(tmp_path / "policy.zip")

    first = rl.evaluate_policy(model, robot, cfg)
    second = rl.evaluate_policy(loaded, robot, cfg)
    assert first["episodes"] == 2 and len(first["per_episode"]) == 2
    assert set(first["per_episode"][0]) == {"loco", "manip", "steps", "fell"}
    assert first["per_episode"][0]["steps"] == 50
    assert first["fitness"] == first["loco"] and first["manip"] == 0.0
    assert first["loco"] == second["loco"]
    assert [e["loco"] for e in first["per_episode"]] == [e["loco"] for e in second["per_episode"]]


def test_evaluate_zero_policy_uses_fixed_goals(robot):
    cfg = small_cfg("loco_manip")
    a = rl.evaluate_policy(None, robot, cfg, episodes=2, seed=7)
    b = rl.evaluate_policy(None, robot, cfg, episodes=2, seed=7)
    assert a == b
    assert a["fitness"] == a["manip"]
    assert all(np.isfinite(e["loco"]) for e in a["per_episode"])


def test_evaluate_individual(robot, tmp_path):
    result = rl.evaluate_individual(GENOTYPE, tiny_cfg("loco_manip"), tmp_path / "ind", seed=0)
    assert "error" not in result, result.get("traceback")
    for key in ("fitness", "loco", "manip", "obs_dim", "act_dim", "train_time_s", "policy", "morphology_class"):
        assert key in result
    assert result["act_dim"] == robot.n_actuators
    assert result["fitness"] == result["manip"]
    assert result["policy"] == "policy.zip"
    assert (tmp_path / "ind" / "policy.zip").exists()
    assert "<mujoco" in (tmp_path / "ind" / "model.xml").read_text()
    loaded = rl.load_policy(tmp_path / "ind" / "policy.zip")
    assert loaded.observation_space.shape == (result["obs_dim"],)


def test_evaluate_individual_reports_errors(tmp_path):
    bad = {"root": 0, "nodes": [{"id": 0, "module": "no_such_module", "params": {}}], "edges": []}
    result = rl.evaluate_individual(bad, tiny_cfg(), tmp_path / "bad", seed=0)
    assert result["fitness"] == 0.0 and "error" in result
