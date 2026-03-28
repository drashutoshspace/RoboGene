"""PPO training of one morphology and its fitness evaluation."""

from __future__ import annotations

import functools
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import phenotype as phenotype_mod
from .envs import RoboGeneEnv
from .genotype import Genotype
from .phenotype import Phenotype

ACTIVATIONS = ("tanh", "relu")


def _build_env(ph: Phenotype, cfg: dict, seed: int | None, monitor_file: str | None) -> RoboGeneEnv:
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(RoboGeneEnv(ph, cfg), filename=monitor_file)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_env(ph: Phenotype, cfg: dict, seed: int | None = None, monitor_file: str | None = None) -> Callable[[], RoboGeneEnv]:
    """Picklable factory of a Monitor-wrapped RoboGeneEnv (usable by SubprocVecEnv)."""
    return functools.partial(_build_env, ph, cfg, seed, monitor_file)


def _policy_kwargs(rl: dict) -> dict[str, Any]:
    import torch

    act = rl.get("activation", "tanh")
    if act not in ACTIVATIONS:
        raise ValueError(f"unknown activation {act!r}, expected one of {ACTIVATIONS}")
    net = list(rl["net_arch"])
    return {
        "net_arch": {"pi": net, "vf": list(net)},
        "activation_fn": torch.nn.Tanh if act == "tanh" else torch.nn.ReLU,
        "ortho_init": bool(rl.get("ortho_init", True)),
    }


def train_policy(ph: Phenotype, cfg: dict, seed: int, log_dir: str | Path | None = None):
    """Train a PPO policy from scratch for rl.budget environment steps."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    # several candidates train in parallel processes
    torch.set_num_threads(1)
    rl = cfg["rl"]
    n_envs = int(rl.get("n_envs", 1))
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    monitor = (lambda i: str(Path(log_dir) / f"monitor_{i}")) if log_dir is not None else (lambda i: None)
    factories = [make_env(ph, cfg, seed + i, monitor(i)) for i in range(n_envs)]
    venv = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=rl["learning_rate"],
        n_steps=int(rl["n_steps"]),
        batch_size=int(rl["batch_size"]),
        n_epochs=int(rl["n_epochs"]),
        gamma=rl["gamma"],
        gae_lambda=rl["gae_lambda"],
        clip_range=rl["clip_range"],
        ent_coef=rl["ent_coef"],
        vf_coef=rl["vf_coef"],
        max_grad_norm=rl["max_grad_norm"],
        normalize_advantage=bool(rl.get("normalize_advantage", True)),
        policy_kwargs=_policy_kwargs(rl),
        device=rl.get("device", "cpu"),
        seed=int(seed),
        verbose=0,
    )
    if log_dir is not None:
        model.set_logger(configure(str(log_dir), ["csv"]))
    try:
        model.learn(total_timesteps=int(rl["budget"]))
    finally:
        venv.close()
    return model


def load_policy(path: str | Path, device: str = "cpu"):
    """Load a policy saved by evaluate_individual or train_policy."""
    from stable_baselines3 import PPO

    return PPO.load(str(path), device=device)


def evaluate_policy(model, ph: Phenotype, cfg: dict, episodes: int | None = None, seed: int | None = None) -> dict:
    """Mean summed progress in meters over fixed-seed episodes; model=None runs zero actions."""
    n = int(episodes if episodes is not None else cfg["episode"]["eval_episodes"])
    base_seed = int(seed if seed is not None else cfg["episode"]["eval_seed"])
    env = RoboGeneEnv(ph, cfg)
    per_episode = []
    try:
        for k in range(n):
            obs, _ = env.reset(seed=base_seed + k)
            loco = manip = 0.0
            steps, fell = 0, False
            while True:
                if model is None:
                    action = np.zeros(env.n_act, dtype=np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                loco += info["progress"]
                manip += info["arm_progress"]
                fell = fell or info["fall"]
                steps += 1
                if terminated or truncated:
                    break
            per_episode.append({"loco": loco, "manip": manip, "steps": steps, "fell": fell})
    finally:
        env.close()
    loco_mean = float(np.mean([e["loco"] for e in per_episode])) if per_episode else 0.0
    manip_mean = float(np.mean([e["manip"] for e in per_episode])) if per_episode else 0.0
    fitness = loco_mean if cfg["task"] == "locomotion" else manip_mean
    return {"loco": loco_mean, "manip": manip_mean, "fitness": fitness, "per_episode": per_episode, "episodes": n}


def evaluate_individual(genotype: dict, cfg: dict, out_dir: str | Path, seed: int) -> dict:
    """Expand, train and score one genotype; failures give fitness 0.0 and an "error" field."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "fitness": 0.0, "loco": 0.0, "manip": 0.0, "obs_dim": 0, "act_dim": 0,
        "train_time_s": 0.0, "policy": None, "morphology_class": None,
    }
    t0 = time.time()
    try:
        ph = phenotype_mod.expand(Genotype.from_dict(genotype))
        result["morphology_class"] = ph.morphology_class
        (out / "model.xml").write_text(phenotype_mod.model_xml(ph, cfg))
        if ph.n_actuators == 0:
            env = RoboGeneEnv(ph, cfg)
            result["obs_dim"], result["act_dim"] = int(env.observation_space.shape[0]), 0
            env.close()
            model = None
        else:
            model = train_policy(ph, cfg, seed)
            result["obs_dim"] = int(model.observation_space.shape[0])
            result["act_dim"] = int(model.action_space.shape[0])
            model.save(str(out / "policy.zip"))
            result["policy"] = "policy.zip"
        result["train_time_s"] = time.time() - t0
        scores = evaluate_policy(model, ph, cfg)
        for key in ("fitness", "loco", "manip"):
            value = float(scores[key])
            result[key] = value if np.isfinite(value) else 0.0
        result["episodes"] = scores["episodes"]
        result["per_episode"] = scores["per_episode"]
    except Exception as exc:  # one failed candidate must not stop the evolution
        result.update({"fitness": 0.0, "loco": 0.0, "manip": 0.0, "train_time_s": time.time() - t0})
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    return result
