"""Experiment configuration: YAML files merged onto DEFAULT_CONFIG."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    # "locomotion" or "loco_manip"
    "task": "locomotion",
    "run_dir": "runs",
    "simulation": {
        "timestep": 0.001,
        "gravity": 1.62,  # m/s^2
        "frame_skip": 20,  # 50 Hz control
        "integrator": "implicitfast",
        "settle_time": 1.0,  # s of zero action before checks and episodes
    },
    "terrain": {
        "type": "flat",  # "flat" or "hilly"
        "friction": 0.5,
        "hilly": {
            "size_x": 60.0,
            "size_y": 20.0,
            "nx": 97,
            "ny": 33,
            "mean": 0.5,  # normalized heights ~ N(mean, std), clipped to [0, 1]
            "std": 0.125,
            "max_height": 0.5,
            "center_x": 20.0,
            "seed": 0,
        },
    },
    "episode": {
        "length": 1000,  # control steps
        "eval_episodes": 5,
        "eval_seed": 1000,
    },
    "goal": {
        "x_range": [-15.0, 15.0],
        "y_range": [-15.0, 15.0],
        "min_distance": 5.0,
        "target_height": [0.2, 0.6],  # arm target above the goal, loco_manip only
    },
    "reward": {
        "alpha": 1.0,
        "beta": 0.01,
        "delta": 1.0,
        "zeta": 0.1,
        "w_arm": 1.0,
        "gamma_r": 0.001,
        "fall_tilt_deg": 90.0,
        "terminate_on_fall": False,
    },
    "ga": {
        "population": 16,
        "elites": 4,
        "generations": 99,
        "tournament_size": 3,
        "crossover_prob": 0.7,
        "mutation_prob": 0.5,
        "mutation_weights": {
            "substitute": 1.0,
            "reassign_port": 1.0,
            "add_leaf": 1.0,
            "remove_leaf": 1.0,
            "perturb_limits": 1.0,
        },
        "min_modules": 2,
        "max_modules": 10,
        "init_modules": [2, 7],
        "max_offspring_attempts": 500,
        "require_wheel": False,
        "require_arm": False,
    },
    "checks": {
        "connectivity": True,
        "ports": True,
        "collision": True,
        "mass_inertia": True,
        "mass_inertia_tol": 0.10,
        "self_penetration": True,
        "stability": False,
        "max_tilt_deg": 5.0,
        "com_margin": 0.05,
        "reachability": False,
        "reach_min": 0.3,
        "align_tol_deg": 5.0,
    },
    "rl": {
        "budget": 200000,  # environment steps per candidate
        "n_envs": 1,
        "learning_rate": 3.0e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "net_arch": [64, 64],
        "activation": "tanh",
        "ortho_init": True,
        "device": "cpu",
    },
    "parallel": {
        "workers": 1,  # candidates trained in parallel
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config onto the defaults and apply overrides."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if cfg["task"] not in ("locomotion", "loco_manip"):
        raise ValueError(f"unknown task {cfg['task']!r}")
    if cfg["terrain"]["type"] not in ("flat", "hilly"):
        raise ValueError(f"unknown terrain {cfg['terrain']['type']!r}")
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
