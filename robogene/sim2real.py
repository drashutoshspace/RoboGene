"""MoonBot hardware commands from recorded policy actions, for open-loop replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from .config import deep_merge, load_config
from .schema import Library, default_library


@dataclass
class WheelMap:
    """Hardware wheel k (0-based), its module (1-based) and joint, and simulated wheel sigma(k)."""

    k: int
    hardware: str
    module: int
    module_joint: str
    node: int
    actuator: int
    sign: float
    v_max: float


def load_mapping(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_wheels(mapping: dict, ph, library: Library | None = None) -> list[WheelMap]:
    """Check the wheel part of the mapping against the phenotype and build sigma and g_k."""
    library = library or default_library()
    name = mapping.get("hardware", {}).get("wheel_module")
    if name not in library.hardware:
        raise ValueError(f"hardware wheel module {name!r} is not in the library ({sorted(library.hardware)})")
    joints = library.hardware[name].joints
    wheels = mapping.get("wheels") or []
    if not wheels:
        raise ValueError("the mapping has no 'wheels' entries")
    nodes = {int(n["id"]): n["module"] for n in ph.genotype["nodes"]}
    act_of = {a.node: i for i, a in enumerate(ph.actuators)}
    out, seen = [], set()
    for k, w in enumerate(wheels):
        label = w.get("hardware", f"wheel {k + 1}")
        node = int(w["node"])
        sign = float(w.get("sign", 1.0))
        if node not in nodes:
            raise ValueError(f"{label}: node {node} is not in the genotype (nodes {sorted(nodes)})")
        if node not in act_of or ph.actuators[act_of[node]].category != "wheel":
            raise ValueError(f"{label}: node {node} ({nodes[node]}) is not an actuated wheel")
        if sign not in (1.0, -1.0):
            raise ValueError(f"{label}: sign must be +1 or -1, got {sign}")
        if node in seen:
            raise ValueError(f"{label}: node {node} is mapped to more than one hardware wheel")
        seen.add(node)
        # hardware wheels fill module 1 first, then module 2
        joint = joints[k % len(joints)]
        out.append(WheelMap(k, label, k // len(joints) + 1, joint["name"], node, act_of[node], sign,
                            float(joint["velocity_limit"])))
    return out


def wheel_commands(actions: np.ndarray, wheels: list[WheelMap]) -> np.ndarray:
    """Wheel velocity commands (T, n_wheels) in rad/s."""
    a = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
    return np.stack([w.sign * a[:, w.actuator] * w.v_max for w in wheels], axis=1)


def limb_command(actions: np.ndarray, ph, center_arm_node: int) -> float:
    """Limb command a_c: mean commanded position (rad) of the center-arm joint, 0 without a joint."""
    nodes = {int(n["id"]) for n in ph.genotype["nodes"]}
    if center_arm_node not in nodes:
        raise ValueError(f"center arm node {center_arm_node} is not in the genotype (nodes {sorted(nodes)})")
    idx = [i for i, a in enumerate(ph.actuators) if a.node == center_arm_node]
    if not idx:
        return 0.0
    act = ph.actuators[idx[0]]
    if act.category == "wheel":
        raise ValueError(f"center arm node {center_arm_node} is a wheel")
    lo, hi = act.ctrl_range
    a = np.clip(np.asarray(actions, dtype=float)[:, idx[0]], -1.0, 1.0)
    return float(np.mean(lo + (a + 1.0) * 0.5 * (hi - lo)))


def hardware_commands(
    actions: np.ndarray, ph, mapping: dict, control_dt: float, library: Library | None = None
) -> tuple[pd.DataFrame, dict]:
    """Command table (t, wheel_k, limb_j) and the scaling used, from recorded actions."""
    library = library or default_library()
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != len(ph.actuators):
        raise ValueError(f"actions must have shape (T, {len(ph.actuators)}), got {actions.shape}")
    wheels = resolve_wheels(mapping, ph, library)
    limb = mapping.get("limb", {})
    n_limb = int(limb.get("joints", 7))
    limb_name = mapping.get("hardware", {}).get("limb")
    if limb_name is not None:
        if limb_name not in library.hardware:
            raise ValueError(f"hardware limb {limb_name!r} is not in the library")
        if len(library.hardware[limb_name].joints) != n_limb:
            raise ValueError(f"limb.joints = {n_limb} but {limb_name} has {len(library.hardware[limb_name].joints)} joints")
    a_c = limb_command(actions, ph, int(limb["center_arm_node"])) if "center_arm_node" in limb else 0.0

    T = len(actions)
    table = {"t": np.arange(T) * control_dt}
    u = wheel_commands(actions, wheels)
    for w in wheels:
        table[f"wheel_{w.k + 1}"] = u[:, w.k]
    for j in range(n_limb):
        table[f"limb_{j + 1}"] = np.full(T, a_c)
    scaling = {
        "wheels": [
            {"hardware": w.hardware, "module": w.module, "joint": w.module_joint, "node": w.node,
             "actuator": w.actuator, "sign": w.sign, "v_max_rad_s": w.v_max, "scale": w.sign * w.v_max}
            for w in wheels
        ],
        "limb": {"center_arm_node": limb.get("center_arm_node"), "a_c_rad": a_c, "joints": n_limb},
    }
    return pd.DataFrame(table), scaling


def simulate_actions(model, ph, cfg: dict, steps: int, goal) -> np.ndarray:
    """Deterministic rollout of the trained policy, returns the (T, n_act) normalized actions."""
    from .envs import RoboGeneEnv

    cfg = deep_merge(cfg, {"episode": {"length": max(int(steps), int(cfg["episode"]["length"]))}})
    env = RoboGeneEnv(ph, cfg)
    obs, _ = env.reset(seed=int(cfg.get("seed", 0)), options={"goal": tuple(float(v) for v in goal)})
    actions = []
    for _ in range(int(steps)):
        a, _ = model.predict(obs, deterministic=True)
        a = np.clip(np.asarray(a, dtype=float).reshape(-1), -1.0, 1.0)
        actions.append(a)
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break
    env.close()
    return np.array(actions)


def run_sim2real(
    entry_path: str | Path,
    mapping_path: str | Path,
    out: str | Path | None = None,
    config_path: str | Path | None = None,
    rollout_fn: Callable[..., np.ndarray] | None = None,
    phenotype=None,
    model=None,
) -> tuple[pd.DataFrame, dict]:
    """Roll out the policy, map its actions to hardware commands and write the CSV and JSON."""
    entry_path = Path(entry_path)
    entry = json.loads(entry_path.read_text())
    folder = entry_path.parent
    config_path = Path(config_path) if config_path else folder.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"run config {config_path} not found, pass --config")
    cfg = load_config(config_path)
    mapping = load_mapping(mapping_path)

    if phenotype is None:
        from .genotype import Genotype
        from .phenotype import expand

        phenotype = expand(Genotype.from_dict(entry["genotype"]))
    policy_path = folder / entry.get("policy", {}).get("checkpoint", "policy.zip")
    if model is None and rollout_fn is None:
        from .rl import load_policy

        model = load_policy(str(policy_path))
    resolve_wheels(mapping, phenotype)  # fail before the rollout on a bad mapping

    rollout = mapping.get("rollout", {})
    steps = int(rollout.get("steps", cfg["episode"]["length"]))
    goal = rollout.get("goal", [10.0, 0.0])
    actions = (rollout_fn or simulate_actions)(model, phenotype, cfg, steps, goal)
    dt = cfg["simulation"]["timestep"] * cfg["simulation"]["frame_skip"]
    df, scaling = hardware_commands(actions, phenotype, mapping, dt)

    meta = {
        "transfer": {
            "wheel": "u_wheel_{k,t} = sign_k * a_{t,sigma(k)} * v_max_k",
            "limb": "u_limb_{j,t} = a_c for every limb joint j",
        },
        "mapping": mapping,
        "scaling": scaling,
        "units": {"t": "s", "wheel": "rad/s", "limb": "rad"},
        "control_dt_s": dt,
        "control_hz": 1.0 / dt,
        "steps_recorded": int(len(df)),
        "sources": {
            "entry": str(entry_path),
            "entry_id": entry.get("id"),
            "morphology_class": entry.get("morphology_class"),
            "policy": str(policy_path),
            "run_config": str(config_path),
            "hardware": [mapping.get("hardware", {}).get("wheel_module"), mapping.get("hardware", {}).get("limb")],
        },
        "note": "Recorded from a deterministic rollout in simulation. The commands are replayed on the "
                "hardware in open loop, without feedback from the robot.",
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return df, meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m robogene.sim2real", description=__doc__.split("\n")[0])
    p.add_argument("--entry", required=True, help="robogene.json of the individual")
    p.add_argument("--mapping", required=True, help="e.g. configs/sim2real_vehicle.yaml")
    p.add_argument("--out", default="commands.csv")
    p.add_argument("--config", help="run config (default: config.yaml two levels above the entry folder)")
    args = p.parse_args(argv)
    df, meta = run_sim2real(args.entry, args.mapping, args.out, args.config)
    wheels = [c for c in df.columns if c.startswith("wheel_")]
    print(f"{len(df)} steps at {meta['control_hz']:.0f} Hz, a_c = {meta['scaling']['limb']['a_c_rad']:.4f} rad, "
          f"max |wheel| = {np.abs(df[wheels].to_numpy()).max():.4f} rad/s")
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
