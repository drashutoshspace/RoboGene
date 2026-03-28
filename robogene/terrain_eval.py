"""Evaluate flat-trained policies on hilly terrain without retraining."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import yaml

from .config import deep_merge, load_config

COLUMNS = ["id", "run", "hash", "morphology_class", "flat_fitness", "hilly_fitness", "ratio"]

# a different task would change the observation size the policy was trained with
KEEP_FROM_RUN = ("task", "seed", "run_dir")


def iter_run_entries(run: str | Path) -> Iterator[tuple[Path, dict]]:
    """(folder, robogene.json entry) of every individual of a run."""
    run = Path(run)
    try:
        from .archive import iter_entries
    except ImportError:
        iter_entries = None
    if iter_entries is None:
        for f in sorted(run.glob("individuals/*/robogene.json")):
            yield f.parent, json.loads(f.read_text())
        return
    for item in iter_entries(run):
        yield _normalize(run, item)


def _normalize(run: Path, item: Any) -> tuple[Path, dict]:
    if isinstance(item, tuple):
        a, b = item
        path, entry = (a, b) if isinstance(b, dict) else (b, a)
    elif isinstance(item, dict):
        entry = item
        path = item.get("_dir") or item.get("dir") or item.get("path") or run / "individuals" / item["id"]
    else:
        path = Path(item)
        entry = json.loads((path / "robogene.json" if path.is_dir() else path).read_text())
    path = Path(path)
    if path.name == "robogene.json":
        path = path.parent
    return path, entry


def collect_entries(runs: list[str | Path]) -> list[dict]:
    """One job per morphology hash over all runs: the entry with the best flat fitness."""
    best: dict[str, dict] = {}
    for run in runs:
        run = Path(run)
        for folder, entry in iter_run_entries(run):
            h = entry.get("hash") or entry["id"]
            loco = float(entry["fitness"]["loco"])
            if h not in best or loco > best[h]["flat"]:
                best[h] = {"run": run, "folder": folder, "entry": entry, "flat": loco}
    return sorted(best.values(), key=lambda j: -j["flat"])


def hilly_config(run: str | Path, terrain_cfg: dict) -> dict:
    """The run's training config with the terrain config merged on top."""
    path = Path(run) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, the run config is needed to match training")
    override = {k: v for k, v in terrain_cfg.items() if k not in KEEP_FROM_RUN}
    return load_config(path, overrides=override)


def evaluate_hilly(entry: dict, policy_path: str, cfg: dict, episodes: int | None = None) -> float:
    """Base progress (m) of a trained policy on the terrain of cfg."""
    from .genotype import Genotype
    from .phenotype import expand
    from .rl import evaluate_policy, load_policy

    ph = expand(Genotype.from_dict(entry["genotype"]))
    model = load_policy(policy_path)
    return float(evaluate_policy(model, ph, cfg, episodes=episodes)["loco"])


def _job(args: tuple) -> float:
    return evaluate_hilly(*args)


def ratio(hilly: float, flat: float) -> float:
    return hilly / flat if flat > 1e-6 else math.nan


def run_terrain_eval(
    runs: list[str | Path],
    terrain_config: str | Path | dict,
    out: str | Path | None = None,
    episodes: int | None = None,
    workers: int = 1,
    evaluate_fn: Callable[..., float] | None = None,
) -> pd.DataFrame:
    """Evaluate every unique morphology of the runs on hilly terrain and write the CSV."""
    if isinstance(terrain_config, dict):
        terrain_cfg = terrain_config
    else:
        with open(terrain_config) as f:
            terrain_cfg = yaml.safe_load(f) or {}
    jobs = collect_entries(runs)
    cfgs = {str(r): hilly_config(r, terrain_cfg) for r in {str(j["run"]) for j in jobs}}
    args = [
        (j["entry"], str(j["folder"] / j["entry"].get("policy", {}).get("checkpoint", "policy.zip")),
         cfgs[str(j["run"])], episodes)
        for j in jobs
    ]
    if workers > 1 and evaluate_fn is None:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            hilly = list(pool.map(_job, args))
    else:
        fn = evaluate_fn or evaluate_hilly
        hilly = [fn(*a) for a in args]

    rows = [
        {
            "id": j["entry"]["id"],
            "run": j["run"].name,
            "hash": j["entry"].get("hash", ""),
            "morphology_class": j["entry"].get("morphology_class", ""),
            "flat_fitness": j["flat"],
            "hilly_fitness": h,
            "ratio": ratio(h, j["flat"]),
        }
        for j, h in zip(jobs, hilly)
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def summary(df: pd.DataFrame) -> str:
    r = df["ratio"].to_numpy(dtype=float)
    finite = r[np.isfinite(r)]
    mean = f"{finite.mean():.3f}" if len(finite) else "n/a"
    lower = int((df["hilly_fitness"] < df["flat_fitness"]).sum())
    return f"{len(df)} morphologies, mean hilly/flat ratio {mean}, {lower} score lower on hilly terrain"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m robogene.terrain_eval", description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument("--config", required=True, help="terrain config, e.g. configs/terrain_hilly.yaml")
    p.add_argument("--out", default="terrain.csv")
    p.add_argument("--episodes", type=int, help="evaluation episodes (default episode.eval_episodes)")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    df = run_terrain_eval(args.runs, args.config, args.out, args.episodes, args.workers)
    print(summary(df))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
