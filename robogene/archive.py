"""Run directory: every evaluated individual, the evolution log and the final top K."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import save_config
from .genotype import Genotype, module_counts, morphology_class
from .schema import Library, default_library

LOG_COLUMNS = [
    "generation", "index", "id", "hash", "morphology_class", "n_modules", "arms", "wheels",
    "fitness", "loco", "manip", "train_time_s", "elite",
]
BEST_COLUMNS = ["generation", "id", "morphology_class", "fitness"]


def individual_id(generation: int, index: int) -> str:
    return f"g{generation:03d}_i{index:02d}"


def _finite(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _episode_count(metrics: dict, cfg: dict) -> int:
    n = metrics.get("episodes", cfg["episode"]["eval_episodes"])
    return len(n) if isinstance(n, (list, tuple)) else int(n)


class Archive:
    """Writes and indexes all evaluations of one evolution run."""

    def __init__(self, run_dir: str | Path, cfg: dict, library: Library | None = None):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.library = library or default_library()
        (self.run_dir / "individuals").mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.run_dir / "config.yaml")
        self.log_path = self.run_dir / "evolution_log.csv"
        self.best_path = self.run_dir / "best_per_generation.csv"
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)
        self.entries: list[dict] = []
        self.best: dict[int, dict] = {}

    def out_dir(self, generation: int, index: int) -> Path:
        return self.run_dir / "individuals" / individual_id(generation, index)

    def make_entry(self, generation: int, index: int, genotype: Genotype, metrics: dict) -> dict:
        """Robot-level RoboGene entry of one evaluation."""
        cfg, sim = self.cfg, self.cfg["simulation"]
        counts = module_counts(genotype, self.library)
        failed = "error" in metrics
        entry = {
            "robogene_version": "1.0",
            "type": "robot",
            "id": individual_id(generation, index),
            "generation": generation,
            "index": index,
            "task": cfg["task"],
            "genotype": genotype.to_dict(),
            "hash": genotype.canonical_hash(),
            "morphology_class": morphology_class(genotype, self.library),
            "modules": {
                "count": len(genotype),
                "arm": counts["arm"],
                "wheel": counts["wheel"],
                "body": counts["body"],
                "hinge": counts["hinge"],
                "connector": counts["connector"],
            },
            "fitness": {
                "value": None if failed else _finite(metrics.get("fitness")),
                "loco": None if failed else _finite(metrics.get("loco")),
                "manip": None if failed else _finite(metrics.get("manip")),
                "episodes": _episode_count(metrics, cfg),
                "unit": "m",
            },
            "policy": {
                "checkpoint": metrics.get("policy"),
                "obs_dim": metrics.get("obs_dim"),
                "act_dim": metrics.get("act_dim"),
                "control_hz": round(1.0 / (sim["timestep"] * sim["frame_skip"]), 6),
                "normalization": "scaled in environment",
            },
            "simulation": {
                "mjcf": "model.xml",
                "gravity": sim["gravity"],
                "timestep": sim["timestep"],
                "terrain": cfg["terrain"]["type"],
            },
        }
        if failed:
            entry["error"] = str(metrics["error"])
        return entry

    def add(self, generation: int, index: int, genotype: Genotype, metrics: dict, elite: bool) -> dict:
        """Archive one evaluation: robogene.json, a log row and the per-generation best."""
        entry = self.make_entry(generation, index, genotype, metrics)
        out = self.out_dir(generation, index)
        out.mkdir(parents=True, exist_ok=True)
        (out / "robogene.json").write_text(json.dumps(entry, indent=2))

        fit = entry["fitness"]
        row = [
            generation, index, entry["id"], entry["hash"], entry["morphology_class"],
            entry["modules"]["count"], entry["modules"]["arm"], entry["modules"]["wheel"],
            fit["value"], fit["loco"], fit["manip"], _finite(metrics.get("train_time_s")), int(bool(elite)),
        ]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(["" if v is None else v for v in row])

        self.entries.append(entry)
        if fit["value"] is not None:
            cur = self.best.get(generation)
            if cur is None or fit["value"] > cur["fitness"]["value"]:
                self.best[generation] = entry
                self._write_best()
        return entry

    def _write_best(self) -> None:
        with open(self.best_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(BEST_COLUMNS)
            for gen in sorted(self.best):
                e = self.best[gen]
                w.writerow([gen, e["id"], e["morphology_class"], e["fitness"]["value"]])

    def top_k(self, k: int) -> list[dict]:
        """The k best evaluated genotypes, unique by hash, each with its best evaluation."""
        best: dict[str, dict] = {}
        for e in self.entries:
            v = e["fitness"]["value"]
            if v is None:
                continue
            if e["hash"] not in best or v > best[e["hash"]]["fitness"]["value"]:
                best[e["hash"]] = e
        ranked = sorted(best.values(), key=lambda e: (-e["fitness"]["value"], e["generation"], e["index"]))
        return ranked[:k]

    def write_final(self, k: int) -> list[dict]:
        """Write final_top_k.json (entries with rank and path) and return its content."""
        final = []
        for rank, e in enumerate(self.top_k(k), start=1):
            final.append({"rank": rank, "path": f"individuals/{e['id']}", **e})
        (self.run_dir / "final_top_k.json").write_text(json.dumps(final, indent=2))
        return final


def load_run(run_dir: str | Path) -> pd.DataFrame:
    """The evolution log of a run as a DataFrame."""
    return pd.read_csv(Path(run_dir) / "evolution_log.csv")


def iter_entries(run_dir: str | Path) -> Iterator[tuple[dict, Path]]:
    """Yield (robogene.json content, individual directory) for every evaluation of a run."""
    for path in sorted((Path(run_dir) / "individuals").glob("*/robogene.json")):
        yield json.loads(path.read_text()), path.parent
