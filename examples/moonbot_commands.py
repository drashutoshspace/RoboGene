"""Train the four-wheel rover, export MoonBot commands and preview their replay."""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

from build_phenotype import rover
from robogene.archive import Archive
from robogene.config import load_config
from robogene.rl import evaluate_individual
from robogene.sim2real import run_sim2real

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, default=20000, help="environment steps")
    p.add_argument("--run", default="runs/examples/moonbot")
    a = p.parse_args()

    cfg = load_config(overrides={"rl": {"budget": a.budget}})
    run = Path(a.run)
    archive = Archive(run, cfg)
    g = rover()
    metrics = evaluate_individual(g.to_dict(), cfg, str(archive.out_dir(0, 0)), seed=0)
    entry = archive.add(0, 0, g, metrics, elite=False)
    print(f"trained {entry['id']} ({entry['morphology_class']}), fitness {metrics['fitness']:.3f} m")

    commands = run / "commands.csv"
    table, _ = run_sim2real(archive.out_dir(0, 0) / "robogene.json", ROOT / "configs/sim2real_vehicle.yaml", out=commands)
    print(table.head())

    sys.path.insert(0, str(ROOT / "moonbot"))
    from replay_commands import replay

    # preview with the wheel channels only, the limb joint names are set per robot
    mapping = yaml.safe_load((ROOT / "moonbot/replay_vehicle.yaml").read_text())
    mapping["channels"] = [c for c in mapping["channels"] if not c["joint"].startswith("SET_")]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(mapping, f)
    replay(str(commands), f.name, dry_run=True)


if __name__ == "__main__":
    main()
