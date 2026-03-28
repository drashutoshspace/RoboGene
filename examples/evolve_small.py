"""Run a small morphology evolution and draw its figures."""

import argparse
from pathlib import Path

from robogene import plots
from robogene.config import load_config
from robogene.evolve import run_evolution


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--population", type=int, default=6)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--run", default="runs/examples/evolution")
    a = p.parse_args()

    cfg = load_config(a.config, overrides={
        "ga": {"generations": a.generations, "population": a.population, "elites": max(1, a.population // 4)},
        "rl": {"budget": a.budget},
    })
    top = run_evolution(cfg, Path(a.run), workers=a.workers)
    plots.main(["all", "--run", a.run])
    for e in top:
        print(f"#{e['rank']} {e['id']} {e['morphology_class']} fitness {e['fitness']['value']:.3f} m")


if __name__ == "__main__":
    main()
