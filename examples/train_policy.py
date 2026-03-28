"""Train a PPO policy for one morphology and measure its fitness."""

import argparse
from pathlib import Path

from build_phenotype import rover
from robogene.config import load_config
from robogene.phenotype import expand
from robogene.rl import evaluate_policy, train_policy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("locomotion", "loco_manip"), default="locomotion")
    p.add_argument("--budget", type=int, default=20000, help="environment steps")
    p.add_argument("--out", default="runs/examples/policy")
    a = p.parse_args()

    cfg = load_config(overrides={"task": a.task, "rl": {"budget": a.budget}})
    ph = expand(rover())
    model = train_policy(ph, cfg, seed=0)
    result = evaluate_policy(model, ph, cfg)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "policy.zip")
    print(f"{a.task}: fitness {result['fitness']:.3f} m (loco {result['loco']:.3f}, manip {result['manip']:.3f}) "
          f"over {result['episodes']} episodes, policy in {out}")


if __name__ == "__main__":
    main()
