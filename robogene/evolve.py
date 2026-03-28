"""Morphology evolution: a genetic algorithm over genotypes with every candidate trained by RL."""

from __future__ import annotations

import argparse
import math
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import ga
from .archive import Archive
from .config import load_config
from .genotype import Genotype
from .schema import default_library

EvaluateFn = Callable[[dict, dict, str, int], dict]


def evaluation_seed(cfg: dict, generation: int, index: int) -> int:
    return int(cfg["seed"]) * 100003 + generation * 1000 + index


def safe_evaluate(evaluate_fn: EvaluateFn | None, genotype: dict, cfg: dict, out_dir: str, seed: int) -> dict:
    """Run one evaluation and turn an exception into an "error" entry."""
    try:
        if evaluate_fn is None:
            from .rl import evaluate_individual as evaluate_fn
        return dict(evaluate_fn(genotype, cfg, out_dir, seed))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def ga_fitness(metrics: dict) -> float:
    """Fitness used for selection; failed or non-finite evaluations rank last."""
    if "error" in metrics:
        return -math.inf
    try:
        value = float(metrics.get("fitness"))
    except (TypeError, ValueError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf


def initial_population(library, cfg: dict, rng: np.random.Generator, check_fn: ga.CheckFn) -> list[Genotype]:
    """M random genotypes that pass the genotype and phenotype checks."""
    population: list[Genotype] = []
    while len(population) < int(cfg["ga"]["population"]):
        g = ga.random_genotype(library, cfg, rng)
        if check_fn(g):
            population.append(g)
    return population


def run_evolution(
    cfg: dict,
    run_dir: str | Path,
    evaluate_fn: EvaluateFn | None = None,
    check_fn: ga.CheckFn | None = None,
    workers: int | None = None,
) -> list[dict]:
    """Evolve for ga.generations generations and return the final top K entries."""
    library = default_library()
    ga_cfg = cfg["ga"]
    M, K, G = int(ga_cfg["population"]), int(ga_cfg["elites"]), int(ga_cfg["generations"])
    if not 0 <= K <= M:
        raise ValueError("ga.elites must be between 0 and ga.population")
    workers = int(workers if workers is not None else cfg["parallel"]["workers"])
    if check_fn is None:
        def check_fn(g: Genotype) -> bool:
            return ga.phenotype_check(g, cfg, library)

    rng = np.random.default_rng(int(cfg["seed"]))
    archive = Archive(run_dir, cfg, library)
    population = initial_population(library, cfg, rng, check_fn)
    elite_flags = [False] * M
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        for gen in range(G):
            t0 = time.time()
            jobs = [
                (evaluate_fn, g.to_dict(), cfg, str(archive.out_dir(gen, i)), evaluation_seed(cfg, gen, i))
                for i, g in enumerate(population)
            ]
            if pool is None:
                results = [safe_evaluate(*job) for job in jobs]
            else:
                results = list(pool.map(safe_evaluate, *zip(*jobs)))

            fitness = []
            for i, (g, metrics) in enumerate(zip(population, results)):
                entry = archive.add(gen, i, g, metrics, elite_flags[i])
                fitness.append(ga_fitness(metrics))
                if "error" in metrics:
                    print(f"  {entry['id']} failed: {metrics['error']}")

            order = sorted(range(M), key=lambda i: (-fitness[i], i))
            finite = [f for f in fitness if math.isfinite(f)]
            best_class = archive.best[gen]["morphology_class"] if gen in archive.best else "none"
            print(
                f"gen {gen:3d}  best {max(finite, default=float('nan')):.3f}  "
                f"mean {np.mean(finite) if finite else float('nan'):.3f}  class {best_class}  "
                f"failed {M - len(finite)}  {time.time() - t0:.1f} s",
                flush=True,
            )
            if gen == G - 1:
                break
            elites = [population[i].copy() for i in order[:K]]
            offspring = [ga.make_offspring(population, fitness, library, cfg, rng, check_fn) for _ in range(M - K)]
            population = elites + offspring
            elite_flags = [True] * K + [False] * (M - K)
    finally:
        if pool is not None:
            pool.shutdown()
    return archive.write_final(K)


def main(argv: list[str] | None = None) -> list[dict]:
    """Command line entry point."""
    p = argparse.ArgumentParser(description="Evolve modular robot morphologies with a GA and PPO.")
    p.add_argument("--config", required=True, help="YAML experiment config")
    p.add_argument("--run-name", help="run folder name (default: task and timestamp)")
    p.add_argument("--run-dir", help="parent folder of runs (default: run_dir in the config)")
    p.add_argument("--workers", type=int, help="candidates trained in parallel")
    p.add_argument("--generations", type=int, help="override ga.generations")
    p.add_argument("--population", type=int, help="override ga.population")
    p.add_argument("--budget", type=int, help="override rl.budget")
    p.add_argument("--seed", type=int, help="override seed")
    args = p.parse_args(argv)

    overrides: dict = {}
    if args.generations is not None:
        overrides.setdefault("ga", {})["generations"] = args.generations
    if args.population is not None:
        overrides.setdefault("ga", {})["population"] = args.population
    if args.budget is not None:
        overrides["rl"] = {"budget": args.budget}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.workers is not None:
        overrides["parallel"] = {"workers": args.workers}
    cfg = load_config(args.config, overrides)

    name = args.run_name or f"{cfg['task']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.run_dir or cfg["run_dir"]) / name
    print(f"run directory {run_dir}")
    final = run_evolution(cfg, run_dir)
    for e in final:
        print(f"#{e['rank']}  {e['id']}  {e['morphology_class']}  fitness {e['fitness']['value']:.3f}")
    return final


if __name__ == "__main__":
    main()
