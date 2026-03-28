import json
import zlib

import numpy as np
import pytest

from robogene.archive import iter_entries, load_run
from robogene.config import load_config
from robogene.evolve import main, run_evolution
from robogene.genotype import Genotype, module_counts
from robogene.schema import default_library

ENTRY_KEYS = {
    "robogene_version", "type", "id", "generation", "index", "task", "genotype", "hash",
    "morphology_class", "modules", "fitness", "policy", "simulation",
}


def fake_evaluate(genotype: dict, cfg: dict, out_dir: str, seed: int) -> dict:
    """Wheel count plus a small noise fixed by the morphology, so re-evaluation is repeatable."""
    g = Genotype.from_dict(genotype)
    counts = module_counts(g, default_library())
    noise = (zlib.crc32(g.canonical_hash().encode()) % 1000) / 1e4
    fitness = counts["wheel"] + noise
    return {"fitness": fitness, "loco": fitness, "manip": float(counts["arm"]), "obs_dim": 10,
            "act_dim": 2, "train_time_s": 0.0, "policy": "policy.zip"}


def failing_evaluate(genotype: dict, cfg: dict, out_dir: str, seed: int) -> dict:
    if seed % 3 == 0:
        raise RuntimeError("simulated failure")
    return fake_evaluate(genotype, cfg, out_dir, seed)


def always(g) -> bool:
    return True


def small_cfg(path="configs/task1_locomotion.yaml", **ga):
    return load_config(path, {"ga": {"population": 6, "elites": 2, "generations": 3, **ga}})


def test_run_evolution_writes_archive(tmp_path):
    cfg = small_cfg()
    final = run_evolution(cfg, tmp_path, evaluate_fn=fake_evaluate, check_fn=always, workers=1)

    log = load_run(tmp_path)
    assert len(log) == 6 * 3
    assert (tmp_path / "config.yaml").exists()
    best = (tmp_path / "best_per_generation.csv").read_text().strip().splitlines()
    assert len(best) == 1 + 3

    entries = list(iter_entries(tmp_path))
    assert len(entries) == 18
    for entry, path in entries:
        assert ENTRY_KEYS <= set(entry)
        assert path.name == entry["id"]
        assert set(entry["modules"]) == {"count", "arm", "wheel", "body", "hinge", "connector"}
        assert entry["fitness"]["unit"] == "m"
        assert entry["policy"]["control_hz"] == 50
        assert entry["simulation"]["gravity"] == 1.62

    for gen in range(2):
        cur = log[log.generation == gen].sort_values("fitness", ascending=False)
        nxt = log[(log.generation == gen + 1) & (log.elite == 1)]
        assert len(nxt) == 2
        assert set(cur.hash.iloc[:2]) == set(nxt.hash)
    assert not log[log.generation == 0].elite.any()

    best_fit = log.groupby("generation").fitness.max().to_numpy()
    assert np.all(np.diff(best_fit) >= 0)

    saved = json.loads((tmp_path / "final_top_k.json").read_text())
    assert saved == final
    assert len(final) == 2 and len({e["hash"] for e in final}) == 2
    assert final[0]["fitness"]["value"] == pytest.approx(log.fitness.max())


def test_run_evolution_task2_and_failures(tmp_path):
    cfg = small_cfg("configs/task2_loco_manip.yaml")
    final = run_evolution(cfg, tmp_path, evaluate_fn=failing_evaluate, check_fn=always, workers=1)
    log = load_run(tmp_path)
    assert len(log) == 18
    assert log.fitness.isna().sum() > 0
    assert (log.arms >= 1).all() and (log.wheels >= 1).all()
    failed = [e for e, _ in iter_entries(tmp_path) if "error" in e]
    assert failed and all(e["fitness"]["value"] is None for e in failed)
    assert all(e["fitness"]["value"] is not None for e in final)


def test_run_evolution_parallel(tmp_path):
    cfg = small_cfg(generations=2)
    final = run_evolution(cfg, tmp_path, evaluate_fn=fake_evaluate, check_fn=always, workers=2)
    assert len(load_run(tmp_path)) == 12
    assert len(final) == 2


def test_run_evolution_is_reproducible(tmp_path):
    cfg = small_cfg(generations=2)
    run_evolution(cfg, tmp_path / "a", evaluate_fn=fake_evaluate, check_fn=always, workers=1)
    run_evolution(cfg, tmp_path / "b", evaluate_fn=fake_evaluate, check_fn=always, workers=1)
    assert list(load_run(tmp_path / "a").hash) == list(load_run(tmp_path / "b").hash)


def test_run_evolution_with_phenotype_check(tmp_path):
    from robogene import phenotype

    cfg = small_cfg(generations=2)
    try:
        phenotype.expand(Genotype.from_dict({"root": 0, "nodes": [{"id": 0, "module": "body_small"}], "edges": []}))
    except NotImplementedError:
        pytest.skip("phenotype.expand is not implemented")
    run_evolution(cfg, tmp_path, evaluate_fn=fake_evaluate, workers=1)
    assert len(load_run(tmp_path)) == 12


def test_cli(tmp_path, monkeypatch):
    import robogene.evolve as evolve

    calls = {}

    def fake_run(cfg, run_dir, **kwargs):
        calls["cfg"], calls["run_dir"] = cfg, run_dir
        return []

    monkeypatch.setattr(evolve, "run_evolution", fake_run)
    main(["--config", "configs/smoke.yaml", "--run-dir", str(tmp_path), "--run-name", "x",
          "--generations", "5", "--population", "8", "--budget", "1000", "--seed", "3", "--workers", "2"])
    cfg = calls["cfg"]
    assert calls["run_dir"] == tmp_path / "x"
    assert cfg["ga"]["generations"] == 5 and cfg["ga"]["population"] == 8
    assert cfg["rl"]["budget"] == 1000 and cfg["seed"] == 3 and cfg["parallel"]["workers"] == 2
