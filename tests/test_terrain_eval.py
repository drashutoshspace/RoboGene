import json
import math
import sys
import types

import pandas as pd
import yaml

from robogene import terrain_eval
from robogene.config import DEFAULT_CONFIG, save_config


def _write_run(root, name, entries):
    run = root / name
    run.mkdir()
    cfg = dict(DEFAULT_CONFIG)
    cfg["task"] = "loco_manip"
    save_config(cfg, run / "config.yaml")
    for e in entries:
        d = run / "individuals" / e["id"]
        d.mkdir(parents=True)
        entry = {"id": e["id"], "hash": e["hash"], "morphology_class": e["cls"],
                 "genotype": {"root": 0, "nodes": [{"id": 0, "module": "body_small"}], "edges": []},
                 "fitness": {"value": e["loco"], "loco": e["loco"], "manip": 0.0},
                 "policy": {"checkpoint": "policy.zip"}}
        (d / "robogene.json").write_text(json.dumps(entry))
    return run


def test_terrain_eval(tmp_path, monkeypatch):
    run_a = _write_run(tmp_path, "a", [
        {"id": "g000_i00", "hash": "h1", "cls": "Legged", "loco": 4.0},
        {"id": "g001_i00", "hash": "h1", "cls": "Legged", "loco": 6.0},
        {"id": "g000_i01", "hash": "h2", "cls": "Arm Only", "loco": 0.0},
    ])
    run_b = _write_run(tmp_path, "b", [
        {"id": "g000_i00", "hash": "h1", "cls": "Legged", "loco": 5.0},
        {"id": "g002_i03", "hash": "h3", "cls": "2Arm+2Wheel", "loco": 10.0},
    ])
    terrain = tmp_path / "terrain.yaml"
    terrain.write_text(yaml.safe_dump({"task": "locomotion", "terrain": {"type": "hilly"}}))

    seen = []

    def fake_load_policy(path):
        return path

    def fake_evaluate(model, ph, cfg, episodes=None, seed=None):
        seen.append((model, cfg, episodes))
        return {"loco": 3.0 if "h3" not in model else 5.0, "manip": 0.0, "fitness": 0.0}

    import robogene.phenotype as phenotype

    rl = types.ModuleType("robogene.rl")
    rl.load_policy = fake_load_policy
    rl.evaluate_policy = fake_evaluate
    monkeypatch.setitem(sys.modules, "robogene.rl", rl)
    monkeypatch.setattr(phenotype, "expand", lambda g, library=None: g)

    out = tmp_path / "terrain.csv"
    terrain_eval.run_terrain_eval([run_a, run_b], terrain, out, episodes=2)
    df = pd.read_csv(out)
    assert list(df.columns) == terrain_eval.COLUMNS
    assert sorted(df["hash"]) == ["h1", "h2", "h3"]
    h1 = df[df["hash"] == "h1"].iloc[0]
    assert h1["id"] == "g001_i00" and h1["run"] == "a" and h1["flat_fitness"] == 6.0
    assert h1["ratio"] == 0.5
    assert math.isnan(df[df["hash"] == "h2"]["ratio"].iloc[0])
    for _, cfg, episodes in seen:
        assert cfg["terrain"]["type"] == "hilly"
        assert cfg["task"] == "loco_manip"  # the run's task is kept
        assert episodes == 2
    assert "3 morphologies" in terrain_eval.summary(df)
