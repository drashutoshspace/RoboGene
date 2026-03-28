import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from robogene import plots

CLASSES = ["Arm Only", "Legged", "2Arm+2Wheel", "3Arm+4Wheel", "1Arm+1Wheel"]


@pytest.fixture
def log_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for gen in range(6):
        for idx in range(5):
            h = f"h{rng.integers(0, 20)}"
            loco = float(rng.uniform(0, 10))
            rows.append({"generation": gen, "index": idx, "id": f"g{gen:03d}_i{idx:02d}", "hash": h,
                         "morphology_class": CLASSES[int(h[1:]) % len(CLASSES)], "fitness": loco,
                         "loco": loco, "manip": float(rng.uniform(0, 3)), "elite": idx == 0})
    return pd.DataFrame(rows)


@pytest.fixture
def terrain_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 14
    flat = rng.uniform(0, 10, n)
    flat[0] = 0.0
    hilly = flat * rng.uniform(0.3, 1.1, n)
    ratio = np.where(flat > 1e-6, hilly / np.where(flat > 0, flat, 1), np.nan)
    return pd.DataFrame({"id": [f"g000_i{i:02d}" for i in range(n)], "run": "r", "hash": [f"h{i}" for i in range(n)],
                         "morphology_class": [CLASSES[i % len(CLASSES)] for i in range(n)],
                         "flat_fitness": flat, "hilly_fitness": hilly, "ratio": ratio})


def _exists(paths):
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_class_color_fixed():
    assert plots.class_color("Legged") == plots.CLASS_COLORS["Legged"]
    assert plots.class_color("5Arm+1Wheel") == plots.class_color("5Arm+1Wheel")
    assert plots.class_order(["X", "Legged", "Arm Only"]) == ["Arm Only", "Legged", "X"]


def test_plot_functions(tmp_path, log_df, terrain_df):
    _exists(plots.plot_generations(log_df, tmp_path / "gen"))
    _exists(plots.plot_boxplot(log_df, tmp_path / "box.png"))
    _exists(plots.plot_tradeoff(log_df, tmp_path / "trade", loco_threshold=1.0))
    _exists(plots.plot_terrain(terrain_df, tmp_path / "terrain"))


def test_unique_morphologies_keeps_best(log_df):
    u = plots.unique_morphologies(log_df)
    assert u["hash"].is_unique
    for h, best in log_df.groupby("hash")["fitness"].max().items():
        assert u.loc[u["hash"] == h, "fitness"].item() == best


def test_cli(tmp_path, log_df, terrain_df):
    run = tmp_path / "run"
    run.mkdir()
    log_df.to_csv(run / "evolution_log.csv", index=False)
    terrain_df.to_csv(tmp_path / "terrain.csv", index=False)
    for cmd in ("generations", "boxplot", "tradeoff"):
        plots.main([cmd, "--run", str(run)])
        assert (run / "figures" / f"{cmd}.png").exists()
        assert (run / "figures" / f"{cmd}.pdf").exists()
    plots.main(["terrain", "--csv", str(tmp_path / "terrain.csv"), "--out", str(tmp_path / "t")])
    assert (tmp_path / "t.png").exists()
    plots.main(["all", "--run", str(run), "--terrain-csv", str(tmp_path / "terrain.csv"), "--out", str(tmp_path / "all")])
    for name in ("generations", "boxplot", "tradeoff", "terrain"):
        assert (tmp_path / "all" / f"{name}.pdf").exists()
