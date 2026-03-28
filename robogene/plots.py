"""Figures of RoboGene runs, written as PNG (300 dpi) and PDF."""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DPI = 300

CLASS_COLORS: dict[str, str] = {
    "Arm Only": "#1f77b4",
    "Legged": "#ff7f0e",
    "2Arm+2Wheel": "#2ca02c",
    "2Arm+3Wheel": "#d62728",
    "3Arm+2Wheel": "#9467bd",
    "3Arm+4Wheel": "#8c564b",
    "Body Only": "#7f7f7f",
    "4Wheel": "#17becf",
}
_FALLBACK = ["#e377c2", "#bcbd22", "#393b79", "#637939", "#7b4173", "#e7ba52", "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"]


def class_color(name: str) -> str:
    """Color of a morphology class, the same in every figure and every run."""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    # crc32 rather than hash(): Python string hashes change between processes
    return _FALLBACK[zlib.crc32(str(name).encode()) % len(_FALLBACK)]


def class_order(names) -> list[str]:
    """Known classes in palette order, then the others alphabetically."""
    present = set(names)
    known = [c for c in CLASS_COLORS if c in present]
    return known + sorted(present - set(known))


def save_figure(fig: Figure, out: str | Path) -> list[Path]:
    """Save fig as out.png (300 dpi) and out.pdf; out may carry any suffix or none."""
    out = Path(out)
    if out.suffix.lower() in (".png", ".pdf"):
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_name(out.name + ".png"), out.with_name(out.name + ".pdf")]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def unique_morphologies(df: pd.DataFrame, by: str = "fitness") -> pd.DataFrame:
    """One row per morphology hash (its best evaluation), elites appear once."""
    key = "hash" if "hash" in df.columns else "id"
    return df.sort_values(by, ascending=False).drop_duplicates(key).reset_index(drop=True)


def _class_legend(ax, classes, title: str = "Morphology Class", **kw) -> None:
    handles = [Patch(facecolor=class_color(c), label=c) for c in classes]
    ax.legend(handles=handles, title=title, fontsize=8, title_fontsize=9, **kw)


def plot_generations(df: pd.DataFrame, out: str | Path) -> list[Path]:
    """Best fitness of every generation, each bar colored by the class of that best individual."""
    best = df.sort_values("fitness", ascending=False).drop_duplicates("generation").sort_values("generation")
    fig = Figure(figsize=(10, 4.5))
    ax = fig.subplots()
    colors = [class_color(c) for c in best["morphology_class"]]
    ax.bar(best["generation"], best["fitness"], color=colors, width=0.8, edgecolor="none")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (m)")
    ax.set_title("Best Fitness per Generation")
    ax.grid(axis="y", alpha=0.3)
    _class_legend(ax, class_order(best["morphology_class"]), loc="upper left")
    return save_figure(fig, out)


def plot_boxplot(df: pd.DataFrame, out: str | Path) -> list[Path]:
    """Fitness distribution by morphology class over all unique evaluated morphologies."""
    u = unique_morphologies(df)
    classes = class_order(u["morphology_class"])
    data = [u.loc[u["morphology_class"] == c, "fitness"].to_numpy() for c in classes]
    fig = Figure(figsize=(max(6.0, 1.3 * len(classes) + 2), 4.8))
    ax = fig.subplots()
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        medianprops={"color": "navy", "linewidth": 2},
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(class_color(c))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels([f"{c}\nn={len(d)}" for c, d in zip(classes, data)], fontsize=8)
    ax.set_xlabel("Morphology Class")
    ax.set_ylabel("Fitness (m)")
    ax.set_title("Fitness Distribution by Morphology Class")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Line2D([0], [0], color="navy", linewidth=2, label="Median"),
            Line2D([0], [0], color="none", marker="D", markerfacecolor="red", markeredgecolor="red", label="Mean"),
        ],
        loc="best",
        fontsize=8,
    )
    return save_figure(fig, out)


def plot_tradeoff(
    df: pd.DataFrame,
    out: str | Path,
    loco_threshold: float | None = None,
    manip_threshold: float | None = None,
) -> list[Path]:
    """Locomotion versus manipulation fitness per unique morphology, with quadrants."""
    u = unique_morphologies(df)
    loco_max = max(float(u["loco"].max()), 0.0)
    manip_max = max(float(u["manip"].max()), 0.0)
    lt = 0.1 * loco_max if loco_threshold is None else loco_threshold
    mt = 0.1 * manip_max if manip_threshold is None else manip_threshold

    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    classes = class_order(u["morphology_class"])
    for c in classes:
        s = u[u["morphology_class"] == c]
        ax.scatter(s["loco"], s["manip"], color=class_color(c), label=c, s=30, alpha=0.8, edgecolors="k", linewidths=0.3)
    ax.axvline(lt, color="gray", linestyle="--", linewidth=1)
    ax.axhline(mt, color="gray", linestyle="--", linewidth=1)
    top = max(loco_max, manip_max, 1e-6)
    ax.plot([0, top], [0, top], color="green", linestyle="--", linewidth=1.2, label="Perfect Balance")

    box = {"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
    for x, y, ha, va, text in (
        (0.02, 0.98, "left", "top", "High Manipulation / Low Locomotion"),
        (0.98, 0.98, "right", "top", "High Both / Ideal"),
        (0.02, 0.02, "left", "bottom", "Low Both (Ineffective)"),
        (0.98, 0.02, "right", "bottom", "High Locomotion / Low Manipulation"),
    ):
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8, bbox=box)

    ax.set_xlabel("Locomotion Fitness (m)")
    ax.set_ylabel("Manipulation Fitness (m)")
    ax.set_title("Locomotion versus Manipulation Fitness")
    ax.grid(alpha=0.3)
    ax.legend(title="Morphology Class", fontsize=8, title_fontsize=9, loc="center right")
    return save_figure(fig, out)


def plot_terrain(df: pd.DataFrame, out: str | Path) -> list[Path]:
    """Flat versus hilly terrain in four panels, from a terrain_eval CSV."""
    df = df.copy()
    classes = class_order(df["morphology_class"])
    fig = Figure(figsize=(13, 10))
    (ax_a, ax_b), (ax_c, ax_d) = fig.subplots(2, 2)
    hatch = "///"

    for c in classes:
        s = df[df["morphology_class"] == c]
        ax_a.scatter(s["flat_fitness"], s["hilly_fitness"], color=class_color(c), label=c, s=30, alpha=0.8,
                     edgecolors="k", linewidths=0.3)
    lo = min(0.0, float(df[["flat_fitness", "hilly_fitness"]].min().min()))
    hi = max(float(df[["flat_fitness", "hilly_fitness"]].max().max()), 1e-6)
    ax_a.plot([lo, hi], [lo, hi], color="gray", linestyle="--", label="Equal performance")
    ax_a.set_xlabel("Flat Terrain Fitness (m)")
    ax_a.set_ylabel("Hilly Terrain Fitness (m)")
    ax_a.set_title("(a) Flat versus Hilly Terrain")
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=7, title="Morphology Class", title_fontsize=8)

    means = df.groupby("morphology_class")[["flat_fitness", "hilly_fitness"]].mean().reindex(classes)
    x = np.arange(len(classes))
    w = 0.38
    cols = [class_color(c) for c in classes]
    ax_b.bar(x - w / 2, means["flat_fitness"], w, color=cols, edgecolor="k", linewidth=0.5)
    ax_b.bar(x + w / 2, means["hilly_fitness"], w, color=cols, edgecolor="k", linewidth=0.5, hatch=hatch, alpha=0.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax_b.set_xlabel("Morphology Class")
    ax_b.set_ylabel("Mean Fitness (m)")
    ax_b.set_title("(b) Mean Fitness by Morphology Class")
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.legend(handles=_terrain_handles(hatch), fontsize=8)

    ratio = pd.to_numeric(df["ratio"], errors="coerce").to_numpy(dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio):
        ax_c.hist(ratio, bins=min(20, max(5, len(ratio) // 2)), color="#999999", edgecolor="k", linewidth=0.5)
        ax_c.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Equal (1.0)")
        ax_c.axvline(ratio.mean(), color="blue", linewidth=1.5, label=f"Mean ({ratio.mean():.2f})")
        ax_c.legend(fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "no finite ratios", transform=ax_c.transAxes, ha="center")
    ax_c.set_xlabel("Hilly / Flat Fitness Ratio")
    ax_c.set_ylabel("Number of Morphologies")
    ax_c.set_title("(c) Performance Ratio")
    ax_c.grid(axis="y", alpha=0.3)

    top = df.sort_values("flat_fitness", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(top))
    h = 0.38
    cols = [class_color(c) for c in top["morphology_class"]]
    ax_d.barh(y + h / 2, top["flat_fitness"], h, color=cols, edgecolor="k", linewidth=0.5)
    ax_d.barh(y - h / 2, top["hilly_fitness"], h, color=cols, edgecolor="k", linewidth=0.5, hatch=hatch, alpha=0.6)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([f"{i} ({c})" for i, c in zip(top["id"], top["morphology_class"])], fontsize=7)
    ax_d.set_xlabel("Fitness (m)")
    ax_d.set_title("(d) Top 10 Morphologies on Flat Terrain")
    ax_d.grid(axis="x", alpha=0.3)
    ax_d.legend(handles=_terrain_handles(hatch), fontsize=8, loc="lower right")

    fig.tight_layout()
    return save_figure(fig, out)


def _terrain_handles(hatch: str) -> list[Patch]:
    return [
        Patch(facecolor="#bbbbbb", edgecolor="k", label="Flat"),
        Patch(facecolor="#bbbbbb", edgecolor="k", hatch=hatch, alpha=0.6, label="Hilly"),
    ]


def read_log(run: str | Path) -> pd.DataFrame:
    path = Path(run) / "evolution_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)


def _out(args, run: str | Path | None, name: str) -> Path:
    if args.out:
        return Path(args.out)
    return Path(run) / "figures" / name


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m robogene.plots", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("generations", "boxplot"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--out", help="output path without suffix (default RUN/figures/<command>)")
    s = sub.add_parser("tradeoff")
    s.add_argument("--run", required=True)
    s.add_argument("--out")
    s.add_argument("--loco-threshold", type=float)
    s.add_argument("--manip-threshold", type=float)
    s = sub.add_parser("terrain")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", help="default: <csv folder>/figures/terrain")
    s = sub.add_parser("all")
    s.add_argument("--run", required=True)
    s.add_argument("--terrain-csv")
    s.add_argument("--out", help="output folder (default RUN/figures)")
    args = p.parse_args(argv)

    written: list[Path] = []
    if args.command == "terrain":
        out = Path(args.out) if args.out else Path(args.csv).parent / "figures" / "terrain"
        written += plot_terrain(pd.read_csv(args.csv), out)
    elif args.command == "all":
        folder = Path(args.out) if args.out else Path(args.run) / "figures"
        df = read_log(args.run)
        written += plot_generations(df, folder / "generations")
        written += plot_boxplot(df, folder / "boxplot")
        if {"loco", "manip"} <= set(df.columns):
            written += plot_tradeoff(df, folder / "tradeoff")
        if args.terrain_csv:
            written += plot_terrain(pd.read_csv(args.terrain_csv), folder / "terrain")
    else:
        df = read_log(args.run)
        out = _out(args, args.run, args.command)
        if args.command == "generations":
            written += plot_generations(df, out)
        elif args.command == "boxplot":
            written += plot_boxplot(df, out)
        else:
            written += plot_tradeoff(df, out, args.loco_threshold, args.manip_threshold)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
