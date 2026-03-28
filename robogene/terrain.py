"""Ground models: a flat plane or a hilly height field."""

from __future__ import annotations

import numpy as np

PLANE_SIZE = "200 200 0.1"
BASE_THICKNESS = 0.1

_LAYOUT_CACHE: dict[tuple, dict] = {}


def hilly_heights(terrain_cfg: dict) -> np.ndarray:
    """(ny, nx) normalized heights in [0, 1], row 0 at the lowest y."""
    h = terrain_cfg["hilly"]
    rng = np.random.RandomState(h["seed"])
    ref = np.clip(rng.normal(h["mean"], h["std"], size=(h["nx"], h["ny"])), 0.0, 1.0)
    return ref.T.copy()


def _field_layout(terrain_cfg: dict) -> dict:
    key = tuple(sorted((k, float(v)) for k, v in terrain_cfg["hilly"].items()))
    if key not in _LAYOUT_CACHE:
        h = terrain_cfg["hilly"]
        heights = hilly_heights(terrain_cfg)
        hmin, span = float(heights.min()), float(heights.max() - heights.min())
        max_height = float(h["max_height"])
        _LAYOUT_CACHE[key] = {
            "heights": heights,
            "hmin": hmin,
            "half_x": 0.5 * float(h["size_x"]),
            "half_y": 0.5 * float(h["size_y"]),
            "center_x": float(h["center_x"]),
            # MuJoCo rescales the elevation data to [0, 1] by its min and max
            "size_z": max_height * span if span > 1e-12 else max_height,
            "base_z": (hmin - float(h["mean"])) * max_height,
            "max_height": max_height,
        }
    return _LAYOUT_CACHE[key]


def terrain_assets_xml(terrain_cfg: dict) -> str:
    """<asset> children: the ground material and, for hilly terrain, the height field."""
    out = [
        '<texture name="grid" type="2d" builtin="checker" rgb1="0.55 0.53 0.5" '
        'rgb2="0.45 0.43 0.4" width="256" height="256"/>',
        '<material name="ground_mat" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0"/>',
    ]
    if terrain_cfg["type"] == "hilly":
        lay = _field_layout(terrain_cfg)
        ny, nx = lay["heights"].shape
        # the elevation attribute lists the row at +y first
        elevation = " ".join(f"{v:.8g}" for v in lay["heights"][::-1].ravel())
        out.append(
            f'<hfield name="terrain" nrow="{ny}" ncol="{nx}" '
            f'size="{lay["half_x"]:.8g} {lay["half_y"]:.8g} {lay["size_z"]:.8g} {BASE_THICKNESS:.8g}" '
            f'elevation="{elevation}"/>'
        )
    return "\n".join(out)


def terrain_geom_xml(terrain_cfg: dict) -> str:
    """The ground geoms: "ground", plus "ground_base" around a height field."""
    friction = f"{float(terrain_cfg.get('friction', 0.5)):.6g} 0.005 0.0001"
    # priority makes MuJoCo use the ground friction instead of the maximum of both geoms
    common = f'friction="{friction}" priority="1" condim="3" material="ground_mat"'
    if terrain_cfg["type"] == "flat":
        return f'<geom name="ground" type="plane" size="{PLANE_SIZE}" pos="0 0 0" {common}/>'
    if terrain_cfg["type"] != "hilly":
        raise ValueError(f"unknown terrain {terrain_cfg['type']!r}")
    lay = _field_layout(terrain_cfg)
    z = lay["base_z"]
    return "\n".join(
        [
            f'<geom name="ground" type="hfield" hfield="terrain" pos="{lay["center_x"]:.8g} 0 {z:.8g}" {common}/>',
            f'<geom name="ground_base" type="plane" size="{PLANE_SIZE}" pos="0 0 {z:.8g}" {common}/>',
        ]
    )


def ground_height(terrain_cfg: dict, x: float, y: float) -> float:
    """World z of the ground surface at (x, y)."""
    if terrain_cfg["type"] == "flat":
        return 0.0
    lay = _field_layout(terrain_cfg)
    return _height_uv(lay, *_to_grid(lay, x, y))


def _to_grid(lay: dict, x: float, y: float) -> tuple[float, float]:
    ny, nx = lay["heights"].shape
    u = (x - lay["center_x"] + lay["half_x"]) / (2 * lay["half_x"]) * (nx - 1)
    v = (y + lay["half_y"]) / (2 * lay["half_y"]) * (ny - 1)
    return u, v


def _height_uv(lay: dict, u: float, v: float) -> float:
    heights = lay["heights"]
    ny, nx = heights.shape
    if not (0.0 <= u <= nx - 1 and 0.0 <= v <= ny - 1):
        return lay["base_z"]
    c = min(int(np.floor(u)), nx - 2)
    r = min(int(np.floor(v)), ny - 2)
    fx, fy = u - c, v - r
    z00, z10 = heights[r, c], heights[r, c + 1]
    z01, z11 = heights[r + 1, c], heights[r + 1, c + 1]
    # MuJoCo splits each cell along the diagonal from (c, r) to (c + 1, r + 1)
    if fx >= fy:
        h = z00 + fx * (z10 - z00) + fy * (z11 - z10)
    else:
        h = z00 + fy * (z01 - z00) + fx * (z11 - z01)
    return lay["base_z"] + (float(h) - lay["hmin"]) * lay["max_height"]


def max_ground_height(terrain_cfg: dict, x_range: tuple[float, float], y_range: tuple[float, float]) -> float:
    """Highest ground point over the rectangle x_range x y_range."""
    if terrain_cfg["type"] == "flat":
        return 0.0
    lay = _field_layout(terrain_cfg)
    heights = lay["heights"]
    ny, nx = heights.shape
    u0, v0 = _to_grid(lay, min(x_range), min(y_range))
    u1, v1 = _to_grid(lay, max(x_range), max(y_range))
    best = lay["base_z"]
    # piecewise linear surface: the maximum is at an inner grid vertex or a boundary breakpoint
    iu = np.arange(max(0, int(np.ceil(u0))), min(nx - 1, int(np.floor(u1))) + 1)
    iv = np.arange(max(0, int(np.ceil(v0))), min(ny - 1, int(np.floor(v1))) + 1)
    if iu.size and iv.size:
        inner = float(heights[np.ix_(iv, iu)].max())
        best = max(best, lay["base_z"] + (inner - lay["hmin"]) * lay["max_height"])
    us = [u0, u1, *range(int(np.ceil(u0)), int(np.floor(u1)) + 1)]
    vs = [v0, v1, *range(int(np.ceil(v0)), int(np.floor(v1)) + 1)]
    for v in (v0, v1):
        diagonals = [v + k for k in range(int(np.ceil(u0 - v)), int(np.floor(u1 - v)) + 1)]
        best = max([best] + [_height_uv(lay, u, v) for u in us + diagonals])
    for u in (u0, u1):
        diagonals = [u - k for k in range(int(np.ceil(u - v1)), int(np.floor(u - v0)) + 1)]
        best = max([best] + [_height_uv(lay, u, v) for v in vs + diagonals])
    return best
