"""RoboGene entries for modules, assemblies and hardware modules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
SHAPES = ("box", "cylinder", "capsule")
JOINTS = ("fixed", "hinge", "continuous")


@dataclass(frozen=True)
class Port:
    id: str
    position: np.ndarray
    normal: np.ndarray  # outward unit normal, module frame
    roll_options_deg: tuple[float, ...]


@dataclass
class ModuleEntry:
    name: str
    category: str
    description: str
    geometry: dict[str, Any]
    actuation: dict[str, Any]
    sensing: list[dict[str, Any]]
    ports: dict[str, Port]
    policy: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def actuated(self) -> bool:
        return self.actuation.get("joint", "fixed") != "fixed"

    @property
    def shape(self) -> str:
        return self.geometry["shape"]

    @property
    def size(self) -> list[float]:
        return list(self.geometry["size"])

    def volume(self) -> float:
        return shape_volume(self.shape, self.size)

    def mass(self) -> float:
        return self.geometry["density"] * self.volume()

    def inertia(self) -> np.ndarray:
        """Principal moments of inertia about the center of mass."""
        return shape_inertia(self.shape, self.size, self.mass())

    def bounding_radius(self) -> float:
        s = self.size
        if self.shape == "box":
            return float(np.linalg.norm(s))
        if self.shape == "cylinder":
            return float(np.hypot(s[0], s[1]))
        return float(s[0] + s[1])  # capsule


@dataclass
class AssemblyEntry:
    name: str
    category: str
    description: str
    attach_port: str
    genotype: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class HardwareModule:
    name: str
    category: str
    description: str
    joints: list[dict[str, Any]]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def shape_volume(shape: str, size: list[float]) -> float:
    if shape == "box":
        hx, hy, hz = size
        return 8.0 * hx * hy * hz
    if shape == "cylinder":
        r, h = size
        return np.pi * r * r * 2.0 * h
    if shape == "capsule":
        r, h = size
        return np.pi * r * r * 2.0 * h + 4.0 / 3.0 * np.pi * r**3
    raise ValueError(f"unknown shape {shape}")


def shape_inertia(shape: str, size: list[float], mass: float) -> np.ndarray:
    """Diagonal inertia of a solid shape; cylinders and capsules lie along z."""
    if shape == "box":
        hx, hy, hz = size
        a, b, c = 2 * hx, 2 * hy, 2 * hz
        return mass / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    if shape == "cylinder":
        r, h = size
        length = 2 * h
        ixx = mass * (3 * r * r + length * length) / 12.0
        return np.array([ixx, ixx, 0.5 * mass * r * r])
    if shape == "capsule":
        r, h = size
        vc = np.pi * r * r * 2 * h
        vs = 4.0 / 3.0 * np.pi * r**3
        rho = mass / (vc + vs)
        mc, ms = rho * vc, rho * vs
        length = 2 * h
        izz = 0.5 * mc * r * r + 0.4 * ms * r * r
        # hemisphere centroids sit h + 3r/8 from the center
        ixx_c = mc * (3 * r * r + length * length) / 12.0
        d = h + 3.0 * r / 8.0
        ixx_s = 0.4 * ms * r * r - ms * (3.0 * r / 8.0) ** 2 + ms * d * d
        ixx = ixx_c + ixx_s
        return np.array([ixx, ixx, izz])
    raise ValueError(f"unknown shape {shape}")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length normal")
    return v / n


def parse_module(data: dict[str, Any]) -> ModuleEntry:
    if data.get("type") != "module":
        raise ValueError(f"{data.get('name')}: not a module entry")
    geom = data["geometry"]
    if geom["shape"] not in SHAPES:
        raise ValueError(f"{data['name']}: unknown shape {geom['shape']}")
    act = data["actuation"]
    if act.get("joint", "fixed") not in JOINTS:
        raise ValueError(f"{data['name']}: unknown joint {act.get('joint')}")
    ports = {}
    for p in data["ports"]:
        if p["id"] in ports:
            raise ValueError(f"{data['name']}: duplicate port {p['id']}")
        ports[p["id"]] = Port(
            id=p["id"],
            position=np.asarray(p["position"], dtype=float),
            normal=_unit(p["normal"]),
            roll_options_deg=tuple(float(r) for r in p.get("roll_options_deg", [0.0])),
        )
    return ModuleEntry(
        name=data["name"],
        category=data["category"],
        description=data.get("description", ""),
        geometry=geom,
        actuation=act,
        sensing=data.get("sensing", []),
        ports=ports,
        policy=data.get("policy", {}),
        raw=data,
    )


class Library:
    """Module library loaded from JSON entries."""

    def __init__(self, root: str | Path = LIBRARY_DIR):
        root = Path(root)
        self.modules: dict[str, ModuleEntry] = {}
        self.assemblies: dict[str, AssemblyEntry] = {}
        self.hardware: dict[str, HardwareModule] = {}
        for f in sorted((root / "modules").glob("*.json")):
            m = parse_module(json.loads(f.read_text()))
            self.modules[m.name] = m
        for f in sorted((root / "assemblies").glob("*.json")):
            d = json.loads(f.read_text())
            self.assemblies[d["name"]] = AssemblyEntry(
                d["name"], d["category"], d.get("description", ""), d["attach_port"], d["genotype"], d
            )
        for f in sorted((root / "hardware").glob("*.json")):
            d = json.loads(f.read_text())
            self.hardware[d["name"]] = HardwareModule(
                d["name"], d["category"], d.get("description", ""), d["joints"], d
            )

    def __getitem__(self, name: str) -> ModuleEntry:
        return self.modules[name]

    def __contains__(self, name: str) -> bool:
        return name in self.modules

    def by_category(self, category: str) -> list[str]:
        return [n for n, m in self.modules.items() if m.category == category]

    @property
    def categories(self) -> list[str]:
        return sorted({m.category for m in self.modules.values()})


@lru_cache(maxsize=1)
def default_library() -> Library:
    return Library()
