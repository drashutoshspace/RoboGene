"""Genotype graph: a tree of module instances joined at their connection ports."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from .schema import Library

@dataclass
class Node:
    id: int
    module: str
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"limit_scale": 0.8}


@dataclass
class Edge:
    parent: int
    parent_port: str
    child: int
    child_port: str
    roll_deg: float = 0.0


@dataclass
class Genotype:
    nodes: list[Node]
    edges: list[Edge]
    root: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genotype":
        return cls(
            nodes=[Node(int(n["id"]), n["module"], dict(n.get("params", {}))) for n in d["nodes"]],
            edges=[
                Edge(int(e["parent"]), e["parent_port"], int(e["child"]), e["child_port"], float(e.get("roll_deg", 0.0)))
                for e in d["edges"]
            ],
            root=int(d.get("root", d["nodes"][0]["id"])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Genotype":
        return cls.from_dict(json.loads(s))

    def copy(self) -> "Genotype":
        return copy.deepcopy(self)

    @property
    def ids(self) -> list[int]:
        return [n.id for n in self.nodes]

    def node(self, nid: int) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def parent_edge(self, nid: int) -> Edge | None:
        for e in self.edges:
            if e.child == nid:
                return e
        return None

    def children(self, nid: int) -> list[int]:
        return [e.child for e in self.edges if e.parent == nid]

    def child_edges(self, nid: int) -> list[Edge]:
        return [e for e in self.edges if e.parent == nid]

    def subtree(self, nid: int) -> list[int]:
        """Node ids of the subtree rooted at nid, depth-first."""
        out, stack = [], [nid]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(reversed(self.children(cur)))
        return out

    def leaves(self) -> list[int]:
        return [n.id for n in self.nodes if not self.children(n.id) and n.id != self.root]

    def used_ports(self, nid: int) -> set[str]:
        used = {e.parent_port for e in self.edges if e.parent == nid}
        pe = self.parent_edge(nid)
        if pe is not None:
            used.add(pe.child_port)
        return used

    def free_ports(self, nid: int, library: Library) -> list[str]:
        mod = library[self.node(nid).module]
        used = self.used_ports(nid)
        return [p for p in mod.ports if p not in used]

    def depth(self, nid: int) -> int:
        d = 0
        e = self.parent_edge(nid)
        while e is not None:
            d += 1
            e = self.parent_edge(e.parent)
        return d

    def next_id(self) -> int:
        return max(self.ids, default=-1) + 1

    def __len__(self) -> int:
        return len(self.nodes)

    def add_node(
        self, module: str, parent: int, parent_port: str, child_port: str, roll_deg: float = 0.0,
        params: dict | None = None,
    ) -> int:
        nid = self.next_id()
        self.nodes.append(Node(nid, module, dict(params or {})))
        self.edges.append(Edge(parent, parent_port, nid, child_port, float(roll_deg)))
        return nid

    def remove_subtree(self, nid: int) -> list[int]:
        if nid == self.root:
            raise ValueError("cannot remove the root")
        gone = set(self.subtree(nid))
        self.nodes = [n for n in self.nodes if n.id not in gone]
        self.edges = [e for e in self.edges if e.child not in gone and e.parent not in gone]
        return sorted(gone)

    def graft(self, other: "Genotype", other_root: int, parent: int, parent_port: str, roll_deg: float) -> dict[int, int]:
        """Copy the subtree of other rooted at other_root under parent_port; returns the id map."""
        pe = other.parent_edge(other_root)
        child_port = pe.child_port if pe is not None else None
        if child_port is None:
            raise ValueError("graft source must not be the root of `other`")
        idmap: dict[int, int] = {}
        for oid in other.subtree(other_root):
            nid = self.next_id()
            idmap[oid] = nid
            src = other.node(oid)
            self.nodes.append(Node(nid, src.module, dict(src.params)))
        self.edges.append(Edge(parent, parent_port, idmap[other_root], child_port, float(roll_deg)))
        for e in other.edges:
            if e.parent in idmap and e.child in idmap:
                self.edges.append(Edge(idmap[e.parent], e.parent_port, idmap[e.child], e.child_port, e.roll_deg))
        return idmap

    def relabel(self) -> "Genotype":
        """Copy with node ids renumbered depth-first from 0."""
        order = self.subtree(self.root)
        idmap = {old: new for new, old in enumerate(order)}
        nodes = [Node(idmap[n.id], n.module, dict(n.params)) for n in sorted(self.nodes, key=lambda n: idmap[n.id])]
        edges = [Edge(idmap[e.parent], e.parent_port, idmap[e.child], e.child_port, e.roll_deg) for e in self.edges]
        edges.sort(key=lambda e: e.child)
        return Genotype(nodes, edges, root=0)

    def canonical_hash(self) -> str:
        """Hash of the morphology, independent of node numbering."""

        def encode(nid: int) -> str:
            n = self.node(nid)
            parts = []
            for e in sorted(self.child_edges(nid), key=lambda e: (e.parent_port, e.child_port, e.roll_deg)):
                parts.append(f"{e.parent_port}>{e.child_port}@{e.roll_deg:g}:{encode(e.child)}")
            params = json.dumps(n.params, sort_keys=True)
            return f"{n.module}{params}[{','.join(parts)}]"

        return hashlib.sha1(encode(self.root).encode()).hexdigest()[:16]


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R a = b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * k @ k


def connection_rotation(n_parent: np.ndarray, n_child: np.ndarray, roll_deg: float) -> np.ndarray:
    """Child frame rotation: child normal onto the reversed parent normal, then roll about it."""
    r_align = _rotation_between(np.asarray(n_child, float), -np.asarray(n_parent, float))
    return axis_angle(n_parent, np.deg2rad(roll_deg)) @ r_align


def edge_transform(g: Genotype, e: Edge, library: Library) -> tuple[np.ndarray, np.ndarray]:
    """Pose (R, t) of the child module frame in the parent module frame."""
    pmod = library[g.node(e.parent).module]
    cmod = library[g.node(e.child).module]
    pp = pmod.ports[e.parent_port]
    cp = cmod.ports[e.child_port]
    R = connection_rotation(pp.normal, cp.normal, e.roll_deg)
    t = pp.position - R @ cp.position
    return R, t


def alignment_error(n_parent: np.ndarray, R: np.ndarray, n_child: np.ndarray) -> float:
    """Zero when the port normals are exactly antiparallel."""
    return float(np.linalg.norm(np.asarray(n_parent) + R @ np.asarray(n_child)))


def align_tolerance(angle_deg: float) -> float:
    """alignment_error tolerance for an angular misalignment."""
    return 2.0 * np.sin(np.deg2rad(angle_deg) / 2.0)


def forward_kinematics(g: Genotype, library: Library) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """World pose of every module at zero joint angles, root at the origin."""
    poses = {g.root: (np.eye(3), np.zeros(3))}
    for nid in g.subtree(g.root)[1:]:
        e = g.parent_edge(nid)
        Rp, pp = poses[e.parent]
        R, t = edge_transform(g, e, library)
        poses[nid] = (Rp @ R, pp + Rp @ t)
    return poses


def adjacency_matrix(g: Genotype) -> np.ndarray:
    """Symmetric adjacency matrix in the order of g.ids."""
    index = {nid: i for i, nid in enumerate(g.ids)}
    A = np.zeros((len(g), len(g)), dtype=int)
    for e in g.edges:
        A[index[e.parent], index[e.child]] = 1
        A[index[e.child], index[e.parent]] = 1
    return A


def validate_genotype(
    g: Genotype,
    library: Library,
    checks: dict[str, Any],
    ga: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Check modules, ports, tree structure, alignment, size and coarse collisions."""
    reasons: list[str] = []
    ids = g.ids
    if len(set(ids)) != len(ids):
        reasons.append("duplicate node ids")
    for n in g.nodes:
        if n.module not in library:
            reasons.append(f"unknown module {n.module}")
    if reasons:
        return False, reasons
    if g.root not in ids:
        return False, ["root missing"]

    parents: dict[int, int] = {}
    for e in g.edges:
        if e.parent not in ids or e.child not in ids:
            reasons.append(f"edge to missing node {e.parent}->{e.child}")
            continue
        if e.child == g.root:
            reasons.append("edge into root")
        if e.child in parents:
            reasons.append(f"node {e.child} has two parents")
        parents[e.child] = e.parent
    if checks.get("connectivity", True):
        reach = set()
        stack = [g.root]
        while stack:
            cur = stack.pop()
            if cur in reach:
                reasons.append("cycle")
                break
            reach.add(cur)
            stack.extend(e.child for e in g.edges if e.parent == cur)
        if reach != set(ids):
            reasons.append("graph not connected")
    if reasons:
        return False, reasons

    if checks.get("ports", True):
        tol = align_tolerance(checks.get("align_tol_deg", 5.0))
        for nid in ids:
            mod = library[g.node(nid).module]
            used = [e.parent_port for e in g.edges if e.parent == nid]
            pe = g.parent_edge(nid)
            if pe is not None:
                used.append(pe.child_port)
            for p in used:
                if p not in mod.ports:
                    reasons.append(f"node {nid} ({mod.name}) has no port {p}")
            if len(used) != len(set(used)):
                reasons.append(f"node {nid} uses a port twice")
        if not reasons:
            for e in g.edges:
                pmod = library[g.node(e.parent).module]
                cmod = library[g.node(e.child).module]
                pp, cp = pmod.ports[e.parent_port], cmod.ports[e.child_port]
                if not any(abs(((e.roll_deg - r) + 180) % 360 - 180) < 1e-6 for r in pp.roll_options_deg):
                    reasons.append(f"roll {e.roll_deg} not admissible at {pmod.name}.{pp.id}")
                R, _ = edge_transform(g, e, library)
                if not np.allclose(R.T @ R, np.eye(3), atol=1e-8) or np.linalg.det(R) < 0:
                    reasons.append("connection rotation not orthonormal")
                if alignment_error(pp.normal, R, cp.normal) > tol:
                    reasons.append(f"misaligned ports {pmod.name}.{pp.id} - {cmod.name}.{cp.id}")

    if ga is not None:
        if len(g) < ga.get("min_modules", 1):
            reasons.append("too few modules")
        if len(g) > ga.get("max_modules", 10**9):
            reasons.append("too many modules")
        counts = module_counts(g, library)
        if ga.get("require_wheel") and counts["wheel"] == 0:
            reasons.append("no wheel")
        if ga.get("require_arm") and counts["arm"] == 0:
            reasons.append("no arm")

    if checks.get("collision", True) and not reasons:
        poses = forward_kinematics(g, library)
        adjacent = {(e.parent, e.child) for e in g.edges} | {(e.child, e.parent) for e in g.edges}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if (a, b) in adjacent:
                    continue
                ma, mb = library[g.node(a).module], library[g.node(b).module]
                d = np.linalg.norm(poses[a][1] - poses[b][1])
                if d < 0.5 * (_inner_radius(ma) + _inner_radius(mb)):
                    reasons.append(f"modules {a} and {b} overlap")
    return (not reasons), reasons


def _inner_radius(mod) -> float:
    s = mod.size
    if mod.shape == "box":
        return float(min(s))
    return float(s[0])


def module_counts(g: Genotype, library: Library) -> dict[str, int]:
    counts = {c: 0 for c in ("body", "connector", "hinge", "arm", "wheel")}
    for n in g.nodes:
        cat = library[n.module].category
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def arm_chains(g: Genotype, library: Library) -> list[list[int]]:
    """Chains of arm segments, each starting at a segment whose parent is not an arm."""
    chains = []
    for n in g.nodes:
        if library[n.module].category != "arm":
            continue
        pe = g.parent_edge(n.id)
        if pe is not None and library[g.node(pe.parent).module].category == "arm":
            continue
        chain = [n.id]
        cur = n.id
        while True:
            nxt = [c for c in g.children(cur) if library[g.node(c).module].category == "arm"]
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)
        chains.append(chain)
    return chains


def morphology_class(g: Genotype, library: Library) -> str:
    """Class name such as "3Arm+4Wheel"; without wheels, three or more arm chains count as "Legged"."""
    c = module_counts(g, library)
    arms, wheels = c["arm"], c["wheel"]
    if wheels == 0 and arms == 0:
        return "Body Only"
    if wheels == 0:
        return "Legged" if len(arm_chains(g, library)) >= 3 else "Arm Only"
    if arms == 0:
        return f"{wheels}Wheel"
    return f"{arms}Arm+{wheels}Wheel"


def assembly_genotype(library: Library, name: str) -> Genotype:
    """Genotype of a library assembly."""
    return Genotype.from_dict(library.assemblies[name].genotype)


def iter_connections(g: Genotype) -> Iterable[Edge]:
    """Edges in depth-first order."""
    for nid in g.subtree(g.root)[1:]:
        yield g.parent_edge(nid)
