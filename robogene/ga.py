"""Genetic operators: random sampling, tournament selection, subgraph crossover and mutation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .genotype import Genotype, Node, assembly_genotype, module_counts, validate_genotype
from .schema import Library

CheckFn = Callable[[Genotype], bool]

ROOT_CATEGORIES = ("body", "connector")
P_ARM_ROOT = 0.1
P_ASSEMBLY = 0.15
MAX_RANDOM_TRIES = 1000
OPERATOR_TRIES = 10


def is_valid(g: Genotype, library: Library, cfg: dict) -> bool:
    return validate_genotype(g, library, cfg["checks"], cfg["ga"])[0]


def _choice(rng: np.random.Generator, items: list):
    return items[int(rng.integers(len(items)))]


def _random_roll(library: Library, g: Genotype, parent: int, port: str, rng: np.random.Generator) -> float:
    options = library[g.node(parent).module].ports[port].roll_options_deg
    return float(_choice(rng, list(options)))


def _roll_admissible(library: Library, g: Genotype, parent: int, port: str, roll: float) -> bool:
    options = library[g.node(parent).module].ports[port].roll_options_deg
    return any(abs(((roll - r) + 180) % 360 - 180) < 1e-6 for r in options)


def assembly_size(library: Library, name: str) -> int:
    return len(library.assemblies[name].genotype["nodes"])


def attach_assembly(
    g: Genotype, library: Library, name: str, parent: int, parent_port: str, roll_deg: float
) -> int:
    """Attach a library assembly in place through its attach_port and return its first node id."""
    asm = assembly_genotype(library, name)
    first = asm.node(asm.root)
    new_root = g.add_node(first.module, parent, parent_port, library.assemblies[name].attach_port,
                          roll_deg, first.params)
    # graft needs a parent edge in the source, so the first segment is added by hand
    for e in asm.child_edges(asm.root):
        g.graft(asm, e.child, new_root, e.parent_port, e.roll_deg)
    return new_root


def _add_random_part(
    g: Genotype, library: Library, rng: np.random.Generator, room: int, category: str | None = None
) -> bool:
    slots = [(nid, p) for nid in g.ids for p in g.free_ports(nid, library)]
    if not slots or room < 1:
        return False
    parent, port = _choice(rng, slots)
    roll = _random_roll(library, g, parent, port, rng)
    if category is None:
        fitting = [a for a in sorted(library.assemblies) if assembly_size(library, a) <= room]
        if fitting and rng.random() < P_ASSEMBLY:
            attach_assembly(g, library, _choice(rng, fitting), parent, port, roll)
            return True
        category = _choice(rng, library.categories)
    module = _choice(rng, library.by_category(category))
    child_port = _choice(rng, list(library[module].ports))
    g.add_node(module, parent, port, child_port, roll)
    return True


def _target_size(cfg: dict, rng: np.random.Generator) -> int:
    ga = cfg["ga"]
    lo, hi = ga.get("init_modules", [ga["min_modules"], ga["max_modules"]])
    lo = max(int(lo), int(ga.get("min_modules", 1)))
    hi = min(int(hi), int(ga.get("max_modules", hi)))
    return int(rng.integers(lo, max(lo, hi) + 1))


def random_genotype(library: Library, cfg: dict, rng: np.random.Generator) -> Genotype:
    """Sample a valid genotype whose size is drawn from ga.init_modules."""
    ga = cfg["ga"]
    max_modules = int(ga.get("max_modules", 10))
    for _ in range(MAX_RANDOM_TRIES):
        size = _target_size(cfg, rng)
        root_cat = "arm" if rng.random() < P_ARM_ROOT else _choice(rng, list(ROOT_CATEGORIES))
        g = Genotype([Node(0, _choice(rng, library.by_category(root_cat)))], [], root=0)
        while len(g) < size and _add_random_part(g, library, rng, size - len(g)):
            pass
        counts = module_counts(g, library)
        if ga.get("require_wheel") and counts["wheel"] == 0:
            _add_random_part(g, library, rng, max_modules - len(g), category="wheel")
        if ga.get("require_arm") and counts["arm"] == 0:
            _add_random_part(g, library, rng, max_modules - len(g), category="arm")
        if is_valid(g, library, cfg):
            return g.relabel()
    raise RuntimeError("could not sample a valid genotype; check the ga and checks settings")


def tournament_select(fitness: list[float], k: int, rng: np.random.Generator) -> int:
    """Index of the best of k members drawn at random from the whole population."""
    picks = rng.choice(len(fitness), size=min(k, len(fitness)), replace=False)
    return int(max(picks, key=lambda i: (fitness[i], -i)))


def crossover(g1: Genotype, g2: Genotype, library: Library, cfg: dict, rng: np.random.Generator,
              tries: int = OPERATOR_TRIES) -> Genotype | None:
    """Subgraph exchange: replace a subtree of g1 by a subtree of g2 at the same parent port."""
    a_nodes = [n for n in g1.ids if n != g1.root]
    b_nodes = [n for n in g2.ids if n != g2.root]
    if not a_nodes or not b_nodes:
        return None
    for _ in range(tries):
        a, b = _choice(rng, a_nodes), _choice(rng, b_nodes)
        child = g1.copy()
        e = child.parent_edge(a)
        parent, port = e.parent, e.parent_port
        roll = e.roll_deg
        child.remove_subtree(a)
        if not _roll_admissible(library, child, parent, port, roll):
            roll = _random_roll(library, child, parent, port, rng)
        child.graft(g2, b, parent, port, roll)
        if is_valid(child, library, cfg):
            return child.relabel()
    return None


def _substitute(g: Genotype, library: Library, cfg: dict, rng: np.random.Generator) -> bool:
    nid = _choice(rng, g.ids)
    node = g.node(nid)
    category = library[node.module].category
    options = [m for m in library.by_category(category) if m != node.module]
    if not options:
        return False
    new = _choice(rng, options)
    if not g.used_ports(nid) <= set(library[new].ports):
        return False
    node.module = new
    return True


def _reassign_port(g: Genotype, library: Library, cfg: dict, rng: np.random.Generator) -> bool:
    movable = [n for n in g.ids if n != g.root]
    if not movable:
        return False
    e = g.parent_edge(_choice(rng, movable))
    free = g.free_ports(e.parent, library)
    if free and rng.random() < 0.5:
        e.parent_port = _choice(rng, free)
        if not _roll_admissible(library, g, e.parent, e.parent_port, e.roll_deg):
            e.roll_deg = _random_roll(library, g, e.parent, e.parent_port, rng)
        return True
    options = library[g.node(e.parent).module].ports[e.parent_port].roll_options_deg
    others = [float(r) for r in options if abs(((r - e.roll_deg) + 180) % 360 - 180) > 1e-6]
    if not others:
        return False
    e.roll_deg = _choice(rng, others)
    return True


def _add_leaf(g: Genotype, library: Library, cfg: dict, rng: np.random.Generator) -> bool:
    room = int(cfg["ga"].get("max_modules", 10)) - len(g)
    return _add_random_part(g, library, rng, room)


def _remove_leaf(g: Genotype, library: Library, cfg: dict, rng: np.random.Generator) -> bool:
    leaves = g.leaves()
    if not leaves:
        return False
    leaf = _choice(rng, leaves)
    if library[g.node(leaf).module].category == "arm" and rng.random() < 0.5:
        start = leaf
        while True:
            pe = g.parent_edge(start)
            if pe is None or pe.parent == g.root or library[g.node(pe.parent).module].category != "arm":
                break
            if len(g.children(pe.parent)) != 1:
                break
            start = pe.parent
        if start != g.root:
            leaf = start
    g.remove_subtree(leaf)
    return True


def _perturb_limits(g: Genotype, library: Library, cfg: dict, rng: np.random.Generator) -> bool:
    jointed = [n.id for n in g.nodes if library[n.module].category in ("hinge", "arm")]
    if not jointed:
        return False
    node = g.node(_choice(rng, jointed))
    scale = float(node.params.get("limit_scale", 1.0)) * float(rng.uniform(0.8, 1.25))
    new = round(float(np.clip(scale, 0.3, 1.0)), 3)
    if new == node.params.get("limit_scale", 1.0):
        return False
    node.params["limit_scale"] = new
    return True


OPERATORS = {
    "substitute": _substitute,
    "reassign_port": _reassign_port,
    "add_leaf": _add_leaf,
    "remove_leaf": _remove_leaf,
    "perturb_limits": _perturb_limits,
}


def apply_operator(name: str, g: Genotype, library: Library, cfg: dict, rng: np.random.Generator,
                   tries: int = OPERATOR_TRIES) -> Genotype | None:
    """Apply one mutation operator; return a valid mutated copy or None."""
    for _ in range(tries):
        out = g.copy()
        if OPERATORS[name](out, library, cfg, rng) and is_valid(out, library, cfg):
            return out.relabel()
    return None


def mutate(g: Genotype, library: Library, cfg: dict, rng: np.random.Generator) -> Genotype:
    """Apply one operator drawn by ga.mutation_weights; try the others if it fails."""
    weights = {k: float(v) for k, v in cfg["ga"]["mutation_weights"].items() if float(v) > 0 and k in OPERATORS}
    names = list(weights)
    while names:
        p = np.array([weights[n] for n in names])
        name = names[int(rng.choice(len(names), p=p / p.sum()))]
        out = apply_operator(name, g, library, cfg, rng)
        if out is not None:
            return out
        names.remove(name)
    return g.copy()


def phenotype_check(g: Genotype, cfg: dict, library: Library | None = None) -> bool:
    """Default offspring filter: expand the genotype and run check_phenotype."""
    from . import phenotype

    try:
        ph = phenotype.expand(g, library)
        return bool(phenotype.check_phenotype(ph, cfg)[0])
    except NotImplementedError:
        raise
    except Exception:
        return False


def make_offspring(
    population: list[Genotype],
    fitness: list[float],
    library: Library,
    cfg: dict,
    rng: np.random.Generator,
    check_fn: CheckFn | None = None,
) -> Genotype:
    """One offspring: tournament selection, crossover, mutation, kept only when valid."""
    ga = cfg["ga"]
    if check_fn is None:
        def check_fn(g: Genotype) -> bool:
            return phenotype_check(g, cfg, library)

    for _ in range(int(ga["max_offspring_attempts"])):
        p1 = population[tournament_select(fitness, ga["tournament_size"], rng)]
        child = None
        if rng.random() < ga["crossover_prob"]:
            p2 = population[tournament_select(fitness, ga["tournament_size"], rng)]
            child = crossover(p1, p2, library, cfg, rng)
        if child is None:
            child = p1.copy()
        if rng.random() < ga["mutation_prob"]:
            child = mutate(child, library, cfg, rng)
        if is_valid(child, library, cfg) and check_fn(child):
            return child.relabel()
    # fall back to a fresh sample so the population keeps its size
    for _ in range(MAX_RANDOM_TRIES):
        g = random_genotype(library, cfg, rng)
        if check_fn(g):
            return g
    raise RuntimeError("could not make a viable offspring")
