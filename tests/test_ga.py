import numpy as np
import pytest

from robogene import ga
from robogene.config import load_config
from robogene.genotype import Genotype, Node, module_counts, validate_genotype
from robogene.schema import default_library

LIB = default_library()


def valid(g, cfg):
    return validate_genotype(g, LIB, cfg["checks"], cfg["ga"])[0]


@pytest.fixture(params=["configs/task1_locomotion.yaml", "configs/task2_loco_manip.yaml"])
def cfg(request):
    return load_config(request.param)


def test_random_genotype_valid(cfg):
    rng = np.random.default_rng(1)
    for _ in range(40):
        g = ga.random_genotype(LIB, cfg, rng)
        assert valid(g, cfg)
        assert g.ids == list(range(len(g)))
        counts = module_counts(g, LIB)
        if cfg["task"] == "loco_manip":
            assert counts["wheel"] >= 1 and counts["arm"] >= 1


def test_random_genotype_uses_assemblies():
    cfg = load_config("configs/task1_locomotion.yaml")
    rng = np.random.default_rng(2)
    sizes = [module_counts(ga.random_genotype(LIB, cfg, rng), LIB)["arm"] for _ in range(200)]
    assert max(sizes) >= 3


def test_operators_keep_genotypes_valid(cfg):
    rng = np.random.default_rng(3)
    pool = [ga.random_genotype(LIB, cfg, rng) for _ in range(20)]
    names = list(ga.OPERATORS)
    for i in range(300):
        g = pool[i % len(pool)]
        before = g.to_dict()
        out = ga.apply_operator(names[i % len(names)], g, LIB, cfg, rng)
        assert g.to_dict() == before
        if out is not None:
            assert valid(out, cfg)
            pool[i % len(pool)] = out
        m = ga.mutate(g, LIB, cfg, rng)
        assert valid(m, cfg)


def test_mutation_changes_genotypes():
    cfg = load_config("configs/task1_locomotion.yaml")
    rng = np.random.default_rng(4)
    g = ga.random_genotype(LIB, cfg, rng)
    hashes = {ga.mutate(g, LIB, cfg, rng).canonical_hash() for _ in range(30)}
    assert len(hashes - {g.canonical_hash()}) > 5


def test_perturb_limits_stays_in_range():
    cfg = load_config("configs/task1_locomotion.yaml")
    rng = np.random.default_rng(5)
    g = Genotype([Node(0, "body_small")], [], 0)
    g.add_node("hinge", 0, "+x", "in")
    for _ in range(50):
        out = ga.apply_operator("perturb_limits", g, LIB, cfg, rng)
        if out is not None:
            g = out
        assert 0.3 <= g.node(1).params.get("limit_scale", 1.0) <= 1.0


def test_crossover_mixes_parents():
    cfg = load_config("configs/task1_locomotion.yaml")
    g1 = Genotype([Node(0, "connector")], [], 0)
    g1.add_node("hinge", 0, "+x", "in")
    g1.add_node("hinge", 0, "-x", "in")
    g2 = Genotype([Node(0, "body_large")], [], 0)
    g2.add_node("wheel_left", 0, "+y", "hub")
    g2.add_node("wheel_left", 0, "-y", "hub")
    assert valid(g1, cfg) and valid(g2, cfg)
    rng = np.random.default_rng(6)
    child = ga.crossover(g1, g2, LIB, cfg, rng)
    assert child is not None and valid(child, cfg)
    modules = {n.module for n in child.nodes}
    assert "connector" in modules and "hinge" in modules and "wheel_left" in modules
    assert len(g1) == 3 and len(g2) == 3


def test_crossover_random_parents_valid(cfg):
    rng = np.random.default_rng(7)
    for _ in range(30):
        a, b = ga.random_genotype(LIB, cfg, rng), ga.random_genotype(LIB, cfg, rng)
        child = ga.crossover(a, b, LIB, cfg, rng)
        if child is not None:
            assert valid(child, cfg)


def test_tournament_select():
    rng = np.random.default_rng(8)
    fitness = [0.0, 5.0, -np.inf, 1.0]
    picks = [ga.tournament_select(fitness, 4, rng) for _ in range(10)]
    assert picks == [1] * 10
    picks = {ga.tournament_select(fitness, 1, rng) for _ in range(200)}
    assert picks == {0, 1, 2, 3}


def test_make_offspring_respects_check_fn(cfg):
    rng = np.random.default_rng(9)
    pop = [ga.random_genotype(LIB, cfg, rng) for _ in range(6)]
    fitness = list(rng.random(6))

    def no_hinges(g):
        return module_counts(g, LIB)["hinge"] == 0

    for _ in range(10):
        child = ga.make_offspring(pop, fitness, LIB, cfg, rng, no_hinges)
        assert valid(child, cfg) and no_hinges(child)
